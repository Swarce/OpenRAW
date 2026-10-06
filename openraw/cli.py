"""
Command-line interface: single files, many files, or whole folders.

    openraw photo.jpg                      -> photo.dng next to it
    openraw photo.jpg out.dng              -> explicit output file (legacy form)
    openraw a.jpg b.jpg -o converted/      -> into a folder
    openraw ~/Pictures/trip -r -o out/     -> whole tree, structure mirrored
    openraw folder/ --jobs 2               -> two files at a time

Batch behavior, chosen for long runs: existing outputs are skipped unless
--overwrite (re-running a half-finished batch resumes instead of redoing
work); one failing file is reported but never stops the batch; the exit
code is 1 if anything failed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from .pipeline import PipelineConfig, OpenRawPipeline

JPEG_EXTS = {".jpg", ".jpeg"}


def _version_report() -> str:
    """Package + key library versions: almost every bug this project has hit
    was library-version-dependent, so bug reports should include this."""
    import platform
    from . import __version__
    lines = [f"OpenRAW {__version__}", f"python {platform.python_version()} on {platform.system()} {platform.machine()}"]
    for mod in ("numpy", "cv2", "PIL", "tifffile", "imagecodecs", "rawpy", "torch"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            if mod == "imagecodecs":
                try:
                    v += f" ({m.jpeg8_version()})"
                except Exception:
                    pass
            lines.append(f"{mod} {v}")
        except Exception:
            lines.append(f"{mod} not installed")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openraw",
        description="Reconstruct pseudo-RAW linear DNGs from JPEGs (part of the OpenRAW project).",
    )
    p.add_argument("--version", action="store_true",
                   help="show package and library versions (include this in bug reports)")
    p.add_argument("inputs", nargs="+", help="JPEG files and/or folders")
    p.add_argument("-o", "--output", help="output folder (or output .dng when converting exactly one file)")
    p.add_argument("-r", "--recursive", action="store_true", help="search folders recursively (output mirrors the tree)")
    p.add_argument("--overwrite", action="store_true", help="replace existing .dng outputs (default: skip them)")
    p.add_argument("-j", "--jobs", type=int, default=1,
                   help="files converted in parallel (default 1; each 18MP image needs a few GB of RAM)")
    p.add_argument("--threads", type=int, default=0,
                   help="encoding threads per file (default: all cores, divided among --jobs)")
    p.add_argument("-q", "--quiet", action="store_true", help="only print errors and the final summary")

    g = p.add_argument_group("output DNG")
    g.add_argument("--compression", choices=["ljpeg", "none"], default="ljpeg",
                   help="ljpeg (default): lossless-JPEG tiles, verified in Adobe's DNG SDK and libraw. "
                        "none: uncompressed, largest.")
    g.add_argument("--bit-depth", type=int, choices=[16, 14, 12, 10], default=12,
                   help="ljpeg only. 12 (default) ~half the size of uncompressed, error <=1/20 of a source "
                        "8-bit step; 10 smaller still (<=1/7 step); 16 bit-exact.")
    g.add_argument("--no-exif", action="store_true",
                   help="don't carry camera metadata (Make/Model/lens/exposure...) into the DNG. GPS is never carried.")
    g.add_argument("--no-preview", action="store_true", help="skip the embedded JPEG preview")
    g.add_argument("--preview-max-dim", type=int, default=1024)
    g.add_argument("--preview-quality", type=int, default=90)

    r = p.add_argument_group("reconstruction")
    r.add_argument("--no-deblock", action="store_true")
    r.add_argument("--no-chroma-refine", action="store_true")
    r.add_argument("--chroma-strength", type=float, default=0.6)
    r.add_argument("--no-dither", action="store_true")
    r.add_argument("--no-deband", action="store_true")
    r.add_argument("--s-curve", type=float, default=0.0,
                   help="EXPERIMENTAL generic contrast-curve removal (0 = off; try 0.1-0.4)")
    r.add_argument("--gamut-widen", type=float, default=0.0,
                   help="EXPERIMENTAL gamut widening -- a look, not a reconstruction (0 = off)")

    m = p.add_argument_group("InvISP network (needs torch: pip install 'openraw[invisp]')")
    m.add_argument("--invisp", action="store_true", help="use the InvISP network instead of the classical stages")
    m.add_argument("--invisp-camera", choices=["NIKON_D700", "Canon_EOS_5D"], default="NIKON_D700")
    m.add_argument("--invisp-pretrained-dir", default="pretrained")
    m.add_argument("--invisp-device", default="cpu", help="'cpu' or 'cuda:0' etc.")
    return p


def _config_from_args(a) -> PipelineConfig:
    return PipelineConfig(
        deblock_enabled=not a.no_deblock,
        chroma_refine_enabled=not a.no_chroma_refine,
        chroma_refine_strength=a.chroma_strength,
        generic_s_curve_strength=a.s_curve,
        experimental_gamut_widen=a.gamut_widen,
        dither=not a.no_dither,
        deband=not a.no_deband,
        use_invisp=a.invisp,
        invisp_camera=a.invisp_camera,
        invisp_pretrained_dir=a.invisp_pretrained_dir,
        invisp_device=a.invisp_device,
        dng_compression=a.compression,
        dng_bit_depth=a.bit_depth,
        encode_threads=a.threads or None,
        preserve_exif=not a.no_exif,
        write_preview=not a.no_preview,
        preview_max_dim=a.preview_max_dim,
        preview_quality=a.preview_quality,
    )


def plan_jobs(inputs: list[str], output: str | None, recursive: bool) -> list[tuple[Path, Path]]:
    """Expand files/folders into (source_jpeg, destination_dng) pairs.

    Raises ValueError for unusable arguments (missing paths, a .dng output
    with several inputs). Duplicate sources are dropped; two sources mapping
    to the same destination is an error rather than a silent overwrite.
    """
    out = Path(output) if output else None
    single_file_out = out is not None and out.suffix.lower() == ".dng"

    sources: list[tuple[Path, Path | None]] = []  # (file, base folder it came from)
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            for f in sorted(it):
                if f.is_file() and f.suffix.lower() in JPEG_EXTS:
                    sources.append((f, p))
        elif p.is_file():
            if p.suffix.lower() not in JPEG_EXTS:
                raise ValueError(f"not a JPEG (.jpg/.jpeg): {p}")
            sources.append((p, None))
        else:
            raise ValueError(f"no such file or folder: {p}")

    seen, uniq = set(), []
    for f, base in sources:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            uniq.append((f, base))

    if single_file_out:
        if len(uniq) != 1:
            raise ValueError(f"output '{out}' is a single .dng file but {len(uniq)} JPEGs were given; "
                             f"pass a folder with -o instead")
        return [(uniq[0][0], out)]

    plan, dests = [], {}
    for f, base in uniq:
        if out is None:
            dest = f.with_suffix(".dng")
        elif base is not None:
            dest = out / f.relative_to(base).with_suffix(".dng")
        else:
            dest = out / (f.stem + ".dng")
        k = dest.resolve()
        if k in dests:
            raise ValueError(f"two inputs would both write {dest}: {dests[k]} and {f}")
        dests[k] = f
        plan.append((f, dest))
    return plan


def _convert_one(src: str, dst: str, config: PipelineConfig) -> dict:
    """Runs in a worker process (or inline for --jobs 1). Never raises."""
    t0 = time.time()
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        tmp = dst + ".partial"  # write-then-rename: an interrupted run never leaves a truncated .dng
        result = OpenRawPipeline(config).run_to_dng(src, tmp)
        os.replace(tmp, dst)
        d = result.decoded
        n_exif = len(d.exif_fields.get("ifd0", {})) + len(d.exif_fields.get("exif_sub", {})) if d.exif_fields else 0
        return dict(ok=True, src=src, dst=dst, secs=time.time() - t0, size=os.path.getsize(dst),
                    w=d.width, h=d.height, q=d.quality_estimate, cq=d.chroma_quality_estimate,
                    sub=d.is_chroma_subsampled, exif=n_exif)
    except Exception as e:  # report, keep the batch going
        try:
            os.remove(dst + ".partial")
        except OSError:
            pass
        return dict(ok=False, src=src, dst=dst, secs=time.time() - t0, error=f"{type(e).__name__}: {e}")


def _print_detail(r: dict) -> None:
    print(f"input:           {r['src']}  ({r['w']}x{r['h']})")
    print(f"luma quality:    {r['q']:.1f} / 100  (drives deblock strength)")
    if r["cq"] is not None:
        sub = "subsampled" if r["sub"] else "4:4:4, NOT subsampled"
        print(f"chroma quality:  {r['cq']:.1f} / 100  ({sub})")
    print(f"camera metadata: {r['exif']} field(s)" if r["exif"] else "camera metadata: none found in source JPEG")
    print(f"output:          {r['dst']}  ({r['size'] / 1e6:.1f} MB)")
    print(f"elapsed:         {r['secs']:.2f}s")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--version" in argv:  # handled before argparse: lazy (imports torch etc.) and keeps line breaks
        print(_version_report())
        return 0
    # Legacy form: `openraw in.jpg out.dng` (two positionals, second a .dng).
    if len(argv) >= 2 and not argv[1].startswith("-") and argv[1].lower().endswith(".dng") \
            and "-o" not in argv and "--output" not in argv:
        argv = [argv[0], "-o", argv[1]] + argv[2:]

    args = _build_parser().parse_args(argv)
    try:
        plan = plan_jobs(args.inputs, args.output, args.recursive)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not plan:
        print("no JPEGs found" + ("" if args.recursive else " (use -r to search subfolders)"), file=sys.stderr)
        return 1

    todo = [(s, d) for s, d in plan if args.overwrite or not d.exists()]
    skipped = len(plan) - len(todo)
    jobs = max(1, min(args.jobs, len(todo) or 1))
    config = _config_from_args(args)
    if not args.threads and jobs > 1:  # split cores among parallel files
        config = replace(config, encode_threads=max(1, (os.cpu_count() or 1) // jobs))

    single = len(plan) == 1
    if not args.quiet and not single:
        print(f"{len(plan)} JPEG(s): {len(todo)} to convert, {skipped} already done (--overwrite to redo), jobs={jobs}")
    if single and skipped:
        print(f"skipped: {plan[0][1]} exists (use --overwrite)")
        return 0

    results, t0 = [], time.time()
    if jobs == 1:
        it = (_convert_one(str(s), str(d), config) for s, d in todo)
    else:
        pool = ProcessPoolExecutor(max_workers=jobs)
        futs = [pool.submit(_convert_one, str(s), str(d), config) for s, d in todo]
        it = (f.result() for f in as_completed(futs))
    try:
        for i, r in enumerate(it, 1):
            results.append(r)
            if single and r["ok"] and not args.quiet:
                _print_detail(r)
            elif not r["ok"]:
                print(f"[{i}/{len(todo)}] FAILED {r['src']}: {r['error']}", file=sys.stderr)
            elif not args.quiet:
                print(f"[{i}/{len(todo)}] {r['src']} -> {r['dst']}  ({r['size'] / 1e6:.1f} MB, {r['secs']:.1f}s)")
    finally:
        if jobs > 1:
            pool.shutdown(cancel_futures=True)

    failed = [r for r in results if not r["ok"]]
    if not single or failed:
        ok = len(results) - len(failed)
        print(f"done in {time.time() - t0:.1f}s: {ok} converted, {skipped} skipped, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
