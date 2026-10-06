#!/usr/bin/env python3
"""
Download MIT-Adobe FiveK images filtered by camera model.

Metadata (which image belongs to which camera) comes from the JSON files
published by yuukicammy/MIT-Adobe-FiveK on Hugging Face (training/validation/
testing.json, 5000 images total). The images themselves are downloaded
straight from the official MIT server:
    https://data.csail.mit.edu/graphics/fivek/img/dng/<name>.dng
    https://data.csail.mit.edu/graphics/fivek/img/tiff16_{a..e}/<name>.tif

Only the Python standard library is required (tqdm is used if installed).

Examples
--------
  # see which cameras exist and how many images each has
  python fivek_download.py --list

  # download every DNG shot with a Nikon D70
  python fivek_download.py --camera "Nikon D70" --out ./fivek

  # several cameras at once (repeat --camera or use commas)
  python fivek_download.py --camera "Nikon D70" --camera "Canon EOS 5D"

  # also grab the Expert C retouched TIFFs (16-bit, ~100 MB each!)
  python fivek_download.py --camera "Leica M8" --experts c

  # only the training split, 8 parallel downloads, dry run first
  python fivek_download.py --camera "Nikon D70" --split train -j 8 --dry-run
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    tqdm = None

METADATA_BASE = "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/raw/main"
SPLIT_FILES = {
    "train": "training.json",
    "val": "validation.json",
    "test": "testing.json",
}
USER_AGENT = "fivek-downloader/1.0 (+https://github.com/Swarce/OpenRAW)"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def norm(s: str) -> str:
    """Lowercase and strip everything that isn't a letter/digit: 'EOS-1D Mark II' -> 'eos1dmarkii'."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def folder_name(make: str, model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{make} {model}").strip("_")


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def load_metadata(cache_dir: Path, splits, metadata_dir: Path = None):
    """Return {basename: entry} merged over the requested splits, tagging each with its split."""
    items = {}
    for split in splits:
        fname = SPLIT_FILES[split]
        if metadata_dir:
            path = metadata_dir / fname
            data = json.loads(path.read_text())
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / fname
            if not path.exists():
                print(f"[meta] downloading {fname} ...")
                path.write_bytes(http_get(f"{METADATA_BASE}/{fname}"))
            data = json.loads(path.read_text())
        for name, entry in data.items():
            entry["_split"] = split
            items[name] = entry
    return items


def camera_label(entry) -> str:
    cam = entry["camera"]
    return f"{cam['make']} {cam['model']}".strip()


def build_camera_index(items):
    """label -> list of basenames"""
    idx = {}
    for name, e in items.items():
        idx.setdefault(camera_label(e), []).append(name)
    return idx


def resolve_cameras(queries, index, contains=False):
    """Map user queries to camera labels. Exact (normalised) match by default."""
    chosen = []
    for q in queries:
        nq = norm(q)
        hits = []
        for label in index:
            make, _, model = label.partition(" ")
            if contains:
                if nq in norm(label):
                    hits.append(label)
            elif nq in (norm(label), norm(model)):
                hits.append(label)
        if not hits:
            near = [l for l in index if nq in norm(l)]
            msg = f"No camera matches {q!r}."
            if near:
                msg += " Did you mean: " + ", ".join(sorted(near)) + " ? (or pass --contains)"
            else:
                msg += " Use --list to see available cameras."
            sys.exit(msg)
        if len(hits) > 1 and not contains:
            sys.exit(f"{q!r} is ambiguous: {', '.join(sorted(hits))}. Use the full 'Make Model' name.")
        for h in hits:
            if h not in chosen:
                chosen.append(h)
    return chosen


# --------------------------------------------------------------------------- #
# downloading
# --------------------------------------------------------------------------- #
def _open(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def download(url: str, dest: Path, retries: int = 4, timeout: int = 60) -> str:
    """Download url -> dest atomically (via .part). Returns 'ok', 'skipped' or raises."""
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_err = None
    # try the URL as published, then the opposite scheme as a fallback
    alt = url.replace("http://", "https://") if url.startswith("http://") else url.replace("https://", "http://")
    for candidate in (url, alt):
        for attempt in range(1, retries + 1):
            try:
                with _open(candidate, timeout) as r, open(tmp, "wb") as f:
                    expected = r.headers.get("Content-Length")
                    total = 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        total += len(chunk)
                if expected and total != int(expected):
                    raise IOError(f"incomplete download ({total}/{expected} bytes)")
                tmp.replace(dest)
                return "ok"
            except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
                last_err = e
                if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                    break  # no point retrying a 404 on this candidate
                time.sleep(min(2 ** attempt, 20))
    if tmp.exists():
        tmp.unlink()
    raise RuntimeError(f"{url}: {last_err}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Download MIT-Adobe FiveK images by camera model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[1] if "Examples" in __doc__ else "",
    )
    ap.add_argument("--list", action="store_true", help="list camera models with image counts and exit")
    ap.add_argument("-c", "--camera", action="append", default=[],
                    help="camera to download, e.g. 'Nikon D70' or just 'D70'. Repeatable / comma-separated. "
                         "Omit to download ALL cameras (~50 GB of DNGs).")
    ap.add_argument("--contains", action="store_true",
                    help="substring matching instead of exact (so 'EOS' matches every Canon EOS)")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                    help="which official split to use (default: all 5000 images)")
    ap.add_argument("--experts", nargs="*", default=[], choices=list("abcde"),
                    help="also download the 16-bit retouched TIFFs from these experts (large!)")
    ap.add_argument("--no-dng", action="store_true", help="skip the RAW DNGs (only useful with --experts)")
    ap.add_argument("-o", "--out", default="./fivek", help="output directory (default: ./fivek)")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="parallel downloads (default: 4)")
    ap.add_argument("--limit", type=int, default=0, help="only fetch the first N images per camera (testing)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be downloaded and exit")
    ap.add_argument("--metadata-dir", type=Path, default=None,
                    help="use already-downloaded training/validation/testing.json from this folder")
    args = ap.parse_args()

    out = Path(args.out)
    splits = list(SPLIT_FILES) if args.split == "all" else [args.split]
    items = load_metadata(out / "_metadata", splits, args.metadata_dir)
    index = build_camera_index(items)

    if args.list:
        print(f"{'Camera':32s} {'Images':>6s}")
        print("-" * 40)
        for label, names in sorted(index.items(), key=lambda kv: -len(kv[1])):
            print(f"{label:32s} {len(names):6d}")
        print("-" * 40)
        print(f"{'Total':32s} {sum(len(v) for v in index.values()):6d}")
        return

    queries = [q.strip() for c in args.camera for q in c.split(",") if q.strip()]
    cameras = resolve_cameras(queries, index, args.contains) if queries else sorted(index)

    # build job list: (url, dest)
    jobs, manifest = [], {}
    for label in cameras:
        cam_dir = out / "raw" / folder_name(*label.split(" ", 1))
        names = sorted(index[label])
        if args.limit:
            names = names[: args.limit]
        manifest[label] = []
        for name in names:
            e = items[name]
            manifest[label].append({"name": name, "split": e["_split"], "license": e.get("license"),
                                    "categories": e.get("categories")})
            if not args.no_dng:
                jobs.append((e["urls"]["dng"], cam_dir / f"{name}.dng"))
            for ex in args.experts:
                jobs.append((e["urls"]["tiff16"][ex], out / "processed" / f"tiff16_{ex}" / f"{name}.tif"))

    print(f"Cameras selected: {len(cameras)}")
    for label in cameras:
        n = len(manifest[label])
        print(f"  {label:32s} {n:5d} images")
    print(f"Total files to fetch: {len(jobs)}  ->  {out.resolve()}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        print("Dry run, nothing downloaded. Selection written to selection.json")
        return

    ok = skipped = 0
    failed = []
    bar = tqdm(total=len(jobs), unit="file") if tqdm else None
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(download, u, d): (u, d) for u, d in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                status = fut.result()
                ok += status == "ok"
                skipped += status == "skipped"
            except Exception as e:  # noqa: BLE001
                failed.append(str(e))
            if bar:
                bar.update(1)
            elif i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    if bar:
        bar.close()

    print(f"Done: {ok} downloaded, {skipped} already present, {len(failed)} failed.")
    if failed:
        (out / "failed.txt").write_text("\n".join(failed))
        print("Failures written to failed.txt - just re-run the same command to retry them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
