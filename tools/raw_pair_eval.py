#!/usr/bin/env python3
"""Ground truth for OpenRAW: score its DNGs against the camera's REAL raw.

Give it photos shot as RAW + JPEG. For each pair it converts the camera's
JPEG with OpenRAW -- the classical pipeline and any InvISP checkpoints, each
as a linear and as a CFA (Bayer) DNG -- then renders every DNG *and the
camera's own raw file* with the same raw converter (LibRaw: camera white
balance, no auto-brightening, linear sRGB, no rotation), aligns them, and
measures how close each conversion comes to what the real raw renders to.

    python tools/raw_pair_eval.py PAIRS_DIR --out results/ \\
        --checkpoint pretrained/openraw-fivek-e53-best.pth --checkpoint pretrained/nikon.pth

PAIRS_DIR holds <name>.JPG next to <name>.ARW / .NEF / .CR2 / .DNG / ... .

Metrics, on the overlapping area (borders trimmed):
  PSNR          linear sRGB, as converted. Includes any overall exposure /
                white-balance offset: the camera's JPEG doesn't say how bright
                the raw was, so this is mostly a measure of that offset.
  PSNR (gain)   after one gain per colour channel, fitted by least squares --
                exposure and white balance matched, so what's left is tone,
                colour and detail. The fairest single number.
  PSNR (gain, display)
                the same, but compared after sRGB gamma encoding: weights
                shadows and midtones as a viewer sees them.

Writes results.csv, results.md and, with --crops, a side-by-side comparison
image per pair. DNGs are cached in --out/dng (re-runs skip finished ones).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_EXTS = (".arw", ".nef", ".cr2", ".cr3", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw")
BORDER = 24  # px trimmed from every side of the overlap before measuring


# --------------------------------------------------------------- pairs, DNGs
def find_pairs(folder) -> list[tuple[str, Path, Path]]:
    files = {p.name.lower(): p for p in Path(folder).iterdir() if p.is_file()}
    pairs = []
    for name, p in sorted(files.items()):
        stem, ext = os.path.splitext(name)
        if ext in (".jpg", ".jpeg"):
            raw = next((files[stem + e] for e in RAW_EXTS if stem + e in files), None)
            if raw is not None:
                pairs.append((p.stem, p, raw))
    return pairs


def methods_for(checkpoints) -> list[tuple[str, dict]]:
    """(name, PipelineConfig overrides). Each method yields a linear and a CFA DNG."""
    out = [("classical", {})]
    for c in checkpoints:
        out.append((f"InvISP {Path(c).stem}", {"use_invisp": True, "invisp_checkpoint": str(c)}))
    return out


def eval_region(jpeg: Path, size: int) -> tuple[int, int, int, int] | None:
    """(y, x, h, w) of the size x size region of the JPEG with the most fine
    detail -- in focus, so it shows what each method does to detail -- on the
    16 px block grid, so the JPEG's 8x8 blocks stay where they were. None =
    the whole frame."""
    if not size:
        return None
    from PIL import Image
    im = Image.open(jpeg).convert("RGB")
    W, H = im.size
    if size >= min(W, H):
        return None
    f = 4
    small = np.asarray(im.reduce(f), np.float32) / 255.0
    y, x = detailed_crop(np.power(small, 2.2), size // f)
    y, x = min(y * f, H - size) // 16 * 16, min(x * f, W - size) // 16 * 16
    return y, x, size, size


def convert(jpeg: Path, method: str, overrides: dict, dng_dir: Path, stem: str, region=None, log=print) -> dict:
    """Both layouts of one method for one photo (or a region of it); the pipeline runs once."""
    from openraw.decode import load_jpeg
    from openraw.dng_writer import write_linear_dng
    from openraw.pipeline import OpenRawPipeline, PipelineConfig, _PIPELINE_VERSION
    slug = method.replace(" ", "_") + (f"__r{region[0]}_{region[1]}_{region[2]}" if region else "")
    paths = {lay: dng_dir / f"{stem}__{slug}__{lay}.dng" for lay in ("linear", "cfa")}
    if all(p.exists() for p in paths.values()):
        return paths
    t = time.time()
    cfg = PipelineConfig(**overrides)
    decoded = load_jpeg(str(jpeg))
    if region:
        y, x, h, w = region
        decoded.rgb = np.ascontiguousarray(decoded.rgb[y:y + h, x:x + w])
        decoded.height, decoded.width = h, w
    res = OpenRawPipeline(cfg).run_decoded(decoded)
    for lay, p in paths.items():
        tmp = p.with_suffix(".part")
        write_linear_dng(str(tmp), res.rgb16, source_jpeg_path=str(jpeg), pipeline_version=_PIPELINE_VERSION,
                         compression=cfg.dng_compression, bit_depth=cfg.dng_bit_depth, layout=lay,
                         exif_fields=res.decoded.exif_fields, write_preview=False)
        os.replace(tmp, p)
    log(f"  {stem}: {method} converted in {time.time() - t:.0f}s")
    return paths


# ----------------------------------------------------------------- rendering
def render(path) -> np.ndarray:
    """Linear sRGB float32 in [0, 1+], unrotated, as an editor's converter would."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        out = r.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1),
                            user_flip=0, output_color=rawpy.ColorSpace.sRGB,
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD)
        s = r.sizes
    # LibRaw reports a DNG's DefaultCrop but postprocess() doesn't apply it. A CFA
    # DNG carries a 4 px demosaicing border there; left in, it shifts the render
    # by 4 px against the linear one and every CFA score came out ~1-2 dB low.
    if s.crop_width and s.crop_height and (s.crop_width, s.crop_height) != (out.shape[1], out.shape[0]):
        out = out[s.crop_top_margin:s.crop_top_margin + s.crop_height,
                  s.crop_left_margin:s.crop_left_margin + s.crop_width]
    return out.astype(np.float32) / 65535.0


def _luma(img):
    return img @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def align(ref: np.ndarray, img: np.ndarray, prior=None, search: int = 48) -> tuple[int, int]:
    """(dy, dx) such that img[y, x] ~ ref[y + dy, x + dx], by phase correlation
    of gamma-encoded luminance around `prior` (default: img centred in ref)."""
    a = np.power(np.clip(_luma(ref), 0, 1), 1 / 2.2)
    b = np.power(np.clip(_luma(img), 0, 1), 1 / 2.2)
    py, px = prior if prior is not None else ((a.shape[0] - b.shape[0]) // 2, (a.shape[1] - b.shape[1]) // 2)
    s = min(b.shape[0], b.shape[1], 2048) - 2 * search
    s = max(64, s // 2 * 2)
    yb, xb = (b.shape[0] - s) // 2, (b.shape[1] - s) // 2
    ya, xa = yb + py, xb + px
    A = a[ya:ya + s, xa:xa + s]; B = b[yb:yb + s, xb:xb + s]
    win = np.outer(np.hanning(s), np.hanning(s)).astype(np.float32)
    FA = np.fft.rfft2((A - A.mean()) * win); FB = np.fft.rfft2((B - B.mean()) * win)
    R = FA * np.conj(FB); R /= np.abs(R) + 1e-12
    c = np.fft.fftshift(np.fft.irfft2(R, s=A.shape))
    cy, cx = s // 2, s // 2
    sub = c[cy - search:cy + search + 1, cx - search:cx + search + 1]
    qy, qx = np.unravel_index(np.argmax(sub), sub.shape)
    return int(py + (qy - search)), int(px + (qx - search))


def overlap(ref, img, dy, dx):
    """The two aligned, equally-sized arrays (borders trimmed)."""
    y0, x0 = max(0, -dy), max(0, -dx)  # in img
    y1 = min(img.shape[0], ref.shape[0] - dy); x1 = min(img.shape[1], ref.shape[1] - dx)
    a = ref[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    b = img[y0:y1, x0:x1]
    t = BORDER
    return a[t:-t, t:-t], b[t:-t, t:-t]


# ------------------------------------------------------------------- metrics
def _psnr(a, b):
    mse = float(np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1 / mse)


def gains(ref, img):
    """Per-channel least-squares gain mapping img onto ref, from unclipped pixels."""
    ok = (ref.max(axis=-1) < 0.98) & (img.max(axis=-1) < 0.98)
    g = []
    for ch in range(3):
        r, i = ref[..., ch][ok], img[..., ch][ok]
        g.append(float((r * i).sum() / max((i * i).sum(), 1e-12)))
    return np.array(g, np.float32)


def _enc(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def metrics(ref, img):
    g = gains(ref, img)
    m = img * g
    return {"psnr": _psnr(ref, img), "psnr_gain": _psnr(ref, m), "psnr_gain_display": _psnr(_enc(ref), _enc(m)),
            "gain_r": g[0], "gain_g": g[1], "gain_b": g[2]}


# ------------------------------------------------------------- comparisons
def detailed_crop(img, size):
    """Top-left corner of the size x size window with the most fine detail."""
    l = np.power(np.clip(_luma(img), 0, 1), 1 / 2.2)
    gy, gx = np.abs(np.diff(l, axis=0))[:, :-1], np.abs(np.diff(l, axis=1))[:-1, :]
    e = gy + gx
    step = size // 2
    best, at = -1.0, (0, 0)
    for y in range(0, e.shape[0] - size, step):
        for x in range(0, e.shape[1] - size, step):
            v = float(np.median(e[y:y + size, x:x + size]))
            if v > best:
                best, at = v, (y, x)
    return at


def _orient(tile, orientation):
    return {3: lambda a: np.rot90(a, 2), 6: lambda a: np.rot90(a, 3), 8: lambda a: np.rot90(a, 1),
            5: lambda a: np.rot90(a, 1), 7: lambda a: np.rot90(a, 3)}.get(orientation, lambda a: a)(tile)


def comparison_image(path, title, tiles, orientation, tile_px=360, cols=4):
    """tiles: [(label, sublabel, float RGB in display space 0..1)] -> one JPEG grid."""
    from PIL import Image, ImageDraw, ImageFont
    font_path = ROOT / "tools" / "logo" / "fonts"
    fonts = sorted(glob.glob(str(font_path / "*.ttf")))
    def font(sz):
        try:
            return ImageFont.truetype(fonts[0], sz) if fonts else ImageFont.load_default()
        except OSError:
            return ImageFont.load_default()
    rows = -(-len(tiles) // cols)
    pad, head, lab = 10, 46, 46
    W = cols * tile_px + (cols + 1) * pad
    H = head + rows * (tile_px + lab) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (24, 24, 27))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 12), title, fill=(235, 235, 235), font=font(22))
    for i, (label, sub, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tile_px + pad); y = head + pad + r * (tile_px + lab + pad)
        t = (np.clip(_orient(img, orientation), 0, 1) * 255 + 0.5).astype(np.uint8)
        im = Image.fromarray(t).resize((tile_px, tile_px), Image.NEAREST if t.shape[0] < tile_px else Image.LANCZOS)
        canvas.paste(im, (x, y))
        d.text((x, y + tile_px + 4), label, fill=(235, 235, 235), font=font(17))
        d.text((x, y + tile_px + 24), sub, fill=(160, 160, 165), font=font(14))
    canvas.save(path, quality=88, optimize=True, progressive=True)


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pairs", help="folder with <name>.JPG + <name>.<raw ext> pairs")
    ap.add_argument("--out", default="raw_pair_eval")
    ap.add_argument("--checkpoint", action="append", default=[], help="InvISP checkpoint to include (repeatable)")
    ap.add_argument("--crops", action="store_true", help="also write a comparison image per pair")
    ap.add_argument("--crop", type=int, default=384, help="crop size in pixels for --crops (shown enlarged)")
    ap.add_argument("--region", type=int, default=0,
                    help="evaluate only the most detailed SIZE x SIZE region of each photo (default: whole "
                         "frame). InvISP on a CPU takes ~40 min per 24 MP photo; 2048 is ~6x faster.")
    a = ap.parse_args(argv)
    from PIL import Image

    out = Path(a.out); dng_dir = out / "dng"; dng_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(a.pairs)
    if not pairs:
        raise SystemExit(f"no JPEG + raw pairs in {a.pairs}")
    methods = methods_for(a.checkpoint)
    print(f"[pairs] {len(pairs)} pair(s), {len(methods)} method(s) x linear/CFA", flush=True)
    rows = []
    for stem, jpg, raw in pairs:
        region = eval_region(jpg, a.region)
        if region:
            print(f"  {stem}: evaluating the {region[2]}x{region[3]} px region at ({region[1]}, {region[0]})", flush=True)
        dngs = {m: convert(jpg, m, ov, dng_dir, stem, region) for m, ov in methods}
        ref = render(raw)
        orientation = Image.open(jpg).getexif().get(0x0112, 1)
        jpeg_lin = np.asarray(Image.open(jpg).convert("RGB"), np.float32) / 255.0
        if region:
            y, x, h, w = region
            jpeg_lin = jpeg_lin[y:y + h, x:x + w]
        jpeg_lin = np.where(jpeg_lin <= 0.04045, jpeg_lin / 12.92, ((jpeg_lin + 0.055) / 1.055) ** 2.4)
        renders = {(m, lay): render(p) for m, ps in dngs.items() for lay, p in ps.items()}
        full_h, full_w = Image.open(jpg).size[::-1]
        prior = ((ref.shape[0] - full_h) // 2 + (region[0] if region else 0),
                 (ref.shape[1] - full_w) // 2 + (region[1] if region else 0))
        dy, dx = align(ref, renders[("classical", "linear")], prior)
        print(f"  {stem}: raw {ref.shape[1]}x{ref.shape[0]}, JPEG offset in raw ({dx:+d}, {dy:+d}) px", flush=True)
        results = {}
        for (m, lay), img in renders.items():
            r_, i_ = overlap(ref, img, dy, dx)
            results[(m, lay)] = metrics(r_, i_)
            rows.append({"pair": stem, "method": m, "layout": lay, **{k: round(float(v), 4) for k, v in results[(m, lay)].items()}})
            print(f"    {m:32s} {lay:6s} PSNR {results[(m, lay)]['psnr']:.2f} | gain-matched "
                  f"{results[(m, lay)]['psnr_gain']:.2f} (display {results[(m, lay)]['psnr_gain_display']:.2f}) dB", flush=True)
        if a.crops:
            r_full, j_full = overlap(ref, jpeg_lin, dy, dx)
            y, x = detailed_crop(r_full, a.crop)
            sl = (slice(y, y + a.crop), slice(x, x + a.crop))
            # one display scale for every raw-derived tile: the true raw's 99.5th percentile -> 0.95
            scale = 0.95 / max(float(np.percentile(_luma(r_full[sl]), 99.5)), 1e-6)
            tiles = [("Camera JPEG", "what OpenRAW starts from", _enc(j_full[sl])),
                     ("Real raw (.ARW)", "ground truth, same rendering", _enc(r_full[sl] * scale))]
            for (m, lay), img in renders.items():
                r_, i_ = overlap(ref, img, dy, dx)
                g = gains(r_, i_)
                res = results[(m, lay)]
                tiles.append((f"{m.replace('openraw-fivek-', '')} ({'CFA' if lay == 'cfa' else 'linear'})",
                              f"{res['psnr_gain']:.1f} dB gain-matched", _enc(i_[sl] * g * scale)))
            name = out / f"{stem}_compare.jpg"
            camera = Image.open(jpg).getexif().get(0x0110, "")
            comparison_image(name, f"{stem}{' (' + camera.strip() + ')' if camera else ''}: {a.crop}x{a.crop} px crop "
                                   f"-- every raw rendered by LibRaw the same way, matched to the real raw's exposure",
                             tiles, orientation)
            print(f"  {stem}: wrote {name}", flush=True)
        del renders, ref

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    # summary table: mean over pairs
    keys = sorted({(r["method"], r["layout"]) for r in rows}, key=lambda k: [m for m, _ in methods].index(k[0]) * 2 + (k[1] == "cfa"))
    lines = ["| Method | Layout | PSNR | PSNR, gain-matched | gain-matched, display |", "|---|---|---|---|---|"]
    for m, lay in keys:
        sel = [r for r in rows if r["method"] == m and r["layout"] == lay]
        mean = lambda k: sum(r[k] for r in sel) / len(sel)  # noqa: E731
        lines.append(f"| {m} | {lay} | {mean('psnr'):.2f} dB | {mean('psnr_gain'):.2f} dB | {mean('psnr_gain_display'):.2f} dB |")
    (out / "results.md").write_text(f"Mean over {len(pairs)} RAW+JPEG pair(s)\n\n" + "\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
