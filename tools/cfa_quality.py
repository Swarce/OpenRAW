#!/usr/bin/env python3
"""
Measure how much `--layout cfa` (Bayer mosaic) costs on YOUR photos.

For each JPEG: run the OpenRAW pipeline once, write the result as a linear DNG
(the reference -- no demosaicing involved) and as a CFA DNG, then let real
demosaic engines reconstruct the CFA one and compare:

    PSNR (dB)      overall fidelity vs the linear output (sRGB 8-bit domain;
                   above ~40 dB differences are generally invisible)
    sharpness      edge energy relative to the linear output (100% = same;
                   above 100% usually means demosaic artifacts, not detail)
    colour error   mean chroma difference -- where false colour shows up

Engines: every demosaic algorithm in your libraw build (via rawpy), plus
Adobe's reference renderer if dng_validate is available (set
OPENRAW_DNG_VALIDATE, see tools/build_dng_validate.sh). Adobe's reference
demosaic is plain bilinear; Lightroom/Camera Raw use a better production one.

    python tools/cfa_quality.py photo.jpg [more.jpg ...] [--crops out_dir]

--crops saves side-by-side 2x crops of each photo's most detailed region:
linear | CFA via AHD | CFA via DCB (libraw) [| linear | CFA (Adobe reference)].
Each engine is shown next to its OWN render of the linear file: Adobe applies
its own tone curve, so comparing across engines would mislead.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALGORITHMS = ("LINEAR", "VNG", "PPG", "AHD", "DCB", "DHT", "AAHD")


def to_srgb8(lin16: np.ndarray) -> np.ndarray:
    x = lin16.astype(np.float32) / 65535
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.clip(x, 0, 1), 1 / 2.4) - 0.055) * 255


def libraw_decode(path, algorithm=None):
    """Linear 16-bit RGB, with the DNG DefaultCrop applied (libraw reports it
    but its postprocess doesn't apply it)."""
    import rawpy
    with rawpy.imread(path) as r:
        s = r.sizes
        kw = dict(output_bps=16, no_auto_bright=True, gamma=(1, 1), use_camera_wb=True)
        if algorithm:
            kw["demosaic_algorithm"] = getattr(rawpy.DemosaicAlgorithm, algorithm)
        o = r.postprocess(**kw)
    if s.crop_width:
        o = o[s.crop_top_margin:s.crop_top_margin + s.crop_height, s.crop_left_margin:s.crop_left_margin + s.crop_width]
    return o


def adobe_render(path, validator):
    """8-bit sRGB render through Adobe's reference DNG SDK (DefaultCrop applied)."""
    import tifffile
    stem = path[:-4] + "_adobe"
    subprocess.run([validator, "-tif", stem, path], capture_output=True)
    return tifffile.imread(stem + ".tif").astype(np.float32)


def _edges(img8):
    import cv2
    g = cv2.cvtColor(np.clip(img8, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)


def metrics(out8, ref8):
    mse = float(((out8 - ref8) ** 2).mean())
    psnr = 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    sharp = _edges(out8).mean() / max(_edges(ref8).mean(), 1e-9)
    chroma = np.abs((out8 - out8.mean(2, keepdims=True)) - (ref8 - ref8.mean(2, keepdims=True))).mean()
    return psnr, sharp, chroma


def detail_crop_box(ref8, size):
    """Top-left of the size x size window with the most edge energy."""
    import cv2
    e = _edges(ref8)
    k = max(8, size // 4)
    small = cv2.resize(e, (max(1, e.shape[1] // k), max(1, e.shape[0] // k)), interpolation=cv2.INTER_AREA)
    n = max(1, size // k)
    best, by, bx = -1, 0, 0
    for y in range(0, max(1, small.shape[0] - n + 1)):
        for x in range(0, max(1, small.shape[1] - n + 1)):
            v = small[y:y + n, x:x + n].sum()
            if v > best:
                best, by, bx = v, y, x
    return min(by * k, max(0, ref8.shape[0] - size)), min(bx * k, max(0, ref8.shape[1] - size))


def save_crops(path, panels, labels, size=160, zoom=2):
    from PIL import Image, ImageDraw
    y, x = detail_crop_box(panels[0], size)
    tiles = []
    for img, label in zip(panels, labels):
        t = Image.fromarray(np.clip(img[y:y + size, x:x + size], 0, 255).astype(np.uint8))
        t = t.resize((t.width * zoom, t.height * zoom), Image.NEAREST)
        canvas = Image.new("RGB", (t.width, t.height + 22), (20, 20, 20))
        canvas.paste(t, (0, 22))
        ImageDraw.Draw(canvas).text((6, 5), label, fill=(230, 230, 230))
        tiles.append(canvas)
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), tiles[0].height), (0, 0, 0))
    xo = 0
    for t in tiles:
        sheet.paste(t, (xo, 0))
        xo += t.width + 4
    sheet.save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jpegs", nargs="+")
    ap.add_argument("--crops", help="folder for visual side-by-side crops")
    ap.add_argument("--bit-depth", type=int, default=12, choices=[16, 14, 12, 10])
    a = ap.parse_args(argv)

    from openraw import OpenRawPipeline
    from openraw.dng_writer import write_linear_dng
    validator = os.environ.get("OPENRAW_DNG_VALIDATE")
    validator = validator if validator and os.path.exists(validator) else None
    if a.crops:
        os.makedirs(a.crops, exist_ok=True)

    for jpeg in a.jpegs:
        rgb16 = OpenRawPipeline().run(jpeg).rgb16
        with tempfile.TemporaryDirectory() as td:
            lin, cfa = os.path.join(td, "linear.dng"), os.path.join(td, "cfa.dng")
            write_linear_dng(lin, rgb16, compression="none")
            write_linear_dng(cfa, rgb16, layout="cfa", bit_depth=a.bit_depth)
            ref = to_srgb8(libraw_decode(lin))
            print(f"\n{jpeg}  ({rgb16.shape[1]}x{rgb16.shape[0]}, CFA DNG {os.path.getsize(cfa) / 1e6:.1f} MB "
                  f"vs linear {os.path.getsize(lin) / 1e6:.1f} MB uncompressed)")
            print(f"  {'engine':20} {'PSNR dB':>8} {'sharpness':>10} {'colour err':>11}")
            outs = {}
            for alg in ALGORITHMS:
                try:
                    out = to_srgb8(libraw_decode(cfa, alg))
                except Exception as e:  # noqa: BLE001 -- not every libraw build has every algorithm
                    print(f"  libraw {alg:13} unavailable ({type(e).__name__})")
                    continue
                outs[alg] = out
                p, sh, c = metrics(out, ref)
                print(f"  libraw {alg:13} {p:8.2f} {sh:9.1%} {c:11.2f}")
            ref_a = None
            if validator:
                ref_a, out_a = adobe_render(lin, validator), adobe_render(cfa, validator)
                outs["Adobe reference"] = out_a
                p, sh, c = metrics(out_a, ref_a)
                print(f"  {'Adobe reference':20} {p:8.2f} {sh:9.1%} {c:11.2f}   (bilinear; Lightroom/ACR are better)")
            if a.crops:
                # Adobe applies its own tone curve, so its CFA render sits next to
                # Adobe's render of the LINEAR file -- never next to libraw's
                panels, labels = [ref], ["linear (libraw)"]
                for n in ("AHD", "DCB"):
                    if n in outs:
                        panels.append(outs[n]); labels.append(f"CFA via {n} (libraw)")
                if ref_a is not None:
                    panels += [ref_a, outs["Adobe reference"]]
                    labels += ["linear (Adobe ref.)", "CFA (Adobe ref.)"]
                dest = os.path.join(a.crops, Path(jpeg).stem + "_cfa_crops.png")
                save_crops(dest, panels, labels)
                print(f"  crops -> {dest}")


if __name__ == "__main__":
    main()
