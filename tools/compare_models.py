#!/usr/bin/env python3
"""Score InvISP checkpoints side by side on the held-out test images.

Uses exactly what train.py's [EVAL] lines use -- the same cameras (resolved
the same way as `train.py --all-downloaded`), the same test images spread
across them, the same centre crops and the same PSNR -- so its numbers are
directly comparable with eval.csv. Adds a per-camera breakdown, which is
where single-camera models (upstream's canon.pth / nikon.pth) and pooled ones
differ.

    python tools/compare_models.py --data_path data/                       # every pretrained/*.pth
    python tools/compare_models.py --data_path data/ pretrained/nikon.pth exps/run/checkpoint/best.pth

Writes a Markdown table (and a CSV with one row per model and image) to
--out. Metrics: raw PSNR = the real JPEG through the inverse network vs the
true raw (what OpenRAW does); rgb PSNR = raw through the forward network vs
the JPEG.

Note on upstream's models: OpenRAW's prepared data subtracts each sensor's
black level and normalizes by each image's white level; upstream trained on
data without black subtraction and with fixed white levels (see
data/README.md). Their raw PSNR here therefore includes that convention
mismatch -- which is also exactly what they meet when used inside OpenRAW.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _camera_of(path: str) -> str:
    return Path(path).parent.parent.name  # <data>/<Camera>/RAW/<name>.npz


def build_eval_set(data_path: str, eval_images: int = 40, eval_crop: int = 512, gamma: bool = True):
    """(dataset subset, camera per item) -- the images train.py evaluates on."""
    import torch
    import train
    from dataset.FiveK_dataset import FiveKDatasetTest

    args = SimpleNamespace(camera=None, all_downloaded=True, download=False, download_jobs=1,
                           delete_dngs=False, data_path=data_path)
    cams = train.resolve_cameras(args)
    test_set = FiveKDatasetTest(opt=SimpleNamespace(data_path=data_path, camera=cams, gamma=gamma, debug_mode=False))
    if not len(test_set):
        raise SystemExit(f"no test images in {data_path} (cameras: {', '.join(cams)})")
    test_set.eval_crop = eval_crop
    idx = train.eval_indices(len(test_set), eval_images)
    paths = [test_set.data["input_RAWs_WBs"][i] for i in idx]
    return torch.utils.data.Subset(test_set, idx), [_camera_of(p) for p in paths], cams


def score(checkpoint: str, subset, device) -> list[tuple[float, float]]:
    """Per image (raw PSNR, rgb PSNR) for one checkpoint."""
    import torch
    import train
    from openraw.invisp_bridge import _load_net
    net = _load_net("", "", device, checkpoint=checkpoint)
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(len(subset)):
            b = subset[i]
            raw, rgb = b["input_raw"][None].to(device), b["target_rgb"][None].to(device)
            out.append((train._psnr(net(rgb, rev=True), raw), train._psnr(net(raw), rgb)))
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def markdown(results: dict, cameras: list[str]) -> str:
    """results: {model name: [(raw, rgb) per image]}; cameras: camera per image."""
    cams = sorted(set(cameras), key=cameras.index)
    count = {c: cameras.count(c) for c in cams}
    head = "| Model | raw PSNR | rgb PSNR | " + " | ".join(f"{c} ({count[c]})" for c in cams) + " |"
    lines = [head, "|" + "---|" * (3 + len(cams))]
    best = max(results, key=lambda m: _mean([r for r, _ in results[m]]))
    for m, vals in results.items():
        raw = _mean([r for r, _ in vals])
        rgb = _mean([g for _, g in vals])
        per = [_mean([r for (r, _), c in zip(vals, cameras) if c == cam]) for cam in cams]
        name = f"**{m}**" if m == best else m
        lines.append(f"| {name} | {raw:.2f} dB | {rgb:.2f} dB | " + " | ".join(f"{p:.2f}" for p in per) + " |")
    return "\n".join(lines) + ("\n\nraw PSNR per camera; the number of test images in brackets." if len(cams) > 1 else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", nargs="*", help="checkpoint files or globs (default: pretrained/*.pth)")
    ap.add_argument("--data_path", default="./data/")
    ap.add_argument("--eval_images", type=int, default=40, help="as train.py's --eval_images")
    ap.add_argument("--eval_crop", type=int, default=512, help="as train.py's --eval_crop")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--out", default="compare", help="output folder for compare.md / compare.csv")
    a = ap.parse_args(argv)
    import torch

    files = []
    for c in a.checkpoints or [str(ROOT / "pretrained" / "*.pth")]:
        hits = sorted(glob.glob(c)) or ([c] if os.path.isfile(c) else [])
        if not hits:
            raise SystemExit(f"no checkpoint matches {c}")
        files += [h for h in hits if h not in files]
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_path = a.data_path if a.data_path.endswith("/") else a.data_path + "/"
    subset, cameras, cams = build_eval_set(data_path, a.eval_images, a.eval_crop)
    print(f"[compare] {len(subset)} test images ({a.eval_crop} px centre crops) from {len(cams)} camera(s); "
          f"{len(files)} model(s) on {device}", flush=True)

    # name models by file; disambiguate same-named files (several best.pth) by their folder
    names = {}
    for f in files:
        n = Path(f).name
        if sum(Path(g).name == n for g in files) > 1:
            n = f"{Path(f).parent.parent.name}/{n}" if Path(f).parent.name == "checkpoint" else f"{Path(f).parent.name}/{n}"
        names[f] = n
    results = {}
    for f in files:
        t = time.time()
        try:
            results[names[f]] = score(f, subset, device)
        except Exception as e:  # noqa: BLE001  -- one unloadable file shouldn't lose the others
            print(f"[compare] {names[f]}: skipped ({type(e).__name__}: {e})", flush=True)
            continue
        r = _mean([x for x, _ in results[names[f]]])
        print(f"[compare] {names[f]}: raw PSNR {r:.2f} dB ({time.time() - t:.0f}s)", flush=True)
    if not results:
        raise SystemExit("no checkpoint could be scored")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = markdown(results, cameras)
    (out / "compare.md").write_text(
        f"# InvISP checkpoints on {len(subset)} held-out test images\n\n"
        f"{a.eval_crop} px centre crops, the images train.py's evaluation uses. raw PSNR: JPEG -> raw "
        f"(OpenRAW's direction; picks best.pth). rgb PSNR: raw -> JPEG.\n\n{table}\n")
    with open(out / "compare.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "camera", "image", "raw_psnr", "rgb_psnr"])
        for m, vals in results.items():
            for i, ((r, g), c) in enumerate(zip(vals, cameras)):
                w.writerow([m, c, i, f"{r:.4f}", f"{g:.4f}"])
    print("\n" + table + f"\n\n[compare] written to {out / 'compare.md'} and compare.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
