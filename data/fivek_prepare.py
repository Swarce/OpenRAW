"""
Per-camera MIT-Adobe FiveK preparation for InvISP training: download only the
DNGs a camera needs, turn them into (RAW .npz, RGB .jpg) training pairs, and
write train/test lists -- in the exact layout dataset/FiveK_dataset.py reads:

    <data_root>/fivek/raw/<Make_Model>/<name>.dng   downloaded originals -- the same
                                                    layout fivek_download.py uses
                                                    with --out <data_root>/fivek
    <data_root>/<CameraDir>/RAW/<name>.npz     demosaiced linear raw + metadata
    <data_root>/<CameraDir>/RGB/<name>.jpg     rendered target
    <data_root>/<CameraDir>_train.txt / _test.txt

Downloading and camera metadata reuse data/fivek_download.py (also usable on
its own). Used by `train.py --camera ... [--download]`; can also run alone:

    python data/fivek_prepare.py --camera "Nikon D70" --download

Preprocessing differs from upstream's data_preprocess.py, deliberately, in
three ways that are invisible with InvISP's two cameras but break others:

  * CFA pattern is read from each DNG (upstream hardcodes RGGB; a GRBG/BGGR
    sensor demosaiced as RGGB swaps colors and trains the model on garbage).
  * Black level is subtracted per CFA channel, from the DNG (upstream never
    subtracts it -- its Canon-only branch is dead code, see data/README.md).
  * White level is stored per image, so the loader normalizes each image by
    its real range (upstream hardcodes 4095 for Canon, 16383 for everyone else
    -- wrong for every 12-bit sensor that isn't the Canon EOS 5D).

Because of the black-level fix, Canon EOS 5D data prepared here differs from
what the shipped canon.pth was trained on. To reproduce upstream exactly, use
data/data_preprocess.py instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fivek_download as fk  # noqa: E402  (the user's downloader, used as a library)

# InvISP shipped these two cameras under fixed folder names with its own
# train/test lists; keep both so results stay comparable to canon/nikon.pth.
LEGACY_DIRS = {fk.norm("Nikon D700"): "NIKON_D700", fk.norm("Canon EOS 5D"): "Canon_EOS_5D"}
JPEG_QUALITY = 90  # same as upstream


def camera_dir_name(label: str) -> str:
    """Folder for preprocessed pairs + split lists (what the loader calls a camera)."""
    return LEGACY_DIRS.get(fk.norm(label)) or fk.folder_name(*label.split(" ", 1))


def raw_dir(data_root: Path, label: str) -> Path:
    """Where a camera's DNGs live: fivek_download.py's layout, e.g.
    data/fivek/raw/Canon_EOS_10D/ (with --out data/fivek)."""
    return Path(data_root) / "fivek" / "raw" / fk.folder_name(*label.split(" ", 1))


def downloaded_cameras(data_root="./data/", log=print) -> list[str]:
    """FiveK camera labels for every data/fivek/raw/<Make_Model>/ folder that
    holds at least one DNG. Folders that match no FiveK camera are reported."""
    root = Path(data_root) / "fivek" / "raw"
    if not root.is_dir():
        return []
    items = fk.load_metadata(Path(data_root) / "fivek" / "_metadata", list(fk.SPLIT_FILES))
    by_folder = {fk.folder_name(*label.split(" ", 1)): label for label in fk.build_camera_index(items)}
    found = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not any(d.glob("*.dng")):
            continue
        if d.name in by_folder:
            found.append(by_folder[d.name])
        else:
            log(f"[data] skipping {d}: folder name matches no FiveK camera")
    return found


# --------------------------------------------------------------------------- #
# preprocessing one DNG
# --------------------------------------------------------------------------- #
def _flip(img, flip):
    return {3: lambda a: np.rot90(a, 2), 5: lambda a: np.rot90(a, 1), 6: lambda a: np.rot90(a, 3)}.get(flip, lambda a: a)(img)


def cfa_pattern(raw) -> str | None:
    """'RGGB'/'GRBG'/'BGGR'/'GBRG' read from the file, or None if not a 2x2 RGB Bayer CFA
    (e.g. X-Trans or CMYG sensors -- skipped rather than mis-demosaiced)."""
    pat = np.asarray(raw.raw_pattern)
    if pat.shape != (2, 2):
        return None
    desc = raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else raw.color_desc
    letters = "".join(desc[i] for i in pat.flatten())
    return letters if sorted(letters) == ["B", "G", "G", "R"] else None


def preprocess_one(dng: str, raw_dir: str, rgb_dir: str) -> tuple[str, str]:
    """-> (status, detail). Never raises: one bad file must not stop a camera's batch."""
    import rawpy
    import colour_demosaicing
    from PIL import Image

    name = Path(dng).stem
    npz, jpg = Path(raw_dir) / f"{name}.npz", Path(rgb_dir) / f"{name}.jpg"
    if npz.exists() and jpg.exists():
        return "skipped", name
    try:
        with rawpy.imread(dng) as raw:
            pattern = cfa_pattern(raw)
            if pattern is None:
                return "unsupported", f"{name}: not a 2x2 RGB Bayer sensor"
            data = raw.raw_image_visible.astype(np.float32)
            black = np.asarray(raw.black_level_per_channel, dtype=np.float32)[raw.raw_colors_visible]
            data = np.maximum(data - black, 0)
            white = float(raw.white_level) - float(np.mean(raw.black_level_per_channel))
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)  # target, as upstream
            wb = np.asarray(raw.camera_whitebalance, dtype=np.float32)
            flip = raw.sizes.flip
        de = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(data, pattern).astype(np.float32)
        # Clip to the sensor's range. The bilinear filter reflects at image
        # borders, which breaks the CFA's parity there: border pixels sum up to
        # ~1.5x too many samples (measured: 4606 on a 3839-white sensor), i.e.
        # values no real photosite can produce. Upstream has the same overshoot.
        np.clip(de, 0, white, out=de)
        de = _flip(de, flip)
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        Path(rgb_dir).mkdir(parents=True, exist_ok=True)
        tmp_j = jpg.with_suffix(".part.jpg")
        Image.fromarray(rgb).save(tmp_j, quality=JPEG_QUALITY, subsampling=1)
        tmp_n = npz.with_suffix(".part.npz")
        np.savez(tmp_n, raw=de, wb=wb, white_level=np.float32(white), cfa_pattern=pattern)
        os.replace(tmp_j, jpg)
        os.replace(tmp_n, npz)  # npz last: its presence marks the pair complete
        return "ok", name
    except Exception as e:  # noqa: BLE001
        return "failed", f"{name}: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# per-camera pipeline
# --------------------------------------------------------------------------- #
def _write_split_lists(data_root: Path, cam_dir: str, names, items, legacy: bool):
    train_txt, test_txt = data_root / f"{cam_dir}_train.txt", data_root / f"{cam_dir}_test.txt"
    if legacy and train_txt.exists() and test_txt.exists():
        return "InvISP's published split"
    # FiveK's official split: train + validation -> train, test -> test
    train = sorted(n for n in names if items[n]["_split"] in ("train", "val"))
    test = sorted(n for n in names if items[n]["_split"] == "test")
    train_txt.write_text("\n".join(train) + "\n")
    test_txt.write_text("\n".join(test) + "\n")
    return f"FiveK official split ({len(train)} train / {len(test)} test)"


def prepare_cameras(queries, data_root="./data/", download=False, jobs=4, workers=None,
                    limit=0, contains=False, log=print, use_available=False) -> list[str]:
    """Make the given cameras trainable. Returns their folder names (what the
    dataset loader expects as camera names).

    use_available: train on whatever DNGs are already downloaded instead of
    stopping when some are missing (used by --all-downloaded)."""
    data_root = Path(data_root)
    items = fk.load_metadata(data_root / "fivek" / "_metadata", list(fk.SPLIT_FILES))
    index = fk.build_camera_index(items)
    labels = fk.resolve_cameras(queries, index, contains)
    cam_dirs = []
    for label in labels:
        cam_dir = camera_dir_name(label)
        legacy = fk.norm(label) in LEGACY_DIRS
        base = data_root / cam_dir
        names = sorted(index[label])
        if legacy:  # InvISP's lists define which images this camera trains on
            lst = data_root / f"{cam_dir}.txt"
            if lst.exists():
                listed = {Path(l.strip()).stem for l in lst.read_text().split() if l.strip()}
                names = [n for n in names if n in listed] or names
        if limit:
            names = names[:limit]
        log(f"[data] {label} -> {cam_dir}/ ({len(names)} images)")

        rdir, old_dir = raw_dir(data_root, label), base / "DNG"
        # fivek_download.py's layout; DNGs left in the older <CameraDir>/DNG/ still count
        dngs = {n: (old_dir / f"{n}.dng") if (old_dir / f"{n}.dng").exists() else (rdir / f"{n}.dng") for n in names}
        done = {n for n in names if (base / "RAW" / f"{n}.npz").exists() and (base / "RGB" / f"{n}.jpg").exists()}
        # a finished pair counts as present even if its DNG was deleted to save space
        missing = [n for n, p in dngs.items() if n not in done and not (p.exists() and p.stat().st_size > 0)]
        if missing and not download:
            if use_available:
                log(f"[data]   {len(missing)} of {len(names)} not downloaded -- training on the {len(names) - len(missing)} present")
                names = [n for n in names if n not in missing]
                missing = []
            else:
                raise SystemExit(f"[data] {label}: {len(missing)} of {len(names)} DNGs missing in {rdir}. "
                                 f"Re-run with --download, or place them there.")
        if missing:
            log(f"[data]   downloading {len(missing)} DNG(s) ...")
            fails = []
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
                futs = {ex.submit(fk.download, items[n]["urls"]["dng"], dngs[n]): n for n in missing}
                for i, f in enumerate(as_completed(futs), 1):
                    try:
                        f.result()
                    except Exception as e:  # noqa: BLE001
                        fails.append(str(e))
                    if i % 25 == 0 or i == len(missing):
                        log(f"[data]   {i}/{len(missing)} downloaded")
            if fails:
                log(f"[data]   {len(fails)} download(s) failed (re-run to retry): {fails[0]}")
            names = [n for n in names if n in done or dngs[n].exists()]

        todo = [n for n in names if n not in done]
        ok_names = set(done) & set(names)
        if todo:
            log(f"[data]   preprocessing {len(todo)} DNG(s) ...")
            # spawn, not fork: forking a multi-threaded process (downloads just ran
            # on threads) can deadlock; spawn is also what Windows always uses
            with ProcessPoolExecutor(max_workers=workers or os.cpu_count() or 1, mp_context=mp.get_context("spawn")) as ex:
                futs = {ex.submit(preprocess_one, str(dngs[n]), str(base / "RAW"), str(base / "RGB")): n for n in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    status, detail = f.result()
                    if status in ("ok", "skipped"):
                        ok_names.add(futs[f])
                    else:
                        log(f"[data]   {status}: {detail}")
                    if i % 25 == 0 or i == len(todo):
                        log(f"[data]   {i}/{len(todo)} preprocessed")
        how = _write_split_lists(data_root, cam_dir, sorted(ok_names), items, legacy)
        log(f"[data]   ready: {len(ok_names)} pairs, {how}")
        cam_dirs.append(cam_dir)
    return cam_dirs


def list_cameras(data_root="./data/"):
    items = fk.load_metadata(Path(data_root) / "fivek" / "_metadata", list(fk.SPLIT_FILES))
    index = fk.build_camera_index(items)
    print(f"{'Camera':30s} {'Images':>6s}  folder")
    for label, names in sorted(index.items(), key=lambda kv: -len(kv[1])):
        note = "  (InvISP split)" if fk.norm(label) in LEGACY_DIRS else ""
        print(f"{label:30s} {len(names):6d}  {camera_dir_name(label)}{note}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare FiveK cameras for InvISP training.")
    ap.add_argument("-c", "--camera", action="append", default=[], help="FiveK camera, e.g. 'Nikon D70' (repeatable/comma)")
    ap.add_argument("--list", action="store_true", help="list cameras and exit")
    ap.add_argument("--download", action="store_true", help="download missing DNGs")
    ap.add_argument("--all-downloaded", action="store_true", help="every camera found in data/fivek/raw/, using what's there")
    ap.add_argument("--contains", action="store_true", help="substring camera matching")
    ap.add_argument("--data-path", default="./data/")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="parallel downloads")
    ap.add_argument("--workers", type=int, default=0, help="preprocessing processes (default: all cores)")
    ap.add_argument("--limit", type=int, default=0, help="first N images per camera (testing)")
    a = ap.parse_args()
    if a.list:
        list_cameras(a.data_path)
        sys.exit(0)
    qs = [q.strip() for c in a.camera for q in c.split(",") if q.strip()]
    if a.all_downloaded:
        qs += [c for c in downloaded_cameras(a.data_path) if c not in qs]
    if not qs:
        ap.error("give at least one --camera, --all-downloaded, or --list")
    prepare_cameras(qs, a.data_path, a.download, a.jobs, a.workers or None, a.limit, a.contains,
                    use_available=a.all_downloaded)
