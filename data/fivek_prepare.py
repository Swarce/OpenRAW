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
import warnings

# colour-science (under colour-demosaicing) warns on import when matplotlib is
# missing, because some of its PLOTTING features need it. We never plot, so
# nothing is lost -- but it printed once per preprocessing worker. Silence
# exactly that message; any other colour-science warning still shows.
warnings.filterwarnings("ignore", message=r'.*"Matplotlib" related API features are not available')
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: dataset.mosaic_store
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
    holds at least one DNG, plus every camera that's already preprocessed (its
    DNGs may have been deleted). Folders that match no FiveK camera are reported."""
    root = Path(data_root) / "fivek" / "raw"
    items = fk.load_metadata(Path(data_root) / "fivek" / "_metadata", list(fk.SPLIT_FILES))
    labels = fk.build_camera_index(items)
    by_folder = {fk.folder_name(*label.split(" ", 1)): label for label in labels}
    found = []
    if root.is_dir():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if not any(d.glob("*.dng")):
                continue
            if d.name in by_folder:
                found.append(by_folder[d.name])
            else:
                log(f"[data] skipping {d}: folder name matches no FiveK camera")
    # cameras already preprocessed (their DNGs may have been deleted with --delete-dngs)
    for label in sorted(labels):
        raw = Path(data_root) / camera_dir_name(label) / "RAW"
        if label not in found and raw.is_dir() and any(raw.glob("*.npz")):
            found.append(label)
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


def preprocess_one(dng: str, raw_dir: str, rgb_dir: str, delete_dng: bool = False) -> tuple[str, str]:
    """-> (status, detail). Never raises: one bad file must not stop a camera's batch.

    Writes the compact mosaic format (dataset/mosaic_store.py): the
    black-subtracted, white-clipped sensor mosaic, rotated to display
    orientation, as 4 lossless-JPEG planes (~18 MB for 18 MP; the old
    demosaiced float32 format was ~216 MB). Demosaicing happens at load time.
    delete_dng: remove the DNG once its pair is safely written."""
    import rawpy
    from PIL import Image
    from dataset import mosaic_store

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
            white = float(raw.white_level) - float(np.mean(raw.black_level_per_channel))
            # black-subtract, and clip to the sensor's range (no photosite exceeds saturation)
            mosaic = np.clip(np.round(data - black), 0, white).astype(np.uint16)
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)  # target, as upstream
            wb = np.asarray(raw.camera_whitebalance, dtype=np.float32)
            flip = raw.sizes.flip
        mosaic, pattern = mosaic_store.rotate_mosaic(mosaic, pattern, flip)
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        Path(rgb_dir).mkdir(parents=True, exist_ok=True)
        tmp_j = jpg.with_suffix(".part.jpg")
        Image.fromarray(rgb).save(tmp_j, quality=JPEG_QUALITY, subsampling=1)
        tmp_n = npz.with_suffix(".part.npz")
        np.savez(tmp_n, **mosaic_store.pack(mosaic, pattern, white, wb))
        os.replace(tmp_j, jpg)
        os.replace(tmp_n, npz)  # npz last: its presence marks the pair complete
        if delete_dng:
            Path(dng).unlink(missing_ok=True)  # only after the pair is complete
        return "ok", name
    except Exception as e:  # noqa: BLE001
        return "failed", f"{name}: {type(e).__name__}: {e}"


def compact_existing(npz_path: str) -> tuple[str, str]:
    """Shrink a pair written in the previous format (demosaiced float32, ~216 MB
    for 18 MP) to the mosaic format in place -- exact (dataset/mosaic_store.py:
    remosaic), no re-download needed. Leaves upstream data_preprocess.py files
    (no white_level) and already-compact files alone."""
    from dataset import mosaic_store
    p = Path(npz_path)
    try:
        with np.load(p) as z:
            if mosaic_store.is_mosaic(z) or "white_level" not in z.files:
                return "skipped", p.stem
            mosaic, pattern = mosaic_store.remosaic(z["raw"], hint=str(z["cfa_pattern"]) if "cfa_pattern" in z.files else None)
            packed = mosaic_store.pack(mosaic, pattern, float(z["white_level"]), z["wb"])
        tmp = p.with_suffix(".part.npz")
        np.savez(tmp, **packed)
        os.replace(tmp, p)
        return "ok", p.stem
    except Exception as e:  # noqa: BLE001
        return "failed", f"{p.stem}: {type(e).__name__}: {e}"


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


def _needs_compacting(npz: Path) -> bool:
    try:
        with np.load(npz) as z:
            return "format" not in z.files and "white_level" in z.files
    except Exception:  # noqa: BLE001
        return False


def prepare_cameras(queries, data_root="./data/", download=False, jobs=4, workers=None,
                    limit=0, contains=False, log=print, use_available=False,
                    delete_dngs=False) -> list[str]:
    """Make the given cameras trainable. Returns their folder names (what the
    dataset loader expects as camera names).

    use_available: train on whatever DNGs are already downloaded instead of
        stopping when some are missing (used by --all-downloaded).
    delete_dngs: delete each DNG once its training pair is written (and any
        left over from earlier runs). Download + preprocessing then run in
        small batches, so peak disk use stays at a few hundred MB instead of a
        whole camera's DNGs.
    Pairs found in the previous, ~12x larger format are shrunk in place."""
    data_root = Path(data_root)
    items = fk.load_metadata(data_root / "fivek" / "_metadata", list(fk.SPLIT_FILES))
    index = fk.build_camera_index(items)
    labels = fk.resolve_cameras(queries, index, contains)
    n_workers = workers or os.cpu_count() or 1
    cam_dirs = []
    # spawn, not fork: forking a multi-threaded process (downloads run on
    # threads) can deadlock; spawn is also what Windows always uses
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
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

            # shrink pairs written in the previous (~12x larger) format
            old = [n for n in sorted(done) if _needs_compacting(base / "RAW" / f"{n}.npz")]
            if old:
                log(f"[data]   shrinking {len(old)} pair(s) from the previous storage format ...")
                for f in as_completed([pool.submit(compact_existing, str(base / "RAW" / f"{n}.npz")) for n in old]):
                    status, detail = f.result()
                    if status == "failed":
                        log(f"[data]   {detail}")
            if delete_dngs:  # leftovers from runs without --delete-dngs
                freed = 0
                for n in done:
                    if dngs[n].exists():
                        freed += dngs[n].stat().st_size
                        dngs[n].unlink()
                if freed:
                    log(f"[data]   deleted already-preprocessed DNGs: {freed / 1e9:.2f} GB freed")

            todo = [n for n in names if n not in done]
            need_dl = set(missing)
            ok_names = set(done) & set(names)
            # with --delete-dngs: small batches (download, preprocess, delete);
            # otherwise one batch, as before
            step = max(8, 4 * max(jobs, n_workers)) if delete_dngs else max(1, len(todo))
            n_dl = n_pp = 0
            for k in range(0, len(todo), step):
                part = todo[k:k + step]
                dl = [n for n in part if n in need_dl]
                if dl:
                    if k == 0:
                        log(f"[data]   downloading {len(need_dl)} DNG(s)" + (" in batches" if delete_dngs else "") + " ...")
                    fails = []
                    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
                        futs = {ex.submit(fk.download, items[n]["urls"]["dng"], dngs[n]): n for n in dl}
                        for f in as_completed(futs):
                            try:
                                f.result()
                            except Exception as e:  # noqa: BLE001
                                fails.append(str(e))
                            n_dl += 1
                            if n_dl % 25 == 0 or n_dl == len(need_dl):
                                log(f"[data]   {n_dl}/{len(need_dl)} downloaded")
                    if fails:
                        log(f"[data]   {len(fails)} download(s) failed (re-run to retry): {fails[0]}")
                    part = [n for n in part if dngs[n].exists()]
                if k == 0 and part:
                    log(f"[data]   preprocessing {len(todo)} DNG(s) ...")
                futs = {pool.submit(preprocess_one, str(dngs[n]), str(base / "RAW"), str(base / "RGB"), delete_dngs): n for n in part}
                for f in as_completed(futs):
                    status, detail = f.result()
                    if status in ("ok", "skipped"):
                        ok_names.add(futs[f])
                    else:
                        log(f"[data]   {status}: {detail}")
                    n_pp += 1
                    if n_pp % 25 == 0 or n_pp == len(todo):
                        log(f"[data]   {n_pp}/{len(todo)} preprocessed")
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
    ap.add_argument("--delete-dngs", action="store_true", help="delete each DNG once preprocessed (saves disk)")
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
                    use_available=a.all_downloaded, delete_dngs=a.delete_dngs)
