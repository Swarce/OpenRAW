"""
RAISE (Dang-Nguyen et al., ACM MMSys 2015) for InvISP training: download NEF
raws from the RAISE list, turn them into the same (RAW .npz, RGB .jpg)
training pairs as FiveK (data/fivek_prepare.py), and write train/test lists.

RAISE is 8,156 uncompressed raws from three Nikon DSLRs (D40, D90, D7000),
shot by four photographers across Europe. Its license allows non-commercial
research and educational use only, and asks that published work cite the
paper -- see https://loki.disi.unitn.it/RAISE/.

Getting the list: on https://loki.disi.unitn.it/RAISE/download.html pick a
package (RAISE-1k ... RAISE-All, or a custom selection), accept the terms, and
save the CSV it gives you. The CSV has one row per image, with the NEF's URL
in its "NEF" column. Then:

    python data/raise_prepare.py --csv RAISE_1k.csv --download --delete-nefs
    python data/raise_prepare.py --csv RAISE_all.csv --download --camera D90 --count 1500

Layout (the same the training loader reads for FiveK cameras):

    <data_root>/raise/raw/<Make_Model>/<File>.NEF      downloads
    <data_root>/RAISE_<Make_Model>/RAW/<File>.npz      compact sensor mosaic + metadata
    <data_root>/RAISE_<Make_Model>/RGB/<File>.jpg      rendered target
    <data_root>/RAISE_<Make_Model>_train.txt / _test.txt

Camera folders are prefixed RAISE_ so they never collide with a FiveK camera
of the same model. RAISE has no official split; each image goes to the test
split by a fixed hash of its name (~10%), so the split never changes when more
images are added. `train.py --all-downloaded` and the Kaggle runner pick these
cameras up next to the FiveK ones.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import multiprocessing as mp
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fivek_download as fk  # noqa: E402  (download helper)
import fivek_prepare as fp  # noqa: E402  (preprocess_one: same pairs as FiveK)

PREFIX = "RAISE_"
TEST_PERCENT = 10


# --------------------------------------------------------------------------- #
# the RAISE list
# --------------------------------------------------------------------------- #
def _col(header: list[str], *names: str) -> str | None:
    """The header entry matching one of `names`, ignoring case/spaces/punctuation."""
    by_norm = {fk.norm(h): h for h in header}
    for n in names:
        if fk.norm(n) in by_norm:
            return by_norm[fk.norm(n)]
    return None


def read_list(csv_path) -> list[dict]:
    """Rows of a RAISE CSV as {"name", "url", "device", "category"}.

    Columns are found by name, case-insensitively: File (image ID), NEF (URL),
    Device (camera), and Keywords/Category when present. A row without a NEF
    URL is skipped. If there's no File column, the ID comes from the URL."""
    raw = Path(csv_path).read_bytes().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    header = reader.fieldnames or []
    c_nef = _col(header, "NEF", "NEF URL", "NEF link")
    if c_nef is None:
        raise SystemExit(f"[raise] {csv_path}: no 'NEF' column (found: {', '.join(header) or 'nothing'}). "
                         "Use the CSV from the RAISE download page.")
    c_file, c_dev = _col(header, "File", "Filename", "Name"), _col(header, "Device", "Camera", "Model")
    c_cat = _col(header, "Keywords", "Category", "Categories")
    rows, seen = [], set()
    for r in reader:
        url = (r.get(c_nef) or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            continue
        name = (r.get(c_file) or "").strip() if c_file else ""
        name = re.sub(r"(?i)\.(nef|tiff?)$", "", name) if name else Path(url.split("?")[0]).stem
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or name in seen:
            continue  # IDs become file names: nothing that could escape the folder
        seen.add(name)
        rows.append({"name": name, "url": url,
                     "device": (r.get(c_dev) or "").strip() if c_dev else "",
                     "category": (r.get(c_cat) or "").strip() if c_cat else ""})
    return rows


def camera_folder(device: str) -> str:
    """'Nikon D90' / 'NIKON D90' / 'D90' -> 'Nikon_D90' (RAISE is all Nikon)."""
    d = re.sub(r"(?i)^nikon(\s+corporation)?\s*", "", device.strip())
    return fk.folder_name("Nikon", d) if d else "Unknown"


def camera_dir_name(device: str) -> str:
    return PREFIX + camera_folder(device)


def in_test_split(name: str) -> bool:
    """Fixed per image: adding images later never moves one between splits."""
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % 100 < TEST_PERCENT


def _order_key(name: str) -> str:
    return hashlib.sha1(b"order:" + name.encode()).hexdigest()


def select(rows, cameras=(), categories=(), start=0, count=0, csv_order=False):
    """Filter by camera (substring of the model, e.g. 'D90') and category
    keyword, then take rows[start:start+count].

    Rows are taken in a fixed shuffled order (by a hash of each name), not the
    CSV's: RAISE's list runs in long same-camera, same-shoot stretches (one
    stretch of 1,300 rows is all D90), so a part taken in list order would be
    one camera and many near-identical burst frames. The shuffled order is the
    same on every run and machine, so --start/--count parts never overlap.
    csv_order: keep the list's order instead."""
    if not csv_order:
        rows = sorted(rows, key=lambda r: _order_key(r["name"]))
    if cameras:
        want = [fk.norm(c) for c in cameras]
        rows = [r for r in rows if any(w in fk.norm(r["device"]) for w in want)]
    if categories:
        want = [c.lower() for c in categories]
        rows = [r for r in rows if any(w in r["category"].lower() for w in want)]
    rows = rows[start:]
    return rows[:count] if count else rows


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def _write_split_lists(data_root: Path, cam_dir: str, names):
    train = sorted(n for n in names if not in_test_split(n))
    test = sorted(n for n in names if in_test_split(n))
    (data_root / f"{cam_dir}_train.txt").write_text("\n".join(train) + ("\n" if train else ""))
    (data_root / f"{cam_dir}_test.txt").write_text("\n".join(test) + ("\n" if test else ""))
    return len(train), len(test)


def prepare(rows, data_root="./data/", download=False, jobs=4, workers=None, delete_nefs=False,
            log=print) -> list[str]:
    """Download (optionally) and preprocess the given RAISE rows. Returns the
    camera folders that now have training pairs. Never stops on one bad file.

    delete_nefs: delete each NEF once its pair is written; downloading and
    preprocessing then run in small batches, so peak disk use stays small."""
    data_root = Path(data_root)
    by_cam: dict[str, list[dict]] = {}
    for r in rows:
        by_cam.setdefault(camera_folder(r["device"]), []).append(r)
    n_workers = workers or os.cpu_count() or 1
    cam_dirs = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        for folder, cam_rows in sorted(by_cam.items()):
            cam_dir = PREFIX + folder
            base, rdir = data_root / cam_dir, data_root / "raise" / "raw" / folder
            log(f"[raise] {folder.replace('_', ' ')} -> {cam_dir}/ ({len(cam_rows)} images)")
            nefs = {r["name"]: rdir / f"{r['name']}.NEF" for r in cam_rows}
            urls = {r["name"]: r["url"] for r in cam_rows}
            # every pair already in the folder counts, also from earlier runs with other rows
            existing = {p.stem for p in (base / "RAW").glob("*.npz") if (base / "RGB" / f"{p.stem}.jpg").exists()} \
                if (base / "RAW").is_dir() else set()
            done = existing & set(nefs)
            todo = [n for n in nefs if n not in done]
            missing = [n for n in todo if not (nefs[n].exists() and nefs[n].stat().st_size > 0)]
            if missing and not download:
                log(f"[raise]   {len(missing)} of {len(nefs)} NEFs not downloaded -- using the "
                    f"{len(nefs) - len(missing)} present (--download to fetch them)")
                todo = [n for n in todo if n not in missing]
                missing = []
            if delete_nefs:  # leftovers from earlier runs
                for n in done:
                    nefs[n].unlink(missing_ok=True)
            need_dl, ok = set(missing), set(existing)
            step = max(8, 4 * max(jobs, n_workers)) if delete_nefs else max(1, len(todo))
            n_dl = n_pp = 0
            for k in range(0, len(todo), step):
                part = todo[k:k + step]
                dl = [n for n in part if n in need_dl]
                if dl:
                    if k == 0:
                        log(f"[raise]   downloading {len(need_dl)} NEF(s)" + (" in batches" if delete_nefs else "") + " ...")
                    fails = []
                    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
                        futs = [ex.submit(fk.download, urls[n], nefs[n]) for n in dl]
                        for f in as_completed(futs):
                            try:
                                f.result()
                            except Exception as e:  # noqa: BLE001
                                fails.append(str(e))
                            n_dl += 1
                            if n_dl % 25 == 0 or n_dl == len(need_dl):
                                log(f"[raise]   {n_dl}/{len(need_dl)} downloaded")
                    if fails:
                        log(f"[raise]   {len(fails)} download(s) failed (re-run to retry): {fails[0]}")
                    part = [n for n in part if nefs[n].exists()]
                if k == 0 and part:
                    log(f"[raise]   preprocessing {len(todo)} NEF(s) ...")
                futs = {pool.submit(fp.preprocess_one, str(nefs[n]), str(base / "RAW"), str(base / "RGB"),
                                    delete_nefs): n for n in part}
                for f in as_completed(futs):
                    status, detail = f.result()
                    if status in ("ok", "skipped"):
                        ok.add(futs[f])
                    else:
                        log(f"[raise]   {status}: {detail}")
                    n_pp += 1
                    if n_pp % 25 == 0 or n_pp == len(todo):
                        log(f"[raise]   {n_pp}/{len(todo)} preprocessed")
            if not ok:
                log(f"[raise]   no pairs for {cam_dir}")
                continue
            n_train, n_test = _write_split_lists(data_root, cam_dir, ok)
            log(f"[raise]   ready: {len(ok)} pairs ({n_train} train / {n_test} test)")
            cam_dirs.append(cam_dir)
    if delete_nefs:
        for d in sorted((data_root / "raise" / "raw").glob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    return cam_dirs


def summary(rows) -> str:
    by_cam: dict[str, int] = {}
    for r in rows:
        by_cam[camera_folder(r["device"])] = by_cam.get(camera_folder(r["device"]), 0) + 1
    return ", ".join(f"{c.replace('_', ' ')}: {n}" for c, n in sorted(by_cam.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare RAISE raws for InvISP training (non-commercial research use).")
    ap.add_argument("--csv", required=True, help="CSV from the RAISE download page")
    ap.add_argument("--download", action="store_true", help="download NEFs that aren't there yet")
    ap.add_argument("--delete-nefs", action="store_true", help="delete each NEF once preprocessed (saves disk)")
    ap.add_argument("-c", "--camera", action="append", default=[],
                    help="only these cameras, e.g. D90 (repeatable/comma; default: all)")
    ap.add_argument("--category", action="append", default=[],
                    help="only rows whose category/keywords contain this, e.g. Outdoor (repeatable/comma)")
    ap.add_argument("--start", type=int, default=0, help="skip the first N selected rows (to prepare in parts)")
    ap.add_argument("--count", type=int, default=0, help="at most N images (default: all selected)")
    ap.add_argument("--csv-order", action="store_true",
                    help="take images in the CSV's order (default: a fixed shuffled order, so any part is a "
                         "representative mix of cameras and scenes)")
    ap.add_argument("--list", action="store_true", help="show what's selected and exit")
    ap.add_argument("--data-path", default="./data/")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="parallel downloads")
    ap.add_argument("--workers", type=int, default=0, help="preprocessing processes (default: all cores)")
    a = ap.parse_args()
    split = lambda xs: [x.strip() for v in xs for x in v.split(",") if x.strip()]  # noqa: E731
    rows = select(read_list(a.csv), split(a.camera), split(a.category), a.start, a.count, a.csv_order)
    print(f"[raise] {len(rows)} image(s) selected" + (f" -- {summary(rows)}" if rows else ""))
    if a.list or not rows:
        sys.exit(0 if rows or a.list else 1)
    cams = prepare(rows, a.data_path, a.download, a.jobs, a.workers or None, a.delete_nefs)
    sys.exit(0 if cams else 1)
