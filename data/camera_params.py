"""Camera settings (exposure time, f-number, ISO, focal length) for training
pairs prepared before they were stored -- for train.py --exif_cond.

Pairs prepared from now on carry them in their .npz. Older pairs don't, and
their DNGs/NEFs are usually deleted after preparing. The settings sit in the
first few kilobytes of each raw file, so this fetches only that much of each
one (an HTTP range request; a larger piece, or the whole file, when a camera
puts its EXIF further in) and writes one table:

    <data_root>/camera_params.json   {"<camera folder>/<name>": [exposure_time, f_number, iso, focal_length]}

which the training data loader reads (unknown values are null). Already-listed
pairs are skipped, so an interrupted run continues where it stopped.

    python data/camera_params.py --data-path ./data/                      # FiveK folders
    python data/camera_params.py --data-path ./data/ --raise-csv RAISE_2k.csv   # + RAISE folders
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fivek_download as fk  # noqa: E402
from openraw import camera_params as cp  # noqa: E402

TABLE = "camera_params.json"
STEPS = (64 << 10, 1 << 20, 0)  # bytes to fetch per try; 0 = the whole file


def fetch_head(url: str, n: int, timeout: int = 60) -> bytes:
    """The first n bytes of url (all of it for n=0). Asks for a range; a server
    that ignores it still only gets read up to n bytes."""
    headers = {"User-Agent": fk.USER_AGENT}
    if n:
        headers["Range"] = f"bytes=0-{n - 1}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return r.read(n) if n else r.read()


def params_from_url(url: str, fetch=fetch_head) -> list:
    """[exposure_time, f_number, iso, focal_length], None where unknown."""
    for n in STEPS:
        try:
            v = cp.as_vector(cp.from_bytes(fetch(url, n)))
            return [None if math.isnan(x) else x for x in v]
        except cp._Truncated:  # EXIF lies further in: fetch more
            continue
    return [None] * 4


def prepared_pairs(data_root: Path) -> dict[str, list[str]]:
    """{camera folder: [names]} for every prepared pair whose .npz lacks camera_params."""
    import numpy as np
    out = {}
    for d in sorted(p for p in data_root.iterdir() if (p / "RAW").is_dir()):
        names = []
        for npz in sorted((d / "RAW").glob("*.npz")):
            if npz.name.endswith(".part.npz"):
                continue
            try:
                with np.load(npz) as z:
                    if "camera_params" in z.files:
                        continue
            except Exception:  # noqa: BLE001 -- unreadable: the loader will complain, not us
                continue
            names.append(npz.stem)
        if names:
            out[d.name] = names
    return out


def url_index(data_root: Path, raise_csvs=()) -> dict[str, str]:
    """name -> raw file URL, from the FiveK metadata and any RAISE CSVs."""
    urls = {}
    try:
        items = fk.load_metadata(data_root / "fivek" / "_metadata", list(fk.SPLIT_FILES))
        urls.update({n: e["urls"]["dng"] for n, e in items.items()})
    except Exception as e:  # noqa: BLE001
        print(f"[params] FiveK metadata unavailable ({type(e).__name__}: {e}); FiveK pairs skipped")
    if raise_csvs:
        import raise_prepare
        for c in raise_csvs:
            urls.update({r["name"]: r["url"] for r in raise_prepare.read_list(c)})
    return urls


def backfill(data_root="./data/", out=None, raise_csvs=(), jobs=16, fetch=fetch_head, log=print) -> Path:
    data_root = Path(data_root)
    out = Path(out) if out else data_root / TABLE
    table = json.loads(out.read_text()) if out.is_file() else {}
    pairs = prepared_pairs(data_root)
    todo = [(cam, n) for cam, names in pairs.items() for n in names if f"{cam}/{n}" not in table]
    if not todo:
        log(f"[params] nothing to do ({len(table)} pairs listed in {out})")
        return out
    urls = url_index(data_root, raise_csvs)
    no_url = [k for k in todo if k[1] not in urls]
    if no_url:
        log(f"[params] no download URL for {len(no_url)} pair(s) (e.g. {no_url[0][0]}/{no_url[0][1]}); "
            f"RAISE folders need --raise-csv")
    todo = [k for k in todo if k[1] in urls]
    log(f"[params] reading EXIF for {len(todo)} pair(s) ({jobs} at a time) -> {out}")

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(table, indent=0, sort_keys=True))
        os.replace(tmp, out)

    done = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(params_from_url, urls[n], fetch): (cam, n) for cam, n in todo}
        for f in as_completed(futs):
            cam, n = futs[f]
            try:
                table[f"{cam}/{n}"] = f.result()
            except Exception as e:  # noqa: BLE001 -- network: retried on the next run
                failed += 1
                if failed <= 3:
                    log(f"[params]   {cam}/{n}: {type(e).__name__}: {e}")
            done += 1
            if done % 200 == 0 or done == len(todo):
                save()
                log(f"[params]   {done}/{len(todo)}")
    save()
    known = sum(1 for v in table.values() if any(x is not None for x in v))
    log(f"[params] {len(table)} pairs listed, {known} with settings" + (f"; {failed} failed (re-run)" if failed else ""))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-path", default="./data/")
    ap.add_argument("--out", help=f"table to write (default <data-path>/{TABLE})")
    ap.add_argument("--raise-csv", action="append", default=[], help="RAISE CSV(s) with the NEF URLs")
    ap.add_argument("-j", "--jobs", type=int, default=16)
    a = ap.parse_args(argv)
    backfill(a.data_path, a.out, a.raise_csv, a.jobs)


if __name__ == "__main__":
    main()
