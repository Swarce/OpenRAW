"""data/raise_prepare.py and the RAISE paths in train.py and the Kaggle runner,
offline: synthetic Bayer raws served as .NEF from a local HTTP server, listed
in a CSV shaped like the one from the RAISE download page."""

from __future__ import annotations

import functools
import http.server
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("rawpy")
pytest.importorskip("colour_demosaicing")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "kaggle"))
import raise_prepare as rp  # noqa: E402
import fivek_prepare as fp  # noqa: E402

from .cfa_dng import make_cfa_dng  # noqa: E402

IMAGES = {"r0001aa": ("Nikon D90", "Outdoor;Nature"), "r0002bb": ("Nikon D90", "Indoor"),
          "r0003cc": ("Nikon D7000", "Outdoor"), "r0004dd": ("Nikon D40", "People")}


class _Q(http.server.SimpleHTTPRequestHandler):
    hits: list = []
    def log_message(self, *a):
        pass
    def do_GET(self):
        _Q.hits.append(self.path)
        super().do_GET()


def _csv(base, images=IMAGES, bom=True):
    lines = ["File,TIFF,NEF,Date,Device,Image Size,Keywords"]
    for n, (dev, kw) in images.items():
        lines.append(f'{n},{base}/TIFF/{n}.TIF,{base}/NEF/{n}.NEF,2013:05:01,{dev},4288x2848,"{kw}"')
    return ("﻿" if bom else "") + "\n".join(lines) + "\n"


@pytest.fixture
def raise_server(tmp_path):
    srv = tmp_path / "server" / "NEF"
    srv.mkdir(parents=True)
    for n in IMAGES:
        make_cfa_dng(str(srv / f"{n}.NEF"), "GRBG", h=320, w=432)  # rawpy reads by content, not extension
    _Q.hits = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Q, directory=str(tmp_path / "server")))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    csv = tmp_path / "RAISE_1k.csv"
    csv.write_text(_csv(base), encoding="utf-8")
    yield tmp_path, csv
    httpd.shutdown()


def test_reads_the_list_and_selects_by_camera_category_and_range(raise_server):
    _, csv = raise_server
    rows = rp.read_list(csv)
    assert [r["name"] for r in rows] == list(IMAGES)
    assert rows[0]["url"].endswith("/NEF/r0001aa.NEF") and rows[0]["device"] == "Nikon D90"
    assert sorted(r["name"] for r in rp.select(rows, cameras=["D90"])) == ["r0001aa", "r0002bb"]
    assert sorted(r["name"] for r in rp.select(rows, categories=["outdoor"])) == ["r0001aa", "r0003cc"]
    assert [r["name"] for r in rp.select(rows, start=1, count=2, csv_order=True)] == ["r0002bb", "r0003cc"]
    assert rp.camera_dir_name("NIKON D7000") == rp.camera_dir_name("Nikon D7000") == "RAISE_Nikon_D7000"


def test_list_without_nef_column_is_a_clear_error(tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("File,TIFF\nr1,http://x/r1.TIF\n")
    with pytest.raises(SystemExit, match="NEF"):
        rp.read_list(bad)


def test_unsafe_ids_are_skipped(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("File,NEF,Device\n../../evil,http://x/a.NEF,Nikon D90\nok1,http://x/ok1.NEF,Nikon D90\n")
    assert [r["name"] for r in rp.read_list(p)] == ["ok1"]


def test_prepare_downloads_preprocesses_splits_and_deletes(raise_server):
    tmp, csv = raise_server
    data = tmp / "data"
    cams = rp.prepare(rp.read_list(csv), str(data) + "/", download=True, jobs=2, workers=1,
                      delete_nefs=True, log=lambda *_: None)
    assert cams == ["RAISE_Nikon_D40", "RAISE_Nikon_D7000", "RAISE_Nikon_D90"]
    for n, (dev, _) in IMAGES.items():
        cam = rp.camera_dir_name(dev)
        assert (data / cam / "RAW" / f"{n}.npz").exists() and (data / cam / "RGB" / f"{n}.jpg").exists()
        listed = (data / f"{cam}_train.txt").read_text().split() + (data / f"{cam}_test.txt").read_text().split()
        assert n in listed
        assert (n in (data / f"{cam}_test.txt").read_text().split()) == rp.in_test_split(n)
    assert not list((data / "raise").rglob("*.NEF"))  # deleted after preprocessing
    # second run: nothing downloaded again, pairs kept
    hits = len(_Q.hits)
    rp.prepare(rp.read_list(csv), str(data) + "/", download=True, workers=1, delete_nefs=True, log=lambda *_: None)
    assert len(_Q.hits) == hits


def test_preparing_in_parts_keeps_earlier_pairs_in_the_lists(raise_server):
    tmp, csv = raise_server
    data = tmp / "data"
    rows = rp.select(rp.read_list(csv), cameras=["D90"])
    rp.prepare(rows[:1], str(data) + "/", download=True, workers=1, log=lambda *_: None)
    rp.prepare(rows[1:], str(data) + "/", download=True, workers=1, log=lambda *_: None)
    listed = set((data / "RAISE_Nikon_D90_train.txt").read_text().split()
                 + (data / "RAISE_Nikon_D90_test.txt").read_text().split())
    assert listed == {"r0001aa", "r0002bb"}


def test_parts_are_a_mix_and_never_overlap():
    """The real list runs in long same-camera stretches; parts must not."""
    rows = [{"name": f"r{i:06x}", "url": "http://x", "category": "",
             "device": "Nikon D90" if i < 3000 else "Nikon D7000"} for i in range(6000)]
    parts = [rp.select(rows, start=s, count=1000) for s in range(0, 6000, 1000)]
    names = [r["name"] for p in parts for r in p]
    assert len(names) == len(set(names)) == 6000  # every image exactly once
    for p in parts:
        share_d90 = sum(r["device"] == "Nikon D90" for r in p) / len(p)
        assert 0.4 < share_d90 < 0.6  # each part ~ the whole list's 50/50 mix
    assert parts[0] == rp.select(list(reversed(rows)), start=0, count=1000)  # list order doesn't matter


def test_test_split_is_about_ten_percent_and_stable():
    names = [f"r{i:06x}" for i in range(5000)]
    share = sum(map(rp.in_test_split, names)) / len(names)
    assert 0.08 < share < 0.12
    assert [rp.in_test_split(n) for n in names[:50]] == [rp.in_test_split(n) for n in names[:50]]


def test_cli_list_mode(raise_server):
    _, csv = raise_server
    out = subprocess.run([sys.executable, str(ROOT / "data" / "raise_prepare.py"), "--csv", str(csv),
                          "--camera", "D90,D40", "--list"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "3 image(s) selected" in out.stdout and "Nikon D90: 2" in out.stdout


def test_training_finds_raise_cameras_next_to_fivek_ones(raise_server):
    """train.py --all-downloaded / the loader see RAISE_* folders as cameras."""
    torch = pytest.importorskip("torch")
    tmp, csv = raise_server
    data = tmp / "data"
    rp.prepare(rp.read_list(csv), str(data) + "/", download=True, workers=1, log=lambda *_: None)
    cams = fp.prepared_other_sources(str(data) + "/")
    assert cams == ["RAISE_Nikon_D40", "RAISE_Nikon_D7000", "RAISE_Nikon_D90"]
    sys.path.insert(0, str(ROOT))
    from dataset.FiveK_dataset import FiveKDatasetTrain
    ds = FiveKDatasetTrain(SimpleNamespace(debug_mode=False, data_path=str(data) + "/", camera=cams, gamma=True))
    n_train = sum(not rp.in_test_split(n) for n in IMAGES)
    assert len(ds) == n_train
    assert torch.isfinite(ds[0]["input_raw"]).all()


def test_train_py_prepare_only_with_only_raise_data(raise_server):
    pytest.importorskip("torch")  # train.py imports it at the top
    tmp, csv = raise_server
    data = tmp / "data"
    rp.prepare(rp.read_list(csv), str(data) + "/", download=True, workers=1, log=lambda *_: None)
    out = subprocess.run([sys.executable, "train.py", "--all-downloaded", "--prepare-only",
                          "--data_path", str(data) + "/"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "prepared: RAISE_Nikon_D40, RAISE_Nikon_D7000, RAISE_Nikon_D90" in out.stdout


def test_kaggle_runner_prepares_raise_from_an_attached_csv(raise_server, monkeypatch):
    import kaggle_runner as kr
    tmp, csv = raise_server
    k = tmp / "kaggle"
    for d in ("input/raise-list", "working", "scratch"):
        (k / d).mkdir(parents=True)
    (k / "input" / "raise-list" / "RAISE_1k.csv").write_text(csv.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(kr, "INPUT", k / "input")
    monkeypatch.setattr(kr, "WORKING", k / "working")
    monkeypatch.setattr(kr, "SCRATCH", k / "scratch")
    assert kr.find_raise_csv(k / "input").name == "RAISE_1k.csv"
    assert kr.run(raise_csv=True, raise_cameras=["D90"], budget_gb=5) == "prepared RAISE"
    out = k / "working" / kr.DATA_DIR
    assert sorted(p.name for p in out.glob("*_train.txt")) == ["RAISE_Nikon_D90_train.txt"]
    assert len(list(out.glob("*/RAW/*.npz"))) == 2
    assert not (out / "raise").exists()
    # the result is a prepared-data root the training side recognises
    assert kr.find_data_roots(k / "working") == [out]


def test_kaggle_raise_prepare_respects_the_budget(raise_server, monkeypatch):
    import kaggle_runner as kr
    tmp, csv = raise_server
    monkeypatch.setattr(kr, "EST_PAIR_MB", 1000.0)  # pretend pairs are 1 GB: nothing fits in 0.5 GB
    out = tmp / "out"
    assert kr.prepare_raise(csv, budget_gb=0.5, out=out) == []
    assert not list(out.glob("*/RAW/*.npz"))
