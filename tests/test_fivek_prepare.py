"""
FiveK training-data pipeline (data/fivek_prepare.py + patched loader), tested
offline with synthetic Bayer DNGs served from a local HTTP server behind fake
FiveK metadata -- the real download -> preprocess -> lists -> loader path.
"""

from __future__ import annotations

import functools
import http.server
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rawpy")
pytest.importorskip("colour_demosaicing")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
import fivek_prepare as fp  # noqa: E402

from .cfa_dng import make_cfa_dng  # noqa: E402

THIRDS = lambda w: ((0, w // 3), (w // 3, 2 * w // 3), (2 * w // 3, w))


@pytest.mark.parametrize("pattern", ["RGGB", "GRBG", "BGGR", "GBRG"])
def test_preprocess_reads_cfa_pattern_from_the_file(tmp_path, pattern):
    """Upstream hardcoded RGGB; any other sensor got red/blue/green swapped.
    Synthetic scene: left third red, middle green, right blue."""
    dng = tmp_path / f"x_{pattern}.dng"
    make_cfa_dng(str(dng), pattern)
    status, _ = fp.preprocess_one(str(dng), str(tmp_path / "RAW"), str(tmp_path / "RGB"))
    assert status == "ok"
    z = np.load(tmp_path / "RAW" / f"x_{pattern}.npz")
    assert str(z["cfa_pattern"]) == pattern
    raw = z["raw"]
    for c, (a, b) in enumerate(THIRDS(raw.shape[1])):
        means = raw[8:-8, a + 8:b - 8].reshape(-1, 3).mean(0)  # interior: away from demosaic edges
        assert means.argmax() == c, f"{pattern}: third {c} dominated by channel {means.argmax()}"


def test_preprocess_subtracts_black_and_stores_white_level(tmp_path):
    dng = tmp_path / "x.dng"
    make_cfa_dng(str(dng), "GRBG", black=256, white=4095)
    fp.preprocess_one(str(dng), str(tmp_path / "RAW"), str(tmp_path / "RGB"))
    z = np.load(tmp_path / "RAW" / "x.npz")
    assert float(z["white_level"]) == pytest.approx(4095 - 256)
    assert z["raw"].min() == pytest.approx(0, abs=1)  # unlit photosites sit at 0 after black subtraction
    # clipped to the sensor range: bilinear demosaicing overshoots ~1.5x at borders
    assert z["raw"].max() <= 4095 - 256


class _Quiet(http.server.SimpleHTTPRequestHandler):
    hits = []
    def log_message(self, *a):
        pass
    def do_GET(self):
        _Quiet.hits.append(self.path)
        super().do_GET()


@pytest.fixture
def fivek_server(tmp_path):
    """Local stand-in for data.csail.mit.edu + pre-seeded FiveK metadata."""
    srv_root = tmp_path / "server" / "dng"
    srv_root.mkdir(parents=True)
    names = {"a0001-x": "train", "a0002-y": "val", "a0003-z": "test"}
    for i, n in enumerate(names):
        make_cfa_dng(str(srv_root / f"{n}.dng"), "GRBG", h=320, w=432)
    _Quiet.hits = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(tmp_path / "server")))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    data_root = tmp_path / "data"
    meta = data_root / "fivek" / "_metadata"
    meta.mkdir(parents=True)
    split_file = {"train": "training.json", "val": "validation.json", "test": "testing.json"}
    for split, fname in split_file.items():
        entries = {n: {"urls": {"dng": f"{base}/dng/{n}.dng", "tiff16": {}}, "camera": {"make": "Testco", "model": "T1"},
                       "license": "Adobe", "categories": {}} for n, s in names.items() if s == split}
        (meta / fname).write_text(json.dumps(entries))
    yield data_root, names
    httpd.shutdown()


def test_prepare_downloads_only_whats_missing_and_writes_official_split(fivek_server):
    data_root, names = fivek_server
    cams = fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=True, jobs=2, workers=1, log=lambda *_: None)
    assert cams == ["Testco_T1"]
    base = data_root / "Testco_T1"
    for n in names:
        assert (base / "DNG" / f"{n}.dng").exists()
        assert (base / "RAW" / f"{n}.npz").exists() and (base / "RGB" / f"{n}.jpg").exists()
    # FiveK official split: train + val -> train, test -> test
    assert (data_root / "Testco_T1_train.txt").read_text().split() == ["a0001-x", "a0002-y"]
    assert (data_root / "Testco_T1_test.txt").read_text().split() == ["a0003-z"]
    # second run: nothing downloaded again
    first = len(_Quiet.hits)
    fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=True, workers=1, log=lambda *_: None)
    assert len(_Quiet.hits) == first


def test_prepare_without_download_flag_reports_missing_files(fivek_server):
    data_root, _ = fivek_server
    with pytest.raises(SystemExit, match="--download"):
        fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=False, log=lambda *_: None)


def test_finished_pairs_count_even_if_dngs_were_deleted(fivek_server):
    data_root, names = fivek_server
    fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=True, workers=1, log=lambda *_: None)
    for d in (data_root / "Testco_T1" / "DNG").glob("*.dng"):
        d.unlink()  # user freed disk space after preprocessing
    fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=False, workers=1, log=lambda *_: None)


def test_loader_pools_cameras_normalizes_per_image_and_trains_one_step(fivek_server):
    """End to end: prepared data -> patched loader (multi-camera, per-image white
    level) -> one real InvISP forward/backward step on CPU."""
    torch = pytest.importorskip("torch")
    data_root, _ = fivek_server
    cams = fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=True, workers=1, log=lambda *_: None)
    sys.path.insert(0, str(ROOT))
    from dataset.FiveK_dataset import FiveKDatasetTrain
    opt = SimpleNamespace(debug_mode=False, data_path=str(data_root) + "/", camera=cams + cams, gamma=True)
    ds = FiveKDatasetTrain(opt)
    assert len(ds) == 4  # 2 train images x camera listed twice: pooling works
    s = ds[0]
    raw = s["input_raw"].numpy()
    assert 0.0 <= raw.min() and raw.max() <= 1.0 + 1e-6  # normalized by THIS image's white level
    assert raw.max() > 0.5  # ...and not crushed by upstream's hardcoded 16383 for a 12-bit sensor

    from openraw.third_party.invisp.model.model import InvISPNet
    net = InvISPNet(channel_in=3, channel_out=3, block_num=2)
    opt_ = torch.optim.Adam(net.parameters(), lr=1e-4)
    x = s["input_raw"][None]
    rgb = net(x)
    loss = (rgb - s["target_rgb"][None]).abs().mean() + (net(rgb, rev=True) - x).abs().mean()
    loss.backward(); opt_.step()
    assert torch.isfinite(loss)
