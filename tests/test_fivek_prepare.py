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
sys.path.insert(0, str(ROOT))
from dataset import mosaic_store  # noqa: E402


def _decoded(npz_path):
    """Demosaiced linear image from a stored pair, whatever the format."""
    z = np.load(npz_path)
    if mosaic_store.is_mosaic(z):
        m, p = mosaic_store.unpack(z)
        return np.clip(mosaic_store.demosaic(m, p), 0, float(z["white_level"])), z
    return z["raw"], z

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
    raw, z = _decoded(tmp_path / "RAW" / f"x_{pattern}.npz")
    assert str(z["cfa_pattern"]) == pattern
    for c, (a, b) in enumerate(THIRDS(raw.shape[1])):
        means = raw[8:-8, a + 8:b - 8].reshape(-1, 3).mean(0)  # interior: away from demosaic edges
        assert means.argmax() == c, f"{pattern}: third {c} dominated by channel {means.argmax()}"


def test_preprocess_subtracts_black_and_stores_white_level(tmp_path):
    dng = tmp_path / "x.dng"
    make_cfa_dng(str(dng), "GRBG", black=256, white=4095)
    fp.preprocess_one(str(dng), str(tmp_path / "RAW"), str(tmp_path / "RGB"))
    raw, z = _decoded(tmp_path / "RAW" / "x.npz")
    assert float(z["white_level"]) == pytest.approx(4095 - 256)
    assert raw.min() == pytest.approx(0, abs=1)  # unlit photosites sit at 0 after black subtraction
    # clipped to the sensor range: bilinear demosaicing overshoots ~1.5x at borders
    assert raw.max() <= 4095 - 256


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
    for n in names:  # DNGs land in fivek_download.py's layout
        assert (data_root / "fivek" / "raw" / "Testco_T1" / f"{n}.dng").exists()
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
    for d in (data_root / "fivek" / "raw" / "Testco_T1").glob("*.dng"):
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


def _copy_from_server(data_root, names, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        (dest / f"{n}.dng").write_bytes((data_root.parent / "server" / "dng" / f"{n}.dng").read_bytes())


def test_all_downloaded_detects_camera_folders_and_uses_whats_there(fivek_server):
    """--all-downloaded: scan data/fivek/raw/<Make_Model>/ (fivek_download.py's
    layout), map folders back to FiveK cameras, train on what's present."""
    data_root, names = fivek_server
    raw = data_root / "fivek" / "raw"
    _copy_from_server(data_root, ["a0001-x", "a0003-z"], raw / "Testco_T1")  # 2 of 3 downloaded
    _copy_from_server(data_root, ["a0002-y"], raw / "Not_A_Camera")          # stray folder
    (raw / "Empty_Folder").mkdir()
    logs = []
    assert fp.downloaded_cameras(str(data_root) + "/", log=logs.append) == ["Testco T1"]
    assert any("Not_A_Camera" in l for l in logs)
    cams = fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=False, workers=1,
                              log=lambda *_: None, use_available=True)
    assert cams == ["Testco_T1"]
    assert (data_root / "Testco_T1_train.txt").read_text().split() == ["a0001-x"]
    assert (data_root / "Testco_T1_test.txt").read_text().split() == ["a0003-z"]
    assert not _Quiet.hits  # nothing downloaded


def test_dngs_in_the_old_location_still_count(fivek_server):
    data_root, names = fivek_server
    _copy_from_server(data_root, list(names), data_root / "Testco_T1" / "DNG")
    fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=False, workers=1, log=lambda *_: None)
    assert len(list((data_root / "Testco_T1" / "RAW").glob("*.npz"))) == 3
    assert not _Quiet.hits


def test_loader_skips_listed_images_that_arent_prepared(fivek_server):
    pytest.importorskip("torch")
    data_root, _ = fivek_server
    cams = fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=True, workers=1, log=lambda *_: None)
    with open(data_root / "Testco_T1_train.txt", "a") as f:  # e.g. a published list, partially downloaded
        f.write("a9999-never-downloaded\n")
    sys.path.insert(0, str(ROOT))
    from dataset.FiveK_dataset import FiveKDatasetTrain
    ds = FiveKDatasetTrain(SimpleNamespace(debug_mode=False, data_path=str(data_root) + "/", camera=cams, gamma=True))
    assert len(ds) == 2



def _old_format_npz(dng, npz_path):
    """A pair in the PREVIOUS storage format (demosaiced float32), as the last
    version of fivek_prepare wrote it -- to test in-place shrinking."""
    import rawpy
    with rawpy.imread(str(dng)) as raw:
        pattern = fp.cfa_pattern(raw)
        data = raw.raw_image_visible.astype(np.float32)
        black = np.asarray(raw.black_level_per_channel, np.float32)[raw.raw_colors_visible]
        white = float(raw.white_level) - float(np.mean(raw.black_level_per_channel))
        data = np.maximum(data - black, 0)
        wb = np.asarray(raw.camera_whitebalance, np.float32)
    de = np.clip(mosaic_store.demosaic(data, pattern), 0, white)
    np.savez(npz_path, raw=de, wb=wb, white_level=np.float32(white), cfa_pattern=pattern)
    return de


def test_new_format_is_much_smaller_and_bit_exact(tmp_path):
    dng = tmp_path / "x.dng"
    make_cfa_dng(str(dng), "GRBG", h=400, w=600)
    fp.preprocess_one(str(dng), str(tmp_path / "RAW"), str(tmp_path / "RGB"))
    old = tmp_path / "old.npz"
    expected = _old_format_npz(dng, old)
    new = tmp_path / "RAW" / "x.npz"
    assert new.stat().st_size * 8 < old.stat().st_size
    assert np.array_equal(_decoded(new)[0], expected)  # same training data, bit for bit


def test_old_format_pairs_are_shrunk_in_place_exactly(fivek_server):
    data_root, names = fivek_server
    fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=True, workers=1, log=lambda *_: None)
    raw = data_root / "Testco_T1" / "RAW"
    expected = {}
    for n in names:  # rewrite every pair in the previous format
        expected[n] = _old_format_npz(data_root / "fivek" / "raw" / "Testco_T1" / f"{n}.dng", raw / f"{n}.npz")
    big = sum((raw / f"{n}.npz").stat().st_size for n in names)
    logs = []
    fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=False, workers=1, log=logs.append)
    assert any("shrinking 3 pair" in l for l in logs)
    assert sum((raw / f"{n}.npz").stat().st_size for n in names) * 8 < big
    for n in names:
        got = _decoded(raw / f"{n}.npz")[0]
        # bit-exact except within 2 px of the image edge, where the old format
        # had clipped the demosaic's edge overshoot at the white level -- those
        # values were already lost in the old file (see mosaic_store.remosaic)
        assert np.array_equal(got[2:-2, 2:-2], expected[n][2:-2, 2:-2])


def test_delete_dngs_removes_them_and_cameras_are_still_found(fivek_server):
    data_root, names = fivek_server
    cams = fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=True, workers=1,
                              log=lambda *_: None, delete_dngs=True)
    assert cams == ["Testco_T1"]
    assert not list((data_root / "fivek" / "raw" / "Testco_T1").glob("*.dng"))  # all deleted
    assert len(list((data_root / "Testco_T1" / "RAW").glob("*.npz"))) == 3     # pairs kept
    # --all-downloaded still finds the camera via its prepared pairs, and nothing re-downloads
    hits = len(_Quiet.hits)
    assert fp.downloaded_cameras(str(data_root) + "/", log=lambda *_: None) == ["Testco T1"]
    fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=False, workers=1,
                       log=lambda *_: None, use_available=True)
    assert len(_Quiet.hits) == hits


def test_delete_dngs_cleans_up_leftovers_from_earlier_runs(fivek_server):
    data_root, _ = fivek_server
    fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=True, workers=1, log=lambda *_: None)
    assert len(list((data_root / "fivek" / "raw" / "Testco_T1").glob("*.dng"))) == 3
    logs = []
    fp.prepare_cameras(["Testco T1"], str(data_root) + "/", workers=1, log=logs.append, delete_dngs=True)
    assert not list((data_root / "fivek" / "raw" / "Testco_T1").glob("*.dng"))
    assert any("GB freed" in l for l in logs)


def test_loader_crop_matches_full_frame_demosaic(fivek_server, monkeypatch):
    """Demosaicing only the training crop must give exactly what demosaicing
    the full frame and cropping would."""
    pytest.importorskip("torch")
    import random as _random
    data_root, _ = fivek_server
    cams = fp.prepare_cameras(["Testco T1"], str(data_root) + "/", download=True, workers=1, log=lambda *_: None)
    from dataset.FiveK_dataset import FiveKDatasetTrain
    ds = FiveKDatasetTrain(SimpleNamespace(debug_mode=False, data_path=str(data_root) + "/", camera=cams, gamma=False))
    monkeypatch.setattr(ds, "random_rotate", lambda a, b: (a, b))
    monkeypatch.setattr(ds, "random_flip", lambda a, b: (a.copy(), b.copy()))
    _random.seed(3)
    sample = ds[0]["input_raw"].numpy().transpose(1, 2, 0)
    _random.seed(3)
    z = np.load(ds.data["input_RAWs_WBs"][0])
    m, p = mosaic_store.unpack(z)
    full = np.clip(mosaic_store.demosaic(m, p), 0, float(z["white_level"]))
    y = _random.randint(0, max(0, m.shape[0] - 256)) // 2 * 2
    x = _random.randint(0, max(0, m.shape[1] - 256)) // 2 * 2
    wb = z["wb"] / z["wb"].max()
    expected = full[y:y + 256, x:x + 256] * wb[:-1] / float(z["white_level"])
    assert np.allclose(sample, expected, atol=1e-5)
