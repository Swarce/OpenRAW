"""tools/compare_models.py: scores checkpoints on exactly the images and the
metric train.py's evaluation uses."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rawpy")
pytest.importorskip("colour_demosaicing")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "data")); sys.path.insert(0, str(ROOT / "tools"))
from .cfa_dng import make_cfa_dng  # noqa: E402


@pytest.fixture
def prepared(tmp_path):
    import fivek_prepare as fp
    data = tmp_path / "data"
    meta = data / "fivek" / "_metadata"; meta.mkdir(parents=True)
    split = {"Testco T1": {"a1": "train", "a2": "test", "a3": "test"}, "Otherco Z9": {"z1": "train", "z2": "test"}}
    for cam, names in split.items():
        raw = data / "fivek" / "raw" / cam.replace(" ", "_"); raw.mkdir(parents=True)
        for n in names:
            make_cfa_dng(str(raw / f"{n}.dng"), "GRBG", h=160, w=224)
    for sp, f in {"train": "training.json", "val": "validation.json", "test": "testing.json"}.items():
        (meta / f).write_text(json.dumps({n: {"urls": {"dng": "http://unused/x.dng", "tiff16": {}},
                                              "camera": {"make": c.split()[0], "model": c.split()[1]}}
                                          for c, ns in split.items() for n, s in ns.items() if s == sp}))
    fp.prepare_cameras(["Testco T1", "Otherco Z9"], str(data) + "/", workers=1, log=lambda *_: None, use_available=True)
    return data


def _random_ckpt(path, seed, full_state=False):
    from openraw.third_party.invisp.model.model import InvISPNet
    torch.manual_seed(seed)
    sd = InvISPNet(channel_in=3, channel_out=3, block_num=8).state_dict()
    torch.save({"net": sd, "optimizer": {}, "ema": sd} if full_state else sd, path)
    return path


def test_scores_match_train_py_evaluation(prepared, tmp_path):
    import compare_models as cm
    import train
    from openraw.invisp_bridge import _load_net
    ck = _random_ckpt(tmp_path / "m.pth", 1)
    subset, cams_per_image, cams = cm.build_eval_set(str(prepared) + "/", eval_images=40, eval_crop=64)
    assert len(subset) == 3 and set(cams_per_image) == {"Testco_T1", "Otherco_Z9"}
    scores = cm.score(str(ck), subset, torch.device("cpu"))
    loader = torch.utils.data.DataLoader(subset, batch_size=1)
    raw, rgb = train.evaluate(_load_net("", "", "cpu", checkpoint=str(ck)), loader, torch.device("cpu"))
    assert sum(r for r, _ in scores) / 3 == pytest.approx(raw, abs=1e-9)
    assert sum(g for _, g in scores) / 3 == pytest.approx(rgb, abs=1e-9)


def test_cli_writes_table_and_csv_and_survives_a_bad_file(prepared, tmp_path):
    import compare_models as cm
    for run in ("a", "b"):
        (tmp_path / run / "checkpoint").mkdir(parents=True)
    a = _random_ckpt(tmp_path / "a" / "checkpoint" / "best.pth", 1)
    b = _random_ckpt(tmp_path / "b" / "checkpoint" / "best.pth", 2, full_state=True)
    bad = tmp_path / "bad.pth"; torch.save({"nothing": torch.zeros(1)}, bad)
    out = tmp_path / "out"
    assert cm.main([str(a), str(b), str(bad), str(ROOT / "pretrained" / "nikon.pth"),
                    "--data_path", str(prepared), "--eval_crop", "64", "--device", "cpu", "--out", str(out)]) == 0
    md = (out / "compare.md").read_text()
    for name in ("a/best.pth", "b/best.pth", "nikon.pth", "Testco_T1 (2)", "Otherco_Z9 (1)"):
        assert name in md  # same-named best.pth files told apart by their run folder
    assert "bad.pth" not in md
    rows = list(csv.DictReader(open(out / "compare.csv")))
    assert len(rows) == 3 * 3 and {r["model"] for r in rows} == {"a/best.pth", "b/best.pth", "nikon.pth"}


def test_full_state_checkpoint_loads_the_weight_average(tmp_path):
    from openraw.invisp_bridge import _load_net
    from openraw.third_party.invisp.model.model import InvISPNet
    torch.manual_seed(0); live = InvISPNet(channel_in=3, channel_out=3, block_num=8).state_dict()
    torch.manual_seed(1); avg = InvISPNet(channel_in=3, channel_out=3, block_num=8).state_dict()
    p = tmp_path / "latest_state.pth"
    torch.save({"net": live, "optimizer": {}, "ema": avg}, p)
    net = _load_net("", "", "cpu", checkpoint=str(p))
    k = next(k for k, v in avg.items() if v.is_floating_point() and v.numel() > 1)
    assert torch.equal(net.state_dict()[k], avg[k])
