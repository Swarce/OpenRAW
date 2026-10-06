"""
train.py's real training loop: parallel data loading, full-state checkpoints,
exact resume, and resume from weights-only checkpoints. Uses a 1-block InvISP
so it fits small machines (the full 8-block net needs ~4 GB for one CPU step);
everything else is train.py's own code. Skipped without torch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rawpy")
pytest.importorskip("colour_demosaicing")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "data"))
from .cfa_dng import make_cfa_dng  # noqa: E402


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    import fivek_prepare as fp
    data = tmp_path / "data"
    raw = data / "fivek" / "raw" / "Testco_T1"; raw.mkdir(parents=True)
    meta = data / "fivek" / "_metadata"; meta.mkdir(parents=True)
    split = {"a1": "train", "a2": "train", "a3": "test"}
    for n in split:
        make_cfa_dng(str(raw / f"{n}.dng"), "GRBG", h=320, w=432)
    for sp, f in {"train": "training.json", "val": "validation.json", "test": "testing.json"}.items():
        (meta / f).write_text(json.dumps({n: {"urls": {"dng": "http://unused/x.dng", "tiff16": {}},
                                              "camera": {"make": "Testco", "model": "T1"}}
                                          for n, s in split.items() if s == sp}))
    cams = fp.prepare_cameras(["Testco T1"], str(data) + "/", workers=1, log=lambda *_: None, use_available=True)

    monkeypatch.chdir(ROOT)
    import train
    from openraw.third_party.invisp.model.model import InvISPNet
    from openraw.third_party.invisp.utils.JPEG import DiffJPEG
    monkeypatch.setattr(train, "InvISPNet", lambda **kw: InvISPNet(channel_in=3, channel_out=3, block_num=1))
    monkeypatch.setattr(train, "DiffJPEG", DiffJPEG(differentiable=True, quality=90), raising=False)
    out = str(tmp_path / "exps") + "/"

    def run(**over):
        kw = dict(task="t", data_path=str(data) + "/", batch_size=1, debug_mode=False, gamma=True, camera=cams,
                  rgb_weight=1, out_path=out, resume=False, loss="L1", lr=1e-4, aug=True, workers=1,
                  epochs=1, start_epoch=None, device="cpu")
        kw.update(over)
        train.main(SimpleNamespace(**kw))
        return train._load(out + "t/checkpoint/latest_state.pth", "cpu")

    return run, Path(out) / "t" / "checkpoint"


def test_train_with_workers_then_resume_exactly(trainer):
    run, ckpt = trainer
    st = run(epochs=1, workers=1)  # 2 train images -> 2 steps/epoch
    assert (st["epoch"], st["step"], st["scheduler"]["last_epoch"]) == (0, 2, 1)
    assert (ckpt / "latest.pth").exists()  # weights-only file stays usable by `openraw --invisp`
    st = run(epochs=2, resume=True)
    assert (st["epoch"], st["step"], st["scheduler"]["last_epoch"]) == (1, 4, 2)
    assert any("exp_avg" in v for v in st["optimizer"]["state"].values())  # Adam moments saved
    assert not list(ckpt.glob("*.tmp"))  # atomic saves leave no temp files


def test_resume_from_weights_only_checkpoint_with_start_epoch(trainer):
    """Runs started before full-state checkpoints only have latest.pth."""
    run, ckpt = trainer
    run(epochs=1)
    (ckpt / "latest_state.pth").unlink()
    st = run(epochs=3, resume=True, start_epoch=2)
    # continued at epoch 2 with the LR schedule fast-forwarded and the step count estimated
    assert (st["epoch"], st["step"], st["scheduler"]["last_epoch"]) == (2, 6, 3)
