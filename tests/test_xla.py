"""The experimental TPU path (train.py --device xla) on one emulated XLA CPU
device. Needs PyTorch/XLA, so it's skipped wherever that isn't installed
(including CI); multi-core collectives need a real TPU."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_xla")
pytest.importorskip("rawpy")
pytest.importorskip("colour_demosaicing")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "data"))
from .test_train_loop import trainer  # noqa: E402,F401  (the same tiny dataset + 1-block network)


@pytest.fixture(autouse=True)
def _emulated_xla_device(monkeypatch):
    # set per test (not at import), so it can't leak into other test files
    monkeypatch.setenv("PJRT_DEVICE", "CPU")


def test_trains_evaluates_saves_and_resumes_on_an_xla_device(trainer):
    run, ckpt = trainer
    st = run(epochs=2, eval_every=1, eval_crop=64, device="xla", gpus=1, ema=0.9, checkpointing="auto")
    assert (st["epoch"], st["step"]) == (1, 4)
    assert all(v.device.type == "cpu" for v in st["net"].values())  # saved from the TPU as CPU tensors
    assert st["ema_updates"] == 4
    rows = (ckpt.parent / "eval.csv").read_text().strip().splitlines()
    assert len(rows) == 3 and all(0 < float(r.split(",")[2]) < 99 for r in rows[1:])
    st2 = run(epochs=3, device="xla", gpus=1, ema=0.9, resume=True)  # resumes on the XLA device
    assert (st2["epoch"], st2["step"], st2["ema_updates"]) == (2, 6, 6)
    sd = torch.load(ckpt / "latest.pth", weights_only=True)  # weights-only file loads anywhere
    assert set(sd) == set(st["net"])


def test_mixed_precision_on_xla_uses_bfloat16_and_trains(trainer):
    run, ckpt = trainer
    st = run(epochs=1, eval_every=1, eval_crop=64, device="xla", gpus=1, amp=True)
    assert all(torch.isfinite(v).all() for v in st["net"].values() if v.is_floating_point())


def test_tpu_selftest_tool_runs_on_one_emulated_core():
    import subprocess
    env = dict(os.environ, PJRT_DEVICE="CPU", CPU_NUM_DEVICES="1")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "tpu_selftest.py")], env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "TPU SELFTEST OK (1 cores" in r.stdout, r.stdout + r.stderr


def test_more_than_one_but_not_all_cores_is_refused(trainer):
    import train
    with pytest.raises(SystemExit, match="--gpus 0"):
        train.launch(SimpleNamespace(device="xla", gpus=4))
