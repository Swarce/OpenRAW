"""tools/amp_check.py runs end to end (small network, CPU, synthetic crops)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))


def test_amp_check_reports_speed_accuracy_and_a_verdict(capsys, monkeypatch):
    import contextlib
    import amp_check
    import torch
    import train
    try:
        native = bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
    except Exception:  # noqa: BLE001
        native = False
    if not native:  # CPU without native bfloat16 (CI runners): check the tool's plumbing only
        monkeypatch.setattr(train, "_amp_context", lambda *a: contextlib.nullcontext())
    # tiny on purpose: CPUs without native bfloat16 (e.g. CI runners) run it very slowly
    assert amp_check.main(["--blocks", "1", "--crop", "64", "--timing_steps", "2", "--train_steps", "3",
                           "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    for s in ("step time: float32", "gradient cosine", "round-trip", "held-out raw PSNR", "VERDICT"):
        assert s in out
