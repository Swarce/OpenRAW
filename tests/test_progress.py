"""OpenRawPipeline's progress callback (used by the desktop app)."""

import os

from openraw.cli import _convert_one
from openraw.pipeline import OpenRawPipeline, PipelineConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "examples", "test_source.jpg")


def test_progress_reports_stages_in_order_up_to_done(tmp_path):
    seen = []
    OpenRawPipeline(progress=lambda stage, f: seen.append((stage, f))).run_to_dng(SRC, str(tmp_path / "a.dng"))
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and seen[-1] == ("Done", 1.0)
    assert {"Reading JPEG", "Writing DNG"} <= {s for s, _ in seen}


def test_convert_one_passes_progress_and_works_without(tmp_path):
    seen = []
    r = _convert_one(SRC, str(tmp_path / "a.dng"), PipelineConfig(), lambda s, f: seen.append(f))
    assert r["ok"] and seen[-1] == 1.0
    assert _convert_one(SRC, str(tmp_path / "b.dng"), PipelineConfig())["ok"]
