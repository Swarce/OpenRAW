"""Tests for the batch/folder CLI (openraw/cli.py)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from openraw.cli import main, plan_jobs


def _jpeg(path: Path, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray((rng.random((40, 56, 3)) * 255).astype(np.uint8)).save(path, quality=85)
    return path


# ---- plan_jobs: pure path logic, no conversion ----

def test_single_file_defaults_to_sibling_dng(tmp_path):
    src = _jpeg(tmp_path / "a.jpg")
    assert plan_jobs([str(src)], None, False) == [(src, tmp_path / "a.dng")]


def test_folder_non_recursive_skips_subfolders_and_non_jpegs(tmp_path):
    _jpeg(tmp_path / "in" / "a.jpg")
    _jpeg(tmp_path / "in" / "B.JPEG")  # case-insensitive extension
    _jpeg(tmp_path / "in" / "sub" / "c.jpg")
    (tmp_path / "in" / "notes.txt").write_text("x")
    names = sorted(s.name for s, _ in plan_jobs([str(tmp_path / "in")], None, False))
    assert names == ["B.JPEG", "a.jpg"]


def test_recursive_mirrors_tree_into_output(tmp_path):
    _jpeg(tmp_path / "in" / "a.jpg")
    _jpeg(tmp_path / "in" / "sub" / "c.jpg")
    out = tmp_path / "out"
    dests = sorted(d for _, d in plan_jobs([str(tmp_path / "in")], str(out), True))
    assert dests == [out / "a.dng", out / "sub" / "c.dng"]


def test_dng_output_with_multiple_inputs_is_an_error(tmp_path):
    a, b = _jpeg(tmp_path / "a.jpg"), _jpeg(tmp_path / "b.jpg")
    with pytest.raises(ValueError, match="single .dng"):
        plan_jobs([str(a), str(b)], str(tmp_path / "x.dng"), False)


def test_two_inputs_mapping_to_same_output_is_an_error_not_an_overwrite(tmp_path):
    _jpeg(tmp_path / "in" / "a.jpg")
    _jpeg(tmp_path / "in" / "a.jpeg")  # both would become a.dng
    with pytest.raises(ValueError, match="both write"):
        plan_jobs([str(tmp_path / "in")], None, False)


def test_missing_path_and_non_jpeg_are_errors(tmp_path):
    with pytest.raises(ValueError, match="no such"):
        plan_jobs([str(tmp_path / "nope.jpg")], None, False)
    (tmp_path / "x.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="not a JPEG"):
        plan_jobs([str(tmp_path / "x.png")], None, False)


def test_duplicate_inputs_are_converted_once(tmp_path):
    src = _jpeg(tmp_path / "a.jpg")
    assert len(plan_jobs([str(src), str(src), str(tmp_path)], None, False)) == 1


# ---- main(): end to end ----

def test_legacy_two_positional_form_still_works(tmp_path):
    src = _jpeg(tmp_path / "a.jpg")
    out = tmp_path / "custom_name.dng"
    assert main([str(src), str(out), "-q"]) == 0
    assert out.exists()


def test_batch_survives_a_bad_file_and_reports_failure(tmp_path):
    _jpeg(tmp_path / "in" / "good1.jpg", 1)
    _jpeg(tmp_path / "in" / "good2.jpg", 2)
    (tmp_path / "in" / "broken.jpg").write_bytes(b"definitely not a jpeg")
    out = tmp_path / "out"
    assert main([str(tmp_path / "in"), "-o", str(out), "-q"]) == 1  # nonzero: something failed
    assert sorted(p.name for p in out.glob("*.dng")) == ["good1.dng", "good2.dng"]
    assert not list(out.glob("*.partial"))  # failed file left nothing behind


def test_rerun_skips_existing_unless_overwrite(tmp_path):
    src = _jpeg(tmp_path / "a.jpg")
    dst = tmp_path / "a.dng"
    assert main([str(src), "-q"]) == 0
    dst.write_bytes(b"sentinel")  # if it were reconverted, this would be replaced
    assert main([str(src), "-q"]) == 0
    assert dst.read_bytes() == b"sentinel"
    assert main([str(src), "-q", "--overwrite"]) == 0
    assert dst.read_bytes() != b"sentinel"


def test_parallel_jobs_produce_same_files_as_serial(tmp_path):
    for i in range(3):
        _jpeg(tmp_path / "in" / f"p{i}.jpg", i)
    assert main([str(tmp_path / "in"), "-o", str(tmp_path / "serial"), "-q"]) == 0
    assert main([str(tmp_path / "in"), "-o", str(tmp_path / "par"), "-q", "--jobs", "2"]) == 0
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for f in sorted((tmp_path / "serial").glob("*.dng")):
        assert digest(f) == digest(tmp_path / "par" / f.name)


def test_threaded_tile_encoding_is_byte_identical(tmp_path):
    """Threads must only change speed, never output: tile order is preserved."""
    from openraw.dng_writer import write_linear_dng
    x = (np.random.default_rng(0).random((600, 700, 3)) * 65535).astype(np.uint16)
    digests = set()
    for t in (1, 3, 8):
        p = tmp_path / f"t{t}.dng"
        write_linear_dng(str(p), x, threads=t)
        digests.add(hashlib.sha256(p.read_bytes()).hexdigest())
    assert len(digests) == 1


def test_version_reports_package_and_libraries(capsys):
    """Bug reports need library versions -- nearly every bug so far was
    version-dependent. Also handled before argparse, so it needs no inputs."""
    from openraw import __version__
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert f"OpenRAW {__version__}" in out
    for lib in ("numpy", "tifffile", "imagecodecs"):
        assert lib in out
