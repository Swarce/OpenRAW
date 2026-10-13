"""The desktop app (openraw.gui), offscreen: building the window in every
theme, the queue, settings, and real conversions through the worker process."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from openraw.gui import style  # noqa: E402
from openraw.gui.window import DONE, SKIPPED, STOPPED, MainWindow  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "examples" / "test_source.jpg"


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    style.load_fonts()
    return a


@pytest.fixture
def win(app, tmp_path):
    s = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    w = MainWindow(settings=s)
    w.method.setCurrentIndex(w.method.findData("classical"))
    yield w
    w.runner.shutdown()
    w.close()
    w.deleteLater()


def _photos(tmp_path, n=2, sub=True) -> Path:
    d = tmp_path / "in"
    (d / "sub").mkdir(parents=True)
    for i in range(n):
        shutil.copy(SRC, d / f"p{i}.jpg")
    if sub:
        shutil.copy(SRC, d / "sub" / "s.JPG")
    (d / "notes.txt").write_text("not a photo")
    return d


def _run(app, w, timeout=180):
    t = time.time()
    while w.running and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.02)
    assert not w.running, "conversion didn't finish"


def test_every_theme_builds_and_renders(app, win, tmp_path):
    for name in style.THEME_CHOICES:
        win.set_theme(name)
        app.processEvents()
        assert not win.grab().isNull()
    for t in style.THEMES.values():
        css = style.stylesheet(t)
        assert t.accent in css and "{" in css


def test_high_contrast_themes_use_orange(app):
    for name in ("High contrast dark", "High contrast light"):
        r, g, b = (int(style.THEMES[name].accent[i:i + 2], 16) for i in (1, 3, 5))
        assert r > g > b  # orange: red > green > blue


def test_queue_adds_jpegs_from_folders_once(win, tmp_path):
    d = _photos(tmp_path)
    assert win.add_paths([str(d)]) == 3            # p0, p1, sub/s.JPG; notes.txt ignored
    assert win.add_paths([str(d / "p0.jpg")]) == 0  # already queued
    assert win.queue.rowCount() == 3
    win.queue.selectRow(0)
    win.remove_selected()
    assert len(win.jobs) == win.queue.rowCount() == 2


def test_destination_mirrors_folders(win, tmp_path):
    d = _photos(tmp_path)
    win.add_paths([str(d)])
    out = tmp_path / "out"
    win.dest_mode.set_index(1)
    win.dest_path.setText(str(out))
    dests = sorted(str(win.destination(j).relative_to(out)) for j in win.jobs)
    assert dests == ["p0.dng", "p1.dng", os.path.join("sub", "s.dng")]
    win.dest_mode.set_index(0)
    assert all(win.destination(j) == j["src"].with_suffix(".dng") for j in win.jobs)


def test_controls_map_to_the_pipeline_config(win):
    win.layout_sel.set_index(1)
    win.aa_slider.setValue(50)
    win.bit_depth.setCurrentIndex(win.bit_depth.findData(14))
    win.keep_exif.setChecked(False)
    c = win.config()
    assert (c.dng_layout, c.dng_cfa_olpf, c.dng_bit_depth, c.preserve_exif, c.use_invisp) == ("cfa", 0.5, 14, False, False)
    win.layout_sel.set_index(0)
    assert win.config().dng_cfa_olpf == 0.0  # anti-aliasing only applies to the mosaic
    assert not win.aa_slider.isEnabled() or not win.aa_widget.isEnabled()


def test_settings_are_remembered(app, tmp_path):
    s = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    w = MainWindow(settings=s)
    w.method.setCurrentIndex(w.method.findData("classical"))
    w.layout_sel.set_index(1)
    w.aa_slider.setValue(35)
    w.set_theme("Darkroom")
    w.save_settings()
    w.runner.shutdown()
    w2 = MainWindow(settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    assert (w2.method.currentData(), w2.layout_sel.index(), w2.aa_slider.value(), w2.theme.name) == \
        ("classical", 1, 35, "Darkroom")
    w2.runner.shutdown()


def test_converts_skips_existing_and_previews(app, win, tmp_path):
    d = _photos(tmp_path)
    win.add_paths([str(d)])
    win.layout_sel.set_index(1)
    win.start()
    _run(app, win)
    assert [j["status"] for j in win.jobs] == [DONE] * 3, [j["detail"] for j in win.jobs]
    for j in win.jobs:
        assert Path(j["dst"]).is_file() and not Path(str(j["dst"]) + ".partial").exists()
    assert "3 done" in win.status_left.text()

    # the comparison renders (needs rawpy)
    pytest.importorskip("rawpy")
    win.queue.selectRow(0)
    t = time.time()
    while win.compare.left is None and time.time() - t < 60:
        app.processEvents()
        time.sleep(0.02)
    assert win.compare.left is not None and win.compare.right is not None

    # converting again skips what's there, unless asked to replace it
    for j in win.jobs:
        j["status"] = "Waiting"
    win.start()
    _run(app, win)
    assert [j["status"] for j in win.jobs] == [SKIPPED] * 3


def test_stop_leaves_no_partial_file(app, win, tmp_path):
    import numpy as np
    from PIL import Image
    big = tmp_path / "big"
    big.mkdir()
    rng = np.random.default_rng(0)
    img = (rng.random((3000, 4000, 3)) * 255).astype(np.uint8)  # a few seconds through the pipeline
    for i in range(2):
        Image.fromarray(img).save(big / f"b{i}.jpg", quality=90)
    win.add_paths([str(big)])
    win.start()
    t = time.time()
    while not any(j["status"] == "Converting" for j in win.jobs) and time.time() - t < 60:
        app.processEvents()
        time.sleep(0.02)
    win.stop()
    assert not win.running
    assert STOPPED in [j["status"] for j in win.jobs]
    assert not list(big.glob("*.partial"))
    assert not list(big.glob("*.dng.partial"))
