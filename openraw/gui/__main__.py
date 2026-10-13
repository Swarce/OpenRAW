"""openraw-gui / python -m openraw.gui"""

from __future__ import annotations

import argparse
import multiprocessing
import sys


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()  # the conversion worker, in a frozen (installer) build
    ap = argparse.ArgumentParser(prog="openraw-gui", description="OpenRAW desktop app")
    ap.add_argument("inputs", nargs="*", help="JPEG files or folders to put in the queue")
    ap.add_argument("--theme", help="start with this theme (e.g. Dark, 'High contrast dark')")
    ap.add_argument("--screenshot", metavar="PNG", help=argparse.SUPPRESS)  # render once and exit (docs, tests)
    ap.add_argument("--selftest", metavar="DIR", help=argparse.SUPPRESS)  # convert a sample, exit 0 if it worked
    a = ap.parse_args(argv)
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The OpenRAW app needs PySide6: pip install 'openraw[gui]'", file=sys.stderr)
        return 1
    from . import style
    from .window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("OpenRAW")
    app.setOrganizationName("OpenRAW")
    style.load_fonts()
    w = MainWindow()
    if a.theme:
        w.set_theme(a.theme)
    if a.inputs:
        w.add_paths(a.inputs)
    if a.selftest:
        return _selftest(app, w, a.selftest)
    w.show()
    if a.screenshot:
        app.processEvents()
        w.grab().save(a.screenshot)
        w.runner.shutdown()
        return 0
    return app.exec()


def _selftest(app, w, out_dir: str) -> int:
    """Smoke test for installers: convert the bundled sample with every
    available method, in the worker process, and check the DNGs."""
    import shutil
    import time
    from pathlib import Path
    from .worker import find_pretrained

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "selftest.log", "w", encoding="utf-8")  # a windowed build has no console

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    sample = find_pretrained("../examples/test_source.jpg")
    if sample is None:
        log("selftest: sample photo not found")
        return 1
    w.dest_mode.set_index(0)
    w.overwrite.setChecked(True)
    ok = True
    import os
    required = [k for k in os.environ.get("OPENRAW_SELFTEST_REQUIRE", "").split(",") if k]
    tested = set()
    for i in range(w.method.count()):
        key = w.method.itemData(i)
        if key == "custom" or not w.method.model().item(i).isEnabled():
            log(f"selftest: {key}: not available, skipped")
            continue
        src = out / f"{key}.jpg"
        shutil.copy(sample, src)
        w.jobs.clear()
        w.queue.setRowCount(0)
        w.add_paths([str(src)])
        w.method.setCurrentIndex(i)
        w.start()
        t = time.time()
        while w.running and time.time() - t < 600:
            app.processEvents()
            time.sleep(0.05)
        job = w.jobs[0]
        good = job["status"] == "Done" and Path(job["dst"]).stat().st_size > 10_000
        ok &= good
        tested.add(key)
        log(f"selftest: {key}: {job['status']} {job['detail']} ({time.time() - t:.1f} s)")
    w.runner.shutdown()
    missing = [k for k in required if k not in tested]
    if missing:
        ok = False
        log(f"selftest: required but not available: {', '.join(missing)}")
    log("selftest: " + ("OK" if ok else "FAILED"))
    log_file.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
