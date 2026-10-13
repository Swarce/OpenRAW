"""Conversions for the desktop app, in a separate process.

One long-lived worker process takes jobs from a queue and reports back on
another: progress per stage, then a result. A separate process keeps the
window responsive (numpy / torch hold the interpreter for seconds at a time)
and makes Stop immediate and safe: the process is terminated and the
half-written ``.partial`` file removed, so no truncated DNG is ever left.

Each job is converted exactly like the command line does it
(cli._convert_one: write to .partial, then rename), so both behave alike.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import time
from dataclasses import asdict
from pathlib import Path


def _worker_main(jobs, events) -> None:  # runs in the worker process
    from ..cli import _convert_one
    from ..pipeline import PipelineConfig

    while True:
        job = jobs.get()
        if job is None:
            return
        jid = job["id"]
        last = [0.0, -1.0]  # time, fraction of the last report

        def progress(stage, fraction, jid=jid, last=last):
            now = time.time()
            if fraction >= 1.0 or fraction - last[1] >= 0.01 or now - last[0] > 0.5:
                last[0], last[1] = now, fraction
                events.put(("progress", jid, stage, fraction))

        events.put(("started", jid))
        result = _convert_one(job["src"], job["dst"], PipelineConfig(**job["config"]), progress)
        events.put(("finished", jid, result))


class Runner:
    """Owns the worker process. Not a QObject: the window polls events()."""

    def __init__(self):
        self.ctx = mp.get_context("spawn")  # same behaviour on every OS; safe with Qt
        self.proc = None
        self.jobs = None
        self.events = None
        self.current: dict | None = None  # job being converted

    def _ensure(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            return
        self.jobs, self.events = self.ctx.Queue(), self.ctx.Queue()
        self.proc = self.ctx.Process(target=_worker_main, args=(self.jobs, self.events), daemon=True,
                                     name="openraw-worker")
        self.proc.start()

    def submit(self, jid: int, src: str, dst: str, config) -> None:
        self._ensure()
        self.jobs.put({"id": jid, "src": str(src), "dst": str(dst), "config": asdict(config)})

    def poll(self) -> list[tuple]:
        """Events since the last call (never blocks)."""
        out = []
        if self.events is None:
            return out
        while True:
            try:
                ev = self.events.get_nowait()
            except (_queue.Empty, OSError, ValueError):
                break
            out.append(ev)
        if self.proc is not None and not self.proc.is_alive() and self.proc.exitcode not in (0, None):
            out.append(("crashed", self.proc.exitcode))
            self.proc = None
        return out

    def stop(self, partial_files=()) -> None:
        """Kill the worker now; remove the given jobs' half-written outputs."""
        if self.proc is not None:
            self.proc.terminate()
            self.proc.join(5)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(5)
        self.proc = None
        for dst in partial_files:
            try:
                os.remove(str(dst) + ".partial")
            except OSError:
                pass

    def shutdown(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            try:
                self.jobs.put(None)
                self.proc.join(2)
            except Exception:  # noqa: BLE001
                pass
        self.stop()


def find_pretrained(filename: str) -> Path | None:
    """A checkpoint shipped with OpenRAW: next to a frozen app, in a source
    checkout, in the working folder, or in $OPENRAW_PRETRAINED."""
    import sys
    here = Path(__file__).resolve()
    candidates = []
    if os.environ.get("OPENRAW_PRETRAINED"):
        candidates.append(Path(os.environ["OPENRAW_PRETRAINED"]))
    if getattr(sys, "_MEIPASS", None):
        candidates.append(Path(sys._MEIPASS) / "pretrained")
    candidates += [here.parents[2] / "pretrained", here.parents[1] / "pretrained", Path.cwd() / "pretrained"]
    for d in candidates:
        if (d / filename).is_file():
            return d / filename
    return None
