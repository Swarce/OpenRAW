"""Live training status from Kaggle, published to a Hugging Face repo.

Kaggle shows a committed notebook's log only in the browser, and only after a
lag. With STATUS_REPO set in the notebook, the runner uploads a small status
snapshot to that (private) Hugging Face dataset repo while training runs: at
every [EVAL] line, and every 15 minutes in between. Anyone with access to the
repo -- including tools that read Hugging Face -- can follow the run without
opening Kaggle.

Each snapshot replaces these files under <task>/ in the repo:
  status.json   state, last evaluation, session time, last update (UTC)
  eval.csv      train.py's evaluation history (epoch, step, raw/rgb PSNR)
  log_tail.txt  the last 200 lines of training output
  STATUS.txt    the runner's DONE / RUN AGAIN line (at the end of a session)
  compare.md    the end-of-session model comparison, when there is one

Publishing never interrupts training: any failure (no token, no network, no
huggingface_hub) is reported once in the log and publishing stops.
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import threading
import time
from pathlib import Path

HEARTBEAT_S = 15 * 60
TAIL_LINES = 200


def hf_token() -> str | None:
    """HF_TOKEN from the environment, else from Kaggle Secrets (Add-ons -> Secrets -> HF_TOKEN)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001 -- not on Kaggle, or no such secret
        return None


class StatusPublisher:
    """Collects training output and uploads snapshots from a background thread.

    api: a huggingface_hub.HfApi-like object (create_repo, create_commit);
    None = make one from `token`. Tests pass a fake."""

    def __init__(self, repo_id: str, task: str, run_dir: Path, token: str | None = None, api=None,
                 heartbeat_s: float = HEARTBEAT_S, log=print, session_start: float | None = None):
        self.repo_id, self.task, self.run_dir = repo_id, task, Path(run_dir)
        self.heartbeat_s, self.log = heartbeat_s, log
        self.session_start = session_start or time.time()
        self.lines: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        self.last_eval = ""
        self.state = "starting"
        self.enabled = True
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self.uploads = 0
        try:
            if api is None:
                from huggingface_hub import HfApi
                api = HfApi(token=token)
            api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
            self.api = api
        except Exception as e:  # noqa: BLE001
            self._disable(f"can't use Hugging Face repo {repo_id}: {type(e).__name__}: {e}")

    def _disable(self, why: str) -> None:
        if self.enabled:
            self.log(f"[status] live status off -- {why}")
        self.enabled = False

    # ---- feeding
    def feed(self, line: str) -> None:
        with self._lock:
            self.lines.append(line.rstrip("\n"))
            if "[EVAL]" in line:
                self.last_eval = line.strip()
        if "[EVAL]" in line:
            self._wake.set()  # publish right after each evaluation

    # ---- publishing
    def snapshot(self) -> dict[str, bytes]:
        with self._lock:
            tail = "\n".join(self.lines) + "\n"
            last_eval = self.last_eval
        status = {
            "task": self.task,
            "state": self.state,
            "updated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "session_hours": round((time.time() - self.session_start) / 3600, 2),
            "last_eval": last_eval,
        }
        files = {"status.json": (json.dumps(status, indent=1) + "\n").encode(), "log_tail.txt": tail.encode()}
        for name, src in (("eval.csv", self.run_dir / "eval.csv"),
                          ("STATUS.txt", self.run_dir.parent.parent / "STATUS.txt"),
                          ("compare.md", self.run_dir.parent.parent / "compare" / "compare.md")):
            try:
                if src.is_file():
                    files[name] = src.read_bytes()
            except OSError:
                pass
        return files

    def publish(self) -> bool:
        if not self.enabled:
            return False
        try:
            from huggingface_hub import CommitOperationAdd
        except Exception:  # noqa: BLE001 -- tests' fake api takes plain tuples
            CommitOperationAdd = None
        files = self.snapshot()
        ops = [CommitOperationAdd(path_in_repo=f"{self.task}/{n}", path_or_fileobj=b) if CommitOperationAdd
               else (f"{self.task}/{n}", b) for n, b in files.items()]
        try:
            self.api.create_commit(repo_id=self.repo_id, repo_type="dataset", operations=ops,
                                   commit_message=f"{self.task}: {self.state}")
            self.uploads += 1
            return True
        except Exception as e:  # noqa: BLE001
            self._disable(f"upload failed: {type(e).__name__}: {e}")
            return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.heartbeat_s)
            self._wake.clear()
            if self._stop.is_set() or not self.enabled:
                break
            self.publish()

    def start(self) -> "StatusPublisher":
        if self.enabled:
            self.state = "training"
            self.log(f"[status] live status -> https://huggingface.co/datasets/{self.repo_id} ({self.task}/)")
            self.publish()
            self._thread = threading.Thread(target=self._loop, name="status-publisher", daemon=True)
            self._thread.start()
        return self

    def finish(self, state: str) -> None:
        """Stop the background thread and publish a final snapshot."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(30)
        self.state = state
        self.publish()
