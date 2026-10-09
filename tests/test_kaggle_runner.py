"""kaggle/kaggle_runner.py, on a simulated Kaggle machine: fake /kaggle/input
and /kaggle/working folders, synthetic Bayer DNGs served from a local HTTP
server behind fake FiveK metadata."""

from __future__ import annotations

import functools
import http.server
import json
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("rawpy")
pytest.importorskip("colour_demosaicing")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle"))
sys.path.insert(0, str(ROOT / "data"))
import kaggle_runner as kr  # noqa: E402

from .cfa_dng import make_cfa_dng  # noqa: E402

CAMS = {"Testco T1": {"t1a": "train", "t1b": "train", "t1c": "test"},
        "Otherco Z9": {"z9a": "train", "z9b": "test"}}


@pytest.fixture
def kaggle(tmp_path, monkeypatch):
    srv = tmp_path / "server"; srv.mkdir()
    for cam, names in CAMS.items():
        for n in names:
            make_cfa_dng(str(srv / f"{n}.dng"), "GRBG", h=320, w=432)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(
        type("Q", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None}), directory=str(srv)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    k = tmp_path / "kaggle"
    for d in ("input", "working", "scratch"):
        (k / d).mkdir(parents=True)
    monkeypatch.setattr(kr, "INPUT", k / "input")
    monkeypatch.setattr(kr, "WORKING", k / "working")
    monkeypatch.setattr(kr, "SCRATCH", k / "scratch")
    meta = k / "working" / kr.DATA_DIR / "fivek" / "_metadata"; meta.mkdir(parents=True)
    for sp, f in {"train": "training.json", "val": "validation.json", "test": "testing.json"}.items():
        (meta / f).write_text(json.dumps({n: {"urls": {"dng": f"{base}/{n}.dng", "tiff16": {}},
                                              "camera": {"make": c.split()[0], "model": c.split()[1]}}
                                          for c, ns in CAMS.items() for n, s in ns.items() if s == sp}))
    yield k
    httpd.shutdown()


def _publish_as_dataset(k, name="openraw-data-ds"):
    """What 'New Dataset' + 'Add Input' does: the prepared output appears read-only under /kaggle/input."""
    import shutil
    shutil.copytree(k / "working" / kr.DATA_DIR, k / "input" / name / kr.DATA_DIR)
    shutil.rmtree(k / "working" / kr.DATA_DIR)


def test_prepare_mode_builds_compact_data_and_deletes_dngs(kaggle):
    assert kr.run(cameras="all", budget_gb=5) == "prepared"
    out = kaggle / "working" / kr.DATA_DIR
    assert sorted(p.name for p in out.glob("*_train.txt")) == ["Otherco_Z9_train.txt", "Testco_T1_train.txt"]
    assert len(list(out.glob("*/RAW/*.npz"))) == 5
    assert not (out / "fivek" / "raw").exists()  # DNGs deleted
    assert (out / "fivek" / "_metadata" / "training.json").exists()  # carried along for train mode


def test_prepare_respects_the_data_budget(kaggle, monkeypatch):
    monkeypatch.setattr(kr, "EST_PAIR_MB", 1000.0)  # pretend pairs are 1 GB: nothing fits in 2 GB
    kr.prepare("all", budget_gb=2)
    assert not list((kaggle / "working" / kr.DATA_DIR).glob("*/RAW/*.npz"))


def test_train_mode_merges_datasets_and_resumes_from_previous_output(kaggle, monkeypatch):
    kr.prepare("all", budget_gb=5)
    _publish_as_dataset(kaggle)
    # previous session's output attached as input: a checkpoint + eval.csv
    prev = kaggle / "input" / "openraw-prev-output" / "exps" / "openraw-fivek"
    (prev / "checkpoint").mkdir(parents=True)
    (prev / "checkpoint" / "latest_state.pth").write_bytes(b"state")
    (prev / "checkpoint" / "best.pth").write_bytes(b"best")
    (prev / "eval.csv").write_text("epoch,step,raw_psnr,rgb_psnr\n0,3,30.0,25.0\n")
    calls = {}
    monkeypatch.setattr(kr, "train", lambda task, data_root, **kw: calls.update(task=task, data_root=data_root, **kw) or 0)
    monkeypatch.setattr(kr, "write_status", lambda task, epochs: calls.update(status_epochs=epochs))
    assert kr.run(task="openraw-fivek") == "trained"
    data_root = calls["data_root"]
    assert sorted(p.name for p in data_root.iterdir() if p.is_symlink()) == ["Otherco_Z9", "Testco_T1"]
    ck = kaggle / "working" / "exps" / "openraw-fivek"
    assert (ck / "checkpoint" / "latest_state.pth").read_bytes() == b"state"  # restored for --resume
    assert (ck / "eval.csv").exists()
    assert calls["epochs"] == kr.suggest_epochs(3) == 65000  # 3 train pairs -> ~195k steps


def test_train_builds_the_right_command(kaggle, monkeypatch):
    kr.prepare("all", budget_gb=5)
    data = kaggle / "working" / kr.DATA_DIR
    seen = {}
    monkeypatch.setattr(kr.subprocess, "call", lambda cmd, cwd, env: seen.update(cmd=cmd, cwd=cwd, env=env) or 0)
    monkeypatch.setattr(kr, "_gpu_count", lambda: 1)
    kr.train("t", data, epochs=7, session_start=__import__("time").time() - 3600)  # 1 h already used
    cmd = seen["cmd"]
    assert cmd[1] == "train.py" and "--all-downloaded" in cmd and "--resume" not in cmd
    assert seen["env"]["PYTHONUNBUFFERED"] == "1"  # logs stream live
    assert cmd[cmd.index("--epochs") + 1] == "7"
    budget = float(cmd[cmd.index("--time_limit_hours") + 1])
    assert abs(budget - (kr.SESSION_HOURS - kr.SAVE_MARGIN_HOURS - 1.0)) < 0.01
    (kaggle / "working" / "exps" / "t" / "checkpoint").mkdir(parents=True)
    (kaggle / "working" / "exps" / "t" / "checkpoint" / "latest_state.pth").write_bytes(b"x")
    kr.train("t", data, epochs=7)
    assert "--resume" in seen["cmd"]


def test_find_checkpoint_prefers_full_state(kaggle):
    a = kaggle / "input" / "a" / "exps" / "t" / "checkpoint"; a.mkdir(parents=True)
    (a / "latest.pth").write_bytes(b"w")
    b = kaggle / "input" / "b" / "exps" / "t" / "checkpoint"; b.mkdir(parents=True)
    (b / "latest_state.pth").write_bytes(b"s")
    assert kr.find_checkpoint("t") == b
    assert kr.find_checkpoint("other-task") is None



# ------------------------------------------------------- multi-GPU decisions
def test_choose_gpus_falls_back_step_by_step(monkeypatch):
    """Self-test as-is -> with NCCL_P2P_DISABLE=1 -> one GPU."""
    monkeypatch.setattr(kr, "_gpu_count", lambda: 2)
    for results, want_gpus, want_p2p in (([True], 2, None), ([False, True], 2, "1"), ([False, False], 1, None)):
        calls = []
        monkeypatch.setattr(kr, "_selftest", lambda env, r=iter(results): calls.append(env) or next(r))
        n, env = kr.choose_gpus(0, {"X": "1"})
        assert (n, env.get("NCCL_P2P_DISABLE")) == (want_gpus, want_p2p)
        assert len(calls) == len(results)
    monkeypatch.setattr(kr, "_gpu_count", lambda: 1)
    monkeypatch.setattr(kr, "_selftest", lambda env: pytest.fail("no self-test with one GPU"))
    assert kr.choose_gpus(0, {})[0] == 1


def test_selftest_that_hangs_counts_as_failure(monkeypatch):
    def hang(*a, **k):
        raise kr.subprocess.TimeoutExpired(cmd="selftest", timeout=k.get("timeout"))
    monkeypatch.setattr(kr.subprocess, "run", hang)
    assert kr._selftest({}, timeout=1) is False


def test_failed_multi_gpu_run_is_retried_on_one_gpu(kaggle, monkeypatch):
    kr.prepare("all", budget_gb=5)
    data = kaggle / "working" / kr.DATA_DIR
    monkeypatch.setattr(kr, "choose_gpus", lambda gpus, env: (2, env))
    cmds = []
    monkeypatch.setattr(kr.subprocess, "call", lambda cmd, cwd, env: cmds.append(cmd) or (1 if len(cmds) == 1 else 0))
    assert kr.train("t", data, epochs=3) == 0
    assert [c[c.index("--gpus") + 1] for c in cmds] == ["2", "1"]


def test_multigpu_selftest_tool_runs_on_cpu_processes():
    """The real tool, two processes over gloo (NCCL needs GPUs)."""
    pytest.importorskip("torch")
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "multigpu_selftest.py"), "--cpu", "2"],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0 and "SELFTEST OK" in r.stdout, r.stdout + r.stderr


def test_compare_only_scores_models_without_training(kaggle, monkeypatch):
    pytest.importorskip("torch")
    kr.run(cameras="all", budget_gb=5)
    _publish_as_dataset(kaggle)
    called = []
    monkeypatch.setattr(kr, "train", lambda *a, **k: called.append(1) or 0)
    assert kr.run(compare_only=True) == "compared"
    assert not called
    md = (kaggle / "working" / "compare" / "compare.md").read_text()
    assert "nikon.pth" in md and "canon.pth" in md and "openraw-fivek-e35-best.pth" in md


def test_new_task_starts_from_another_runs_best_weights(kaggle, monkeypatch):
    import torch
    kr.run(cameras="all", budget_gb=5)
    _publish_as_dataset(kaggle)
    old = kaggle / "input" / "prev-output" / "exps" / "old-run" / "checkpoint"; old.mkdir(parents=True)
    torch.save({"w": torch.ones(1)}, old / "best.pth")
    torch.save({"w": torch.zeros(1)}, old / "latest.pth")
    seen = {}
    def fake_train(task, data_root, **kw):
        ck = kaggle / "working" / "exps" / task / "checkpoint"
        seen["files"] = sorted(p.name for p in ck.iterdir())
        seen["w"] = torch.load(ck / "latest.pth")["w"].item()
        return 0
    monkeypatch.setattr(kr, "train", fake_train)
    monkeypatch.setattr(kr, "compare", lambda *a, **k: 0)
    monkeypatch.setattr(kr, "write_status", lambda *a, **k: "")
    kr.run(task="new-run", init_from="old-run")
    assert seen == {"files": ["latest.pth"], "w": 1.0}  # best.pth copied as a weights-only start, no state
    with pytest.raises(SystemExit, match="no checkpoint found"):
        kr.run(task="other-run", init_from="missing-run")
