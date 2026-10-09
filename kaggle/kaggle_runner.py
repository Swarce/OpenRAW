"""
OpenRAW on Kaggle: unattended InvISP training across free GPU sessions.

One notebook (kaggle/openraw_kaggle.ipynb) calls run(); what happens depends
on what's attached to it:

  PREPARE mode -- no prepared data attached. Downloads the chosen FiveK
    cameras, preprocesses them into the compact training format, deletes each
    DNG right after, and writes everything to /kaggle/working/openraw-data/,
    staying under the 20 GB Kaggle keeps between sessions. Run it on CPU (no
    GPU quota spent), then turn the output into a Kaggle Dataset (one click).

  TRAIN mode -- one or more prepared datasets attached. Merges them, resumes
    from the previous session's checkpoint if the notebook's own last output is
    attached, trains until shortly before the 12 h session limit, stops cleanly
    between epochs, and leaves checkpoints + eval.csv + best.pth in
    /kaggle/working for the next session.

Kaggle limits this is built around: 12 h per GPU session, ~30 GPU h/week,
20 GB saved in /kaggle/working (scratch space elsewhere is wiped).

Paths come from environment variables so the logic is testable off Kaggle.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORKING = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
SCRATCH = Path(os.environ.get("KAGGLE_SCRATCH", "/tmp/openraw"))  # wiped between sessions; fine for symlinks

SESSION_HOURS = 12.0      # Kaggle GPU session limit
SAVE_MARGIN_HOURS = 0.75  # leave time for Kaggle to save /kaggle/working
DATA_DIR = "openraw-data"
UPSTREAM_STEPS = 195_000  # InvISP's training budget: 650 images x 300 epochs
EST_PAIR_MB = 15.0        # compact pair (mosaic + JPEG target) before measuring


def log(*a):
    print("[kaggle]", *a, flush=True)


# --------------------------------------------------------------------- data
_SKIP_DIRS = {"RAW", "RGB", "exps", "checkpoint", "raw", "_metadata", "__pycache__"}


def find_data_roots(root: Path = None, max_depth: int = 8) -> list[Path]:
    """Prepared-data roots under `root`: folders holding <Camera>_train.txt
    next to a <Camera>/RAW/ folder of .npz pairs. Searched 8 levels deep:
    Kaggle mounts inputs at varying depths (e.g. datasets/<user>/<name>/,
    notebooks/<user>/<name>/), and a dataset made from a notebook's output
    adds levels of its own."""
    root = Path(root or INPUT)
    found = []
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        if len(d.relative_to(root).parts) > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [n for n in dirnames if n not in _SKIP_DIRS]  # never walk thousands of pairs
        cams = [f[: -len("_train.txt")] for f in filenames if f.endswith("_train.txt")]
        if any((d / c / "RAW").is_dir() and any((d / c / "RAW").glob("*.npz")) for c in cams):
            found.append(d)
            dirnames[:] = []  # don't descend into a data root's camera folders
    return sorted(found)


def _unused_data_dirs(root: Path, roots: list[Path], max_depth: int = 10) -> list[Path]:
    """Folders named like prepared data (openraw-data) that weren't used --
    reported, so a dataset that's attached but not found doesn't go unnoticed."""
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        d = Path(dirpath)
        if len(d.relative_to(root).parts) > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [n for n in dirnames if n not in _SKIP_DIRS]
        if d.name == DATA_DIR and d not in roots and not any(r in d.parents for r in roots):
            out.append(d)
    return out


def attached_inputs(root: Path = None) -> list[Path]:
    """Each attached input's folder: /kaggle/input/<name>, or in Kaggle's newer
    layout /kaggle/input/{datasets,notebooks,...}/<owner>/<name>."""
    root = Path(root or INPUT)
    out = []
    for top in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        if top.name in ("datasets", "notebooks", "models", "competitions"):
            out += sorted(q for owner in top.iterdir() if owner.is_dir() for q in owner.iterdir() if q.is_dir())
        else:
            out.append(top)
    return out


def trained_cameras(ckpt_dir: Path) -> list[str]:
    """The cameras a run was trained on, from its commandline_args.yaml (next
    to its checkpoint folder); [] if unknown."""
    import json
    try:
        return list(json.loads((Path(ckpt_dir).parent / "commandline_args.yaml").read_text()).get("camera") or [])
    except (OSError, ValueError):
        return []


def merge_data(roots: list[Path], dest: Path) -> list[str]:
    """Combine several prepared datasets (read-only Kaggle inputs) into one
    writable data root of symlinks + copied split lists. Returns camera names."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    cams = []
    for r in roots:
        meta = r / "fivek" / "_metadata"
        if meta.is_dir() and not (dest / "fivek" / "_metadata").exists():
            shutil.copytree(meta, dest / "fivek" / "_metadata")
        for lst in sorted(r.glob("*_train.txt")):
            cam = lst.name[: -len("_train.txt")]
            if not (r / cam / "RAW").is_dir():
                continue
            if (dest / cam).exists():
                log(f"camera {cam} appears in more than one dataset -- using the first ({dest / cam})")
                continue
            (dest / cam).symlink_to(r / cam, target_is_directory=True)
            for split in ("train", "test"):
                src = r / f"{cam}_{split}.txt"
                if src.exists():
                    shutil.copy(src, dest / f"{cam}_{split}.txt")
            cams.append(cam)
    return cams


def count_train_pairs(data_root: Path) -> int:
    n = 0
    for lst in Path(data_root).glob("*_train.txt"):
        cam = lst.name[: -len("_train.txt")]
        names = [x.strip() for x in lst.read_text().split() if x.strip()]
        n += sum((Path(data_root) / cam / "RAW" / f"{x}.npz").exists() for x in names)
    return n


def suggest_epochs(n_train: int) -> int:
    """Epochs matching upstream InvISP's ~195k-step budget for this dataset."""
    return max(1, round(UPSTREAM_STEPS / max(1, n_train)))


# --------------------------------------------------------------- checkpoints
def find_checkpoint(task: str, root: Path = None) -> Path | None:
    """The checkpoint folder of `task` in an attached input (the notebook's own
    previous output), preferring a full-state checkpoint, newest first."""
    root = Path(root or INPUT)
    hits = []
    for name in ("latest_state.pth", "latest.pth"):
        for p in root.rglob(f"exps/{task}/checkpoint/{name}"):
            hits.append((name == "latest_state.pth", p.stat().st_mtime, p.parent))
        if hits:
            break
    return max(hits)[2] if hits else None


def start_from(init_from: str, out_path: Path, task: str, root: Path = None) -> Path | None:
    """Start `task` from another run's weights: copy that run's best.pth (else
    latest.pth) from the attached inputs -- init_from is its TASK name, or a
    path to a .pth -- to <out_path>/<task>/checkpoint/latest.pth. train.py's
    --resume then loads it as a weights-only checkpoint: epoch 0, a fresh
    optimizer, LR schedule, test set and best score. Returns the source file."""
    root = Path(root or INPUT)
    src = Path(init_from) if str(init_from).endswith(".pth") else None
    if src is None:
        for name in ("best.pth", "latest.pth"):
            hits = sorted(root.rglob(f"exps/{init_from}/checkpoint/{name}"), key=lambda p: p.stat().st_mtime)
            if hits:
                src = hits[-1]
                break
    if src is None or not src.is_file():
        return None
    dst = Path(out_path) / task / "checkpoint"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst / "latest.pth")
    return src


def restore_checkpoint(ckpt_dir: Path, out_path: Path, task: str) -> Path:
    """Copy the previous session's checkpoints + eval.csv into the writable
    output folder, where train.py --resume looks for them."""
    dst = Path(out_path) / task / "checkpoint"
    dst.mkdir(parents=True, exist_ok=True)
    for f in Path(ckpt_dir).iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    ev = Path(ckpt_dir).parent / "eval.csv"
    if ev.exists():
        shutil.copy2(ev, dst.parent / "eval.csv")
    return dst


# ------------------------------------------------------------------ prepare
def _du_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9


def prepare(cameras="all", budget_gb: float = 18.0, out: Path = None, jobs: int = 4, workers: int | None = None):
    """Download + preprocess cameras into out (default /kaggle/working/
    openraw-data), camera by camera, never starting one that would push the
    total past budget_gb (Kaggle keeps 20 GB). DNGs are deleted as they're
    processed. Returns the camera folders prepared."""
    sys.path.insert(0, str(REPO / "data"))
    import fivek_download as fk
    import fivek_prepare as fp

    out = Path(out or WORKING / DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)
    items = fk.load_metadata(out / "fivek" / "_metadata", list(fk.SPLIT_FILES))
    index = fk.build_camera_index(items)
    if cameras == "all":
        labels = sorted(index, key=lambda l: -len(index[l]))  # biggest cameras first
    else:
        labels = fk.resolve_cameras(list(cameras), index)

    done_cams, pairs_before = [], 0
    for label in labels:
        used = _du_gb(out)
        n_pairs = sum(1 for _ in out.glob("*/RAW/*.npz"))
        per_pair = (used * 1e3 / n_pairs) if n_pairs else EST_PAIR_MB
        need = len(index[label]) * per_pair / 1e3
        if used + need > budget_gb:
            log(f"skipping {label}: ~{need:.1f} GB would exceed the {budget_gb:.0f} GB budget ({used:.1f} GB used)")
            continue
        log(f"preparing {label} ({len(index[label])} images, ~{need:.1f} GB) ...")
        # Run as its own process via fivek_prepare.py's CLI (which has a proper
        # __main__ guard): its preprocessing pool uses spawn, which re-imports
        # the parent's main module in every worker -- calling it in-process
        # from an unguarded script or pipe crashed the pool in testing.
        cmd = [sys.executable, str(REPO / "data" / "fivek_prepare.py"), "--camera", label, "--download",
               "--delete-dngs", "--data-path", str(out) + "/", "--jobs", str(jobs)]
        if workers:
            cmd += ["--workers", str(workers)]
        if subprocess.call(cmd) != 0:
            log(f"preparing {label} failed -- continuing with the other cameras")
            continue
        done_cams.append(fp.camera_dir_name(label))
    shutil.rmtree(out / "fivek" / "raw", ignore_errors=True)  # emptied by --delete-dngs
    log(f"prepared {len(done_cams)} camera(s), {sum(1 for _ in out.glob('*/RAW/*.npz'))} pairs, "
        f"{_du_gb(out):.1f} GB in {out}")
    return done_cams


def find_raise_csv(root: Path = None) -> Path | None:
    """The RAISE list (CSV with a NEF column) among the attached inputs."""
    root = Path(root or INPUT)
    for p in sorted(root.rglob("*.csv")) if root.is_dir() else []:
        try:
            head = p.open(encoding="utf-8-sig", errors="replace").readline()
        except OSError:
            continue
        if "nef" in [h.strip().strip('"').lower() for h in head.split(",")]:
            return p
    return None


def prepare_raise(csv_path, budget_gb: float = 18.0, out: Path = None, cameras=(), start: int = 0,
                  chunk: int = 60, jobs: int = 4, workers: int | None = None) -> list[str]:
    """Download + preprocess RAISE NEFs from `csv_path` (see data/raise_prepare.py)
    into out, in chunks of `chunk` images, stopping before the output would pass
    budget_gb. NEFs are deleted as they're processed. start: skip that many
    selected images -- to prepare the next part in another session (the log
    says where to continue). Returns the camera folders prepared."""
    sys.path.insert(0, str(REPO / "data"))
    import raise_prepare as rp

    out = Path(out or WORKING / DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)
    rows = rp.select(rp.read_list(csv_path), list(cameras))
    log(f"RAISE list {csv_path}: {len(rows)} images ({rp.summary(rows)}); starting at #{start}")
    pos = start
    while pos < len(rows):
        used = _du_gb(out)
        n_pairs = sum(1 for _ in out.glob("*/RAW/*.npz"))
        per_pair = (used * 1e3 / n_pairs) if n_pairs else EST_PAIR_MB
        n = min(chunk, len(rows) - pos)
        if used + n * per_pair / 1e3 > budget_gb:
            n = int((budget_gb - used) * 1e3 / per_pair)
            if n <= 0:
                break
        cmd = [sys.executable, str(REPO / "data" / "raise_prepare.py"), "--csv", str(csv_path), "--download",
               "--delete-nefs", "--data-path", str(out) + "/", "--jobs", str(jobs),
               "--start", str(pos), "--count", str(n)]
        for c in cameras:
            cmd += ["--camera", c]
        if workers:
            cmd += ["--workers", str(workers)]
        if subprocess.call(cmd) != 0:
            log(f"RAISE images #{pos}-#{pos + n - 1} failed -- continuing")
        pos += n
    shutil.rmtree(out / "raise", ignore_errors=True)  # only the (emptied) NEF download folders
    cams = sorted(p.parent.parent.name for p in out.glob(f"{rp.PREFIX}*/RAW/*.npz"))
    cams = sorted(set(cams))
    log(f"prepared {sum(1 for _ in out.glob(f'{rp.PREFIX}*/RAW/*.npz'))} RAISE pairs ({', '.join(cams)}), "
        f"{_du_gb(out):.1f} GB in {out}")
    if pos < len(rows):
        log(f"budget reached: {len(rows) - pos} images left. To prepare them, run again in a NEW notebook "
            f"(its own output) with RAISE_START = {pos}")
    return cams


# -------------------------------------------------------------------- train
def _gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return 0


def _selftest(env: dict, timeout: float = 240) -> bool:
    """tools/multigpu_selftest.py in a subprocess, killed after `timeout` s: a
    broken interconnect hangs rather than failing."""
    try:
        r = subprocess.run([sys.executable, str(REPO / "tools" / "multigpu_selftest.py")], env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"multi-GPU self-test: no response within {timeout:.0f}s (hung)")
        return False
    ok = r.returncode == 0 and "SELFTEST OK" in r.stdout
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
    log(f"multi-GPU self-test: {'passed' if ok else 'FAILED'} -- {tail[0][:200]}")
    return ok


def choose_gpus(gpus: int, env: dict) -> tuple[int, dict]:
    """Decide how many GPUs to train on, and with which NCCL settings, by
    testing GPU-to-GPU communication first. The first two-GPU session on Kaggle
    produced no output for hours -- consistent with a collective hanging.
    Order: as-is, then NCCL_P2P_DISABLE=1 (the usual fix when peer-to-peer
    over PCIe is broken on cloud VMs), then one GPU."""
    n = _gpu_count() if gpus == 0 else gpus
    if n < 2:
        return 1, env
    if _selftest(env):
        return n, env
    env2 = dict(env, NCCL_P2P_DISABLE="1")
    if _selftest(env2):
        log("multi-GPU works with NCCL_P2P_DISABLE=1 -- training with it")
        return n, env2
    log("multi-GPU communication doesn't work on this machine -- training on 1 GPU")
    return 1, env


def train(task: str, data_root: Path, epochs: int | None = None, time_limit_hours: float | None = None,
          workers: int = 3, extra_args: list[str] | None = None, session_start: float | None = None,
          gpus: int = 0) -> int:
    """Run train.py on all cameras in data_root, resuming if a checkpoint is in
    the output folder, within the remaining session time. gpus: 0 = every GPU
    the session has (both on "GPU T4 x2"), after a communication self-test.
    A multi-GPU run that fails is retried once on one GPU, resuming from the
    last saved epoch. Returns train.py's exit code."""
    out_path = WORKING / "exps"
    n = count_train_pairs(data_root)
    epochs = epochs or suggest_epochs(n)
    # unbuffered: output from train.py and its per-GPU processes appears live
    # in Kaggle's log (buffered output is invisible until a buffer fills or the
    # process exits -- a stalled run then shows nothing at all)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    gpus, env = choose_gpus(gpus, env)

    def run_once(n_gpus: int) -> int:
        limit = time_limit_hours
        if limit is None:
            elapsed = (time.time() - session_start) / 3600 if session_start else 0.0
            limit = max(0.25, SESSION_HOURS - SAVE_MARGIN_HOURS - elapsed)
        resume = (out_path / task / "checkpoint" / "latest_state.pth").exists() or \
                 (out_path / task / "checkpoint" / "latest.pth").exists()
        cmd = [sys.executable, "train.py", "--task", task, "--all-downloaded", "--gamma", "--aug",
               "--data_path", str(data_root) + "/", "--out_path", str(out_path) + "/",
               "--epochs", str(epochs), "--workers", str(workers), "--gpus", str(n_gpus),
               "--time_limit_hours", f"{limit:.3f}"] + (["--resume"] if resume else []) + (extra_args or [])
        log(f"{n} training pairs -> {epochs} epochs (~{n * epochs:,} steps); {n_gpus} GPU(s); "
            f"time budget {limit:.2f} h; {'resuming' if resume else 'fresh start'}")
        log(" ".join(cmd))
        return subprocess.call(cmd, cwd=str(REPO), env=env)

    code = run_once(gpus)
    if code != 0 and gpus > 1:
        log(f"multi-GPU training exited with code {code} -- retrying on 1 GPU from the last saved epoch")
        code = run_once(1)
    return code


def compare(task: str, data_root: Path, out: Path = None, extra: list | None = None) -> int:
    """Score the repo's pretrained/*.pth, this task's best.pth / latest.pth and
    any attached checkpoints on the held-out test images (tools/compare_models.py)
    -> compare/compare.md in the output. Returns its exit code; never raises."""
    out = Path(out or WORKING / "compare")
    ckpts = sorted(str(p) for p in (REPO / "pretrained").glob("*.pth"))
    # this task's checkpoints: from this session's training, else the attached previous output
    own = WORKING / "exps" / task / "checkpoint"
    if not own.is_dir():
        own = find_checkpoint(task, INPUT)
    if own:
        ckpts += [str(Path(own) / n) for n in ("best.pth", "latest.pth") if (Path(own) / n).exists()]
    ckpts += [str(c) for c in (extra or [])]
    cmd = [sys.executable, str(REPO / "tools" / "compare_models.py"), "--data_path", str(data_root) + "/",
           "--out", str(out)] + ckpts
    log(f"comparing {len(ckpts)} checkpoint(s) on the held-out test images ...")
    try:
        return subprocess.call(cmd, cwd=str(REPO), env=dict(os.environ, PYTHONUNBUFFERED="1"))
    except Exception as e:  # noqa: BLE001
        log(f"comparison failed: {e}")
        return 1


def amp_check(data_root: Path) -> int:
    """tools/amp_check.py on this session's GPU and real training crops: is
    mixed precision (train.py --amp) faster here, and at what accuracy cost?"""
    cmd = [sys.executable, str(REPO / "tools" / "amp_check.py"), "--data_path", str(data_root) + "/"]
    log("checking mixed precision on this GPU (a few minutes) ...")
    try:
        return subprocess.call(cmd, cwd=str(REPO), env=dict(os.environ, PYTHONUNBUFFERED="1"))
    except Exception as e:  # noqa: BLE001
        log(f"mixed-precision check failed: {e}")
        return 1


def write_status(task: str, epochs: int) -> str:
    """Leave a one-line STATUS.txt in the output saying whether to run again."""
    import torch
    st_path = WORKING / "exps" / task / "checkpoint" / "latest_state.pth"
    msg = "no checkpoint yet"
    if st_path.exists():
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        done = st["epoch"] + 1
        best = st.get("best_raw_psnr", -1)
        state = "DONE" if done >= epochs else "RUN AGAIN to continue"
        msg = f"{state}: {done}/{epochs} epochs, step {st['step']:,}" + (f", best raw PSNR {best:.2f} dB" if best > 0 else "")
    (WORKING / "STATUS.txt").write_text(msg + "\n")
    log(msg)
    return msg


# ---------------------------------------------------------------------- main
def run(task="openraw", cameras="all", budget_gb=18.0, epochs=None, workers=3, session_start=None,
        extra_train_args=None, gpus=0, raise_csv=None, raise_cameras=(), raise_start=0, compare_only=False,
        init_from=None, amp_check_only=False):
    """raise_csv: prepare RAISE instead of FiveK or training -- a path, or True
    to use the CSV with a NEF column among the attached inputs.
    compare_only: don't train; score the checkpoints on the held-out test
    images (needs the prepared data attached; works on CPU).
    init_from: when `task` has no checkpoint yet, start it from another run's
    weights -- that run's TASK name (its output attached) or a .pth path.
    amp_check_only: don't train; measure whether mixed precision (--amp) is
    faster on this GPU and what it costs in accuracy (tools/amp_check.py)."""
    session_start = session_start or time.time()
    for a in attached_inputs(INPUT):
        log(f"input: {a}")
    if raise_csv:
        csv_path = find_raise_csv(INPUT) if raise_csv is True else Path(raise_csv)
        if not csv_path or not Path(csv_path).exists():
            raise SystemExit("RAISE: no CSV found. Attach the CSV from the RAISE download page as a dataset, "
                             "or give its path as RAISE_CSV.")
        log("RAISE_CSV set -> RAISE PREPARE mode")
        prepare_raise(csv_path, budget_gb=budget_gb, cameras=raise_cameras, start=raise_start)
        log("Next: 'Save Version' finishes, then on this notebook's output page choose 'New Dataset', and "
            "attach that dataset to the training notebook next to the FiveK one.")
        return "prepared RAISE"
    roots = find_data_roots(INPUT)
    if not roots:
        log("no prepared data attached -> PREPARE mode")
        prepare(cameras, budget_gb=budget_gb)
        log("Next: 'Save Version' finishes, then on this notebook's output page choose 'New Dataset', and "
            "attach that dataset to this notebook (Add Input). The next run trains.")
        return "prepared"
    log(f"prepared data found: {', '.join(str(r) for r in roots)} -> TRAIN mode")
    data_root = SCRATCH / DATA_DIR
    shutil.rmtree(data_root, ignore_errors=True)
    cams = merge_data(roots, data_root)
    log(f"cameras: {', '.join(cams)}")
    for d in _unused_data_dirs(INPUT, roots):
        log(f"WARNING: {d} looks like prepared data but holds no usable camera folders -- not used")
    if compare_only:
        code = compare(task, data_root)
        return "compared" if code == 0 else f"comparison exited with code {code}"
    if amp_check_only:
        code = amp_check(data_root)
        return "amp checked" if code == 0 else f"amp check exited with code {code}"
    ck = find_checkpoint(task, INPUT)
    if ck:
        missing = [c for c in trained_cameras(ck) if c not in cams]
        if missing and not compare_only and not amp_check_only:
            # resuming on part of the data silently changes what the run learns
            # AND what its evaluation measures -- stop before spending GPU time
            raise SystemExit(f"run {task!r} was trained on cameras that aren't in the attached data: "
                             f"{', '.join(missing)}. Attach the dataset(s) with them (found only: "
                             f"{', '.join(str(r) for r in roots)}).")
        restore_checkpoint(ck, WORKING / "exps", task)
        log(f"restored checkpoint from {ck}")
    elif init_from:
        src = start_from(init_from, WORKING / "exps", task)
        if src is None:
            raise SystemExit(f"INIT_FROM = {init_from!r}: no checkpoint found. Attach that run's output "
                             f"(it contains exps/{init_from}/checkpoint/best.pth), or give a .pth path.")
        log(f"new run {task!r} starting from the weights in {src}")
    n = count_train_pairs(data_root)
    epochs = epochs or suggest_epochs(n)
    code = train(task, data_root, epochs=epochs, workers=workers, session_start=session_start,
                 extra_args=extra_train_args, gpus=gpus)
    write_status(task, epochs)
    compare(task, data_root)  # a minute or two on a GPU; leaves compare/compare.md in the output
    return "trained" if code == 0 else f"train.py exited with code {code}"
