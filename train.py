import numpy as np
import os, time, random
import argparse
import json

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

# PATCHED (OpenRAW, not upstream) import paths: upstream's repo has
# model/ and utils/ as top-level packages sitting next to this file; in
# openraw they live under openraw/third_party/invisp/ instead (see /NOTICE.md).
# dataset/ and config/ ARE top-level here, matching upstream, so those
# two imports are unchanged.
from openraw.third_party.invisp.model.model import InvISPNet
from dataset.FiveK_dataset import FiveKDatasetTrain, FiveKDatasetTest
from config.config import get_arguments

from openraw.third_party.invisp.utils.JPEG import DiffJPEG


def _fivek_prepare():
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    import fivek_prepare
    return fivek_prepare


def resolve_cameras(args):
    """PATCHED (OpenRAW): the camera folders to train on, preparing FiveK ones
    as needed (--download etc.). Also used by tools/compare_models.py, so it
    evaluates on exactly the cameras -- and so the test images -- training does."""
    _fp = _fivek_prepare()
    _queries = [q.strip() for c in (args.camera or []) for q in c.split(",") if q.strip()]
    # folders prepared from other datasets (data/raise_prepare.py: RAISE_*) are used as they are
    _other = [q for q in _queries if q.startswith(_fp.OTHER_SOURCE_PREFIXES)]
    _queries = [q for q in _queries if q not in _other]
    if args.all_downloaded:
        _found = _fp.downloaded_cameras(args.data_path)
        _other += [c for c in _fp.prepared_other_sources(args.data_path) if c not in _other]
        if not _found and not _other:
            raise SystemExit(f"[data] --all-downloaded: no camera folders with DNGs under {args.data_path}fivek/raw/ "
                             f"and no prepared RAISE_* folders in {args.data_path}")
        print(f"[data] found {len(_found) + len(_other)} camera(s): " + ", ".join(_found + _other))
        _queries += [c for c in _found if c not in _queries]
    for _c in _other:
        if not (os.path.isfile(os.path.join(args.data_path, f"{_c}_train.txt"))
                and os.path.isdir(os.path.join(args.data_path, _c, "RAW"))):
            raise SystemExit(f"[data] {_c}: not prepared in {args.data_path} (run data/raise_prepare.py first)")
    if not _queries and not _other:
        _queries = ["NIKON_D700"]  # upstream's default camera
    return (_fp.prepare_cameras(_queries, args.data_path, download=args.download, jobs=args.download_jobs,
                                use_available=args.all_downloaded, delete_dngs=args.delete_dngs)
            if _queries else []) + _other


def eval_indices(n_test, n_images):
    """Which test images evaluation uses: n_images spread evenly over the test
    list (which runs camera by camera). Shared with tools/compare_models.py."""
    k = max(1, n_test // max(1, n_images))
    return list(range(0, n_test, k))[:n_images]


# PATCHED (OpenRAW, not upstream): all setup below runs only when this file is
# executed, not when it's imported. Upstream ran it at import time, which breaks
# multiprocessing's "spawn" start method (the default on Windows and macOS, and
# used for data preprocessing here): every worker re-imports the main script,
# and would re-run argument parsing, data prep and GPU setup recursively.
if __name__ == "__main__":
    parser = get_arguments()
    parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save checkpoint. ")
    parser.add_argument("--resume", dest='resume', action='store_true',  help="Resume training. ")
    parser.add_argument("--loss", type=str, default="L1", choices=["L1", "L2"], help="Choose which loss function to use. ")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--aug", dest='aug', action='store_true', help="Use data augmentation.")
    # PATCHED (OpenRAW, not upstream): parallel loading, proper resume, device.
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                        help="data-loading worker processes (default: up to 8). Loading an 18 MP pair takes "
                             "~0.3-0.5 s of CPU; with 0 workers the GPU mostly waits on it.")
    parser.add_argument("--epochs", type=int, default=300, help="total epochs (upstream: 300)")
    parser.add_argument("--time_limit_hours", type=float, default=0,
                        help="stop cleanly after the last epoch that fits in this many hours (checkpoint saved; "
                             "--resume continues). For time-limited sessions such as Kaggle's 12 h. 0 = no limit")
    parser.add_argument("--start_epoch", type=int, default=None,
                        help="with --resume on a weights-only checkpoint (made before full-state "
                             "checkpoints existed): the epoch to continue from")
    parser.add_argument("--device", default="cuda", help="'cuda' (default), 'cuda:1', ... or 'cpu' (slow; for testing)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="GPUs to train on in parallel (each takes a different image per step, gradients are "
                             "averaged: effective batch = gpus x batch_size). 0 = all available. --workers is "
                             "split across them.")
    parser.add_argument("--blocks", type=int, default=8, help="InvISP depth (upstream: 8). Smaller only for testing.")
    parser.add_argument("--eval_every", type=int, default=0,
                        help="evaluate on held-out test images every N epochs (default: ~10 times per run, "
                             "plus the last epoch); 0 = auto, -1 = never")
    parser.add_argument("--log_every", type=int, default=50,
                        help="print a step line every N steps, with losses averaged over them (default 50; 1 = every "
                             "step, as upstream). Per-step lines can swamp notebook log viewers.")
    parser.add_argument("--ema", type=float, default=0.999,
                        help="exponential moving average of the weights, updated every step with this decay "
                             "(0.999 averages over ~1,000 steps). Evaluation, best.pth, latest.pth and NNNN.pth "
                             "use the averaged weights, which are smoother and usually score a little higher "
                             "than any single step's. 0 = off.")
    parser.add_argument("--eval_images", type=int, default=40, help="max test images per evaluation (spread across cameras)")
    parser.add_argument("--eval_crop", type=int, default=512, help="centre crop size for evaluation")
    parser.add_argument("--amp", action="store_true",
                        help="mixed precision (experimental): the dense sub-networks, where nearly all the compute "
                             "is, run in float16 on CUDA (bfloat16 on CPU); the invertible coupling, the JPEG "
                             "simulation, the losses and evaluation stay float32. Check speed and accuracy first "
                             "with tools/amp_check.py.")
    parser.add_argument("--checkpointing", choices=["auto", "on", "off"], default="auto",
                        help="gradient checkpointing: recompute each block's activations in the backward pass instead "
                             "of storing them. A 256 px step needs ~6.2 GB without it, ~2.5 GB with it, for ~30-40%% "
                             "extra compute; results are identical. auto: on unless the GPU has >= 12 GB.")
    args = parser.parse_args()
    print("Parsed arguments: {}".format(args))

    # PATCHED (OpenRAW, not upstream): data preparation, BEFORE any GPU check so
    # it also works on machines without one (--prepare-only). Resolves FiveK camera
    # names, downloads only that camera's missing DNGs (--download), preprocesses
    # them (data/fivek_prepare.py) and writes the train/test lists the loader reads.
    if args.list_cameras:
        _fivek_prepare().list_cameras(args.data_path)
        raise SystemExit(0)
    args.camera = resolve_cameras(args)
    if args.prepare_only:
        print("[data] prepared:", ", ".join(args.camera))
        raise SystemExit(0)

    # PATCHED (OpenRAW, not upstream): the original here was
    #   os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    #   os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([...]))
    #   os.system('rm tmp')
    # which shells out to nvidia-smi, greps its output, and picks the GPU
    # with the most free memory -- a reasonable multi-GPU convenience, but
    # it crashes confusingly (FileNotFoundError or an empty-list np.argmax)
    # on any machine where nvidia-smi isn't on PATH in exactly the expected
    # form, or where CUDA isn't available at all. Same behavior when it
    # works, a clear error instead of a cryptic one when it can't -- see
    # docs/training.md for what this means on your actual machine.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        # Diagnose WHY, instead of just "no GPU" -- the usual cause on Windows
        # and macOS is a CPU-only PyTorch build, which `pip install torch` from
        # PyPI installs there by default, even on machines with an NVIDIA card.
        import shutil, sys as _s
        _cuda_build = torch.version.cuda
        _smi = shutil.which("nvidia-smi")
        if _cuda_build is None:
            _why = (f"Your PyTorch ({torch.__version__}) is a CPU-only build: it can't use any GPU. "
                    f"On Windows/macOS, `pip install torch` from PyPI installs that build by default.\n"
                    f"With an NVIDIA GPU, install the CUDA build:\n"
                    f"    pip uninstall -y torch torchvision\n"
                    f"    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130\n"
                    f"(Python {_s.version_info.major}.{_s.version_info.minor} needs a current CUDA index like cu130; "
                    f"older ones such as cu118/cu121 have no wheels for new Pythons.)")
        elif not _smi:
            _why = (f"PyTorch is a CUDA {_cuda_build} build, but no NVIDIA driver was found (nvidia-smi not on PATH). "
                    f"Install/update the NVIDIA driver. AMD/Intel GPUs can't run CUDA builds.")
        else:
            _why = (f"PyTorch is a CUDA {_cuda_build} build and an NVIDIA driver is present, but CUDA still isn't "
                    f"usable -- usually a driver too old for CUDA {_cuda_build}. Update the driver, or install a "
                    f"PyTorch build for an older CUDA (see https://pytorch.org/get-started/locally/).")
        raise RuntimeError("No usable CUDA GPU (torch.cuda.is_available() is False).\n" + _why +
                           "\nSee docs/training.md.")
    # PATCHED (OpenRAW, not upstream): upstream picked the GPU with the most free
    # memory via `nvidia-smi | grep` + `rm`, setting CUDA_VISIBLE_DEVICES. That
    # fails on Windows (no grep/rm) and can be ignored once CUDA is initialized.
    # Now: ask nvidia-smi for free memory directly (any OS) and select the GPU
    # by device index. Only when --device is plain "cuda" and there are 2+ GPUs.
    if args.gpus == 0:
        args.gpus = max(1, torch.cuda.device_count()) if args.device.startswith("cuda") else 1
    if args.gpus > 1 and args.device.startswith("cuda") and torch.cuda.device_count() < args.gpus:
        raise SystemExit(f"--gpus {args.gpus}: only {torch.cuda.device_count()} GPU(s) visible")
    if args.gpus == 1 and args.device == "cuda" and torch.cuda.device_count() > 1:
        try:
            import subprocess
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout
            free_mem = [int(x) for x in out.split()]
            if len(free_mem) == torch.cuda.device_count():
                args.device = "cuda:%d" % int(np.argmax(free_mem))
                print(f"[INFO] using {args.device} (most free memory)")
        except Exception as e:
            print(f"[WARN] GPU auto-select failed ({e}); using cuda:0")

    # (the differentiable JPEG module is now created inside main(), per process:
    # building it here would initialize CUDA on GPU 0 in the parent process)

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(args.out_path+"%s"%args.task, exist_ok=True)
    os.makedirs(args.out_path+"%s/checkpoint"%args.task, exist_ok=True)

    with open(args.out_path+"%s/commandline_args.yaml"%args.task , 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def _psnr(a, b):
    mse = torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).item()
    return 99.0 if mse == 0 else 10 * float(np.log10(1.0 / mse))


def evaluate(net, loader, device):
    """PATCHED (OpenRAW): held-out evaluation, absent upstream (it evaluated in
    separate scripts after training). Returns mean PSNRs over the test crops:
      raw_psnr -- the REAL JPEG through the INVERSE network vs the true raw:
                  OpenRAW's actual use case, and the metric for best.pth;
      rgb_psnr -- raw through the forward network vs the JPEG."""
    was_training = net.training
    net.eval()
    raw_p, rgb_p = [], []
    with torch.no_grad():
        for b in loader:
            raw, rgb = b["input_raw"].to(device), b["target_rgb"].to(device)
            rgb_p.append(_psnr(net(raw), rgb))
            raw_p.append(_psnr(net(rgb, rev=True), raw))
    net.train(was_training)
    return float(np.mean(raw_p)), float(np.mean(rgb_p))


def _amp_context(device_type, dtype):
    """The reduced-precision region (a function so tests can swap it out)."""
    return torch.autocast(device_type, dtype=dtype)


def _enable_amp(net, device):
    """PATCHED (OpenRAW): mixed precision, applied where it's safe. Each
    InvBlock's coupling (y1 = x1 + F(x2); y2 = x2 * exp(s(y1)) + G(y1)) is
    invertible whatever F, G and H compute, as long as the additions,
    multiplications and exp themselves are exact enough -- so only the dense
    sub-networks F, G, H (all the convolutions, ~all the FLOPs) run in reduced
    precision, and hand float32 back to the coupling. The invertible 1x1
    convolutions, the differentiable JPEG, the losses and evaluation are
    untouched. Wraps forward() only; the state_dict is unchanged."""
    from openraw.third_party.invisp.model.model import DenseBlock
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    n = 0
    for m in net.modules():
        if isinstance(m, DenseBlock):
            f = m.forward
            def fwd(x, f=f):
                with _amp_context(device.type, dtype):
                    return f(x).float()
            m.forward = fwd
            n += 1
    return n


def _enable_checkpointing(net):
    """Gradient checkpointing per InvBlock (PATCHED, OpenRAW): each block's
    activations are recomputed during backward instead of stored. Measured: a
    256 px training step needs ~5.7 GB of activations (+ CUDA context) without
    it -- more than a 6 GB GPU, where Windows silently spills into system RAM
    and steps took 7-21 s on a user's RTX 3050 -- and ~2.5 GB with it. Exact:
    identical loss and gradients for all parameters (verified, test pinned).
    Wraps forward() only; the state_dict is unchanged."""
    from torch.utils.checkpoint import checkpoint
    for op in net.operations:
        f = op.forward
        op.forward = (lambda f: lambda x, rev=False: checkpoint(f, x, rev, use_reentrant=False))(f)


def _save(obj, path):
    """Write-then-rename: an interrupted save never leaves a corrupt checkpoint."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)  # our own files
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location=device)


class _EMA:
    """PATCHED (OpenRAW): exponential moving average of the weights. With a
    small final learning rate and batch size 1-2, the live weights keep
    jittering around a good solution, so any one snapshot can score noticeably
    better or worse than its neighbours (held-out raw PSNR moved 39.47 ->
    38.80 dB between two evaluations 18 epochs apart). The average over the
    last ~1/(1-decay) steps doesn't jitter, and is what gets evaluated and
    saved for inference. Lives on rank 0 only (it alone evaluates and saves);
    the live weights are identical on every rank anyway.

    Warm-up: the effective decay is min(decay, (1+n)/(10+n)) after n updates,
    so a freshly started average isn't dominated by its starting point."""

    def __init__(self, net, decay, state=None, updates=0):
        self.decay, self.updates = decay, updates
        self.src = net.state_dict()  # live references to the training weights
        self.avg = {k: v.detach().clone() for k, v in self.src.items()}
        if state is not None:
            for k, v in state.items():
                self.avg[k].copy_(v)
        self.keys_f = [k for k, v in self.avg.items() if v.is_floating_point()]
        self.keys_other = [k for k, v in self.avg.items() if not v.is_floating_point()]

    @torch.no_grad()
    def update(self):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        torch._foreach_lerp_([self.avg[k] for k in self.keys_f], [self.src[k] for k in self.keys_f], 1.0 - d)
        for k in self.keys_other:
            self.avg[k].copy_(self.src[k])

    def state_dict(self):
        return self.avg


def _average_gradients(params, world):
    """PATCHED (OpenRAW): multi-GPU data parallelism by hand. Each process ran
    forward + inverse + backward on its own image; average the gradients so
    every process takes the identical optimizer step. Not DDP: InvISP runs the
    SAME network twice (forward, then inverse) before one backward, which DDP's
    once-per-backward gradient hooks aren't designed around. One all-reduce of
    all gradients flattened (~1.4M floats, ~6 MB): a few ms per step."""
    import torch.distributed as dist
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat /= world
    off = 0
    for p, g in zip(params, grads):
        n = g.numel()
        p.grad = flat[off:off + n].view_as(p)
        off += n


def _worker(rank, world, args, port):
    """One process per GPU (spawned by launch())."""
    import torch.distributed as dist
    cuda = args.device.startswith("cuda")
    if cuda:
        torch.cuda.set_device(rank)
        args.device = f"cuda:{rank}"
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    from datetime import timedelta
    # A collective that never completes (e.g. a broken GPU interconnect) raises
    # after this long instead of hanging the session. Generous enough for rank
    # 0's per-epoch evaluation + saves, during which the others wait.
    dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=world,
                            timeout=timedelta(minutes=15))
    if rank:  # only rank 0 talks; the others' output would duplicate it
        import builtins
        builtins.print = lambda *a, **k: None
    try:
        main(args, rank=rank, world=world)
    finally:
        dist.destroy_process_group()


def launch(args):
    """Train on args.gpus processes (1 = plain single-process training)."""
    world = getattr(args, "gpus", 1) or 1
    if world == 1:
        return main(args)
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as sk:  # a free local port for the processes to rendezvous on
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    print(f"[INFO] multi-GPU: {world} processes, one per {'GPU' if args.device.startswith('cuda') else 'CPU process'}; "
          f"effective batch size {world * args.batch_size}"
          + ("; NCCL_P2P_DISABLE=1" if os.environ.get("NCCL_P2P_DISABLE") == "1" else ""), flush=True)
    mp.spawn(_worker, args=(world, args, port), nprocs=world, join=True)


def main(args, rank=0, world=1):
    # PATCHED (OpenRAW, not upstream) throughout: device-agnostic, parallel
    # data loading, full-state checkpoints + exact resume, honest timing.
    device = torch.device(args.device)
    ckpt = args.out_path + "%s/checkpoint/" % args.task
    os.makedirs(ckpt, exist_ok=True)
    # ======================================define the model======================================
    net = InvISPNet(channel_in=3, channel_out=3, block_num=getattr(args, "blocks", 8)).to(device)
    jpeg = DiffJPEG if isinstance(DiffJPEG, torch.nn.Module) else DiffJPEG(differentiable=True, quality=90)
    jpeg = jpeg.to(device)
    gpu_gb = torch.cuda.get_device_properties(device).total_memory / 2**30 if device.type == "cuda" else None
    use_ckpt = {"on": True, "off": False}.get(getattr(args, "checkpointing", "auto"), gpu_gb is None or gpu_gb < 12)
    use_amp = bool(getattr(args, "amp", False))
    if use_amp:
        n_sub = _enable_amp(net, device)
        print(f"[INFO] mixed precision: {n_sub} dense sub-networks in "
              f"{'float16' if device.type == 'cuda' else 'bfloat16'}, everything else float32")
    # float16 gradients can underflow to zero: scale the loss up for backward, and
    # skip steps whose gradients overflowed (the scale adapts). Off = a no-op.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    if use_ckpt:
        _enable_checkpointing(net)
    print(f"[INFO] gradient checkpointing: {'on' if use_ckpt else 'off'}"
          + (f" (GPU {gpu_gb:.1f} GB; a 256 px step needs ~6.2 GB without it, ~2.5 GB with it)" if gpu_gb else ""))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # PATCHED (OpenRAW): upstream's LR drops at epochs 50 and 80 of 300 were
    # tuned for ONE camera (~650 images, ~195k steps). They now scale with
    # --epochs, so the drops land at the same fraction of training whatever
    # the run length (identical to upstream at --epochs 300).
    milestones = sorted({max(1, round(args.epochs * m / 300)) for m in (50, 80)})
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    start_epoch, step = 0, 0

    if args.resume:
        if os.path.exists(ckpt + "latest_state.pth"):
            st = _load(ckpt + "latest_state.pth", device)
            net.load_state_dict(st["net"])
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            start_epoch, step = st["epoch"] + 1, st["step"]
            print(f"[INFO] resumed from {ckpt}latest_state.pth: continuing at epoch {start_epoch}, step {step}")
        elif os.path.exists(ckpt + "latest.pth"):
            # weights-only checkpoint (upstream format / runs started before full-state saves)
            net.load_state_dict(_load(ckpt + "latest.pth", device))
            start_epoch = args.start_epoch or 0
            for _ in range(start_epoch):  # fast-forward the LR schedule to where training was
                scheduler.step()
            print(f"[INFO] loaded weights from {ckpt}latest.pth (no optimizer/epoch state saved in it): "
                  f"continuing at epoch {start_epoch}" + ("" if args.start_epoch else
                  " -- pass --start_epoch N to continue the epoch count and LR schedule") +
                  "; optimizer momentum restarts")
        else:
            raise SystemExit(f"--resume: no checkpoint in {ckpt}")

    if world > 1:
        # every process must start from identical weights (also after resume)
        import torch.distributed as dist
        for t in net.state_dict().values():
            dist.broadcast(t, src=0)

    ema, eval_net = None, net
    if rank == 0 and getattr(args, "ema", 0) and args.ema > 0:
        st = _load(ckpt + "latest_state.pth", device) if args.resume and os.path.exists(ckpt + "latest_state.pth") else {}
        ema = _EMA(net, args.ema, st.get("ema"), st.get("ema_updates", 0))
        # evaluated in a separate network built from scratch -- not a deepcopy of
        # `net`, whose checkpointed forward closures would still point at `net`
        eval_net = InvISPNet(channel_in=3, channel_out=3, block_num=getattr(args, "blocks", 8)).to(device)
        print(f"[INFO] weight averaging (EMA) decay {args.ema}: evaluation and saved weights use the average"
              + (" (continuing the saved average)" if st.get("ema") is not None else
                 " (starting from the current weights)" if args.resume else ""))
    elif rank == 0 and use_amp:
        # evaluation always runs in full precision, so scores stay comparable
        eval_net = InvISPNet(channel_in=3, channel_out=3, block_num=getattr(args, "blocks", 8)).to(device)

    print("[INFO] Start data loading and preprocessing")
    RAWDataset = FiveKDatasetTrain(opt=args)
    sampler = None
    workers = args.workers if world == 1 else max(1, -(-args.workers // world))  # --workers is split across GPUs
    if world > 1:
        from torch.utils.data.distributed import DistributedSampler
        # each process gets a different, disjoint share of every epoch
        sampler = DistributedSampler(RAWDataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(RAWDataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                            drop_last=True, num_workers=workers, persistent_workers=workers > 0,
                            pin_memory=device.type == "cuda")
    # (no worker_init_fn needed: since PyTorch 1.9 each worker gets its own NumPy
    # seed, so augmentations differ across workers -- verified; we require >=1.10)
    if args.resume and step == 0 and start_epoch:  # weights-only resume: estimate the step count
        step = start_epoch * len(dataloader)
    if device.type == "cuda":
        # fixed 256 px input: let cuDNN benchmark and pick the fastest conv algorithms
        torch.backends.cudnn.benchmark = True
    total_steps = len(dataloader) * args.epochs
    images_per_epoch = len(dataloader) * args.batch_size * world
    print(f"[INFO] LR schedule: x0.5 at epochs {milestones} of {args.epochs}; {total_steps:,} steps in total")
    if images_per_epoch and abs(images_per_epoch * args.epochs / 195_000 - 1) > 0.5:
        print(f"[INFO] note: upstream InvISP trained on ~195,000 image samples (one camera, 650 images x 300 "
              f"epochs). For a similar amount of training on this dataset: "
              f"--epochs {max(1, round(195_000 / images_per_epoch))}")
    print(f"[INFO] {len(RAWDataset)} training images, {len(dataloader)} steps/epoch"
          + (f" per GPU ({world} GPUs, {images_per_epoch} images/epoch)" if world > 1 else "")
          + f", epochs {start_epoch}..{args.epochs - 1}, {workers} loader worker(s)"
          + (" per GPU" if world > 1 else "") + f", device {device}")

    eval_loader, best_raw = None, -1.0
    if args.eval_every >= 0 and rank == 0:  # only rank 0 evaluates and saves
        try:
            test_set = FiveKDatasetTest(opt=args)
        except (FileNotFoundError, OSError):
            test_set = None
        if test_set is not None and len(test_set):
            test_set.eval_crop = args.eval_crop
            idx = eval_indices(len(test_set), args.eval_images)  # spread across cameras
            eval_loader = DataLoader(torch.utils.data.Subset(test_set, idx), batch_size=1, shuffle=False,
                                     num_workers=min(2, args.workers))
            eval_every = args.eval_every or max(1, args.epochs // 10)
            print(f"[INFO] evaluation: {len(idx)} held-out test images ({args.eval_crop} px centre crops) "
                  f"every {eval_every} epoch(s) and at the end -> {args.out_path}{args.task}/eval.csv, "
                  f"best.pth by raw PSNR")
        else:
            print("[INFO] evaluation: no test images found -- skipped")
    if args.resume and os.path.exists(ckpt + "latest_state.pth"):
        best_raw = _load(ckpt + "latest_state.pth", device).get("best_raw_psnr", -1.0)

    print("[INFO] Start to train")
    run_start, epochs_done, start_step = time.time(), 0, step
    weights = (lambda: ema.state_dict()) if ema is not None else net.state_dict  # what gets saved for inference
    log_every = max(1, getattr(args, "log_every", 50) or 1)
    params = list(net.parameters())
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)  # a different shuffle each epoch, the same on every process
        epoch_time = time.time()
        data_t0 = time.time()
        # running sums for --log_every: kept on the device and only read when a
        # line is printed, so steps between log lines don't wait on a GPU sync
        win = torch.zeros(3, device=device); win_n = 0; win_data = win_compute = 0.0
        ep_sum = torch.zeros(3, device=device); ep_n = 0
        for i_batch, sample_batched in enumerate(dataloader):
            data_time = time.time() - data_t0  # waiting for the loader -- upstream didn't count this
            step_time = time.time()

            input, target_rgb, target_raw = (sample_batched['input_raw'].to(device, non_blocking=True),
                                             sample_batched['target_rgb'].to(device, non_blocking=True),
                                             sample_batched['target_raw'].to(device, non_blocking=True))

            reconstruct_rgb = net(input)
            reconstruct_rgb = torch.clamp(reconstruct_rgb, 0, 1)
            rgb_loss = F.l1_loss(reconstruct_rgb, target_rgb)
            reconstruct_rgb = jpeg(reconstruct_rgb)
            reconstruct_raw = net(reconstruct_rgb, rev=True)
            raw_loss = F.l1_loss(reconstruct_raw, target_raw)

            loss = args.rgb_weight * rgb_loss + raw_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if world > 1:
                # averaged before the scaler looks at them: an overflow on any
                # process becomes inf on every process, so all skip the same steps
                _average_gradients(params, world)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update()

            losses = torch.stack([loss.detach(), raw_loss.detach(), rgb_loss.detach()])
            win += losses; ep_sum += losses; win_n += 1; ep_n += 1
            win_data += data_time; win_compute += time.time() - step_time
            last_in_epoch = i_batch == len(dataloader) - 1
            if win_n >= log_every or step == start_step or last_in_epoch:
                l, r, g = (win / win_n).tolist()
                print("task: %s Epoch: %d Step: %d || loss: %.5f raw_loss: %.5f rgb_loss: %.5f || lr: %f || data %.3fs compute %.3fs%s" % (
                    args.task, epoch, step, l, r, g, optimizer.param_groups[0]['lr'], win_data / win_n,
                    win_compute / win_n, f" (mean of {win_n} steps)" if win_n > 1 else ""))
                win.zero_(); win_n = 0; win_data = win_compute = 0.0
            if step == start_step and device.type == "cuda":
                used = torch.cuda.max_memory_reserved(device) / 2**30
                print(f"[INFO] GPU memory after first step: {used:.2f} of {gpu_gb:.1f} GB")
                if used > 0.9 * gpu_gb:
                    print("[WARN] GPU memory is nearly full. On Windows the NVIDIA driver then silently spills "
                          "into system RAM and steps get 10-100x slower. Use --checkpointing on, or set NVIDIA "
                          "Control Panel > CUDA - Sysmem Fallback Policy > Prefer No Sysmem Fallback to get an "
                          "out-of-memory error instead of a silent slowdown.")
            step += 1
            data_t0 = time.time()

        scheduler.step()
        if rank != 0:  # rank 0 evaluates, saves and decides whether to stop
            if world > 1 and _stop_flag(False, rank):
                break
            continue
        if eval_loader is not None and ((epoch + 1) % eval_every == 0 or epoch == args.epochs - 1):
            t_eval = time.time()
            if eval_net is not net:
                eval_net.load_state_dict(ema.state_dict() if ema is not None else net.state_dict())
            raw_psnr, rgb_psnr = evaluate(eval_net, eval_loader, device)
            is_best = raw_psnr > best_raw
            if is_best:
                best_raw = raw_psnr
                _save(weights(), ckpt + "best.pth")
            csv = args.out_path + "%s/eval.csv" % args.task
            new_file = not os.path.exists(csv)
            with open(csv, "a") as f:
                if new_file:
                    f.write("epoch,step,raw_psnr,rgb_psnr\n")
                f.write(f"{epoch},{step},{raw_psnr:.4f},{rgb_psnr:.4f}\n")
            print(f"[EVAL] epoch {epoch}{' (EMA)' if ema is not None else ''}: raw PSNR {raw_psnr:.2f} dB "
                  f"(JPEG -> raw, the OpenRAW direction) | "
                  f"rgb PSNR {rgb_psnr:.2f} dB" + (" | new best -> best.pth" if is_best else f" | best {best_raw:.2f}")
                  + f" | {time.time() - t_eval:.0f}s")
        # weights-only latest.pth stays compatible with `openraw --invisp`;
        # latest_state.pth carries everything --resume needs (live weights + average)
        _save(weights(), ckpt + "latest.pth")
        _save({"net": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "epoch": epoch, "step": step, "best_raw_psnr": best_raw,
               **({"ema": ema.state_dict(), "ema_updates": ema.updates} if ema is not None else {})},
              ckpt + "latest_state.pth")
        if (epoch + 1) % 10 == 0:
            _save(weights(), ckpt + "%04d.pth" % epoch)
            print("[INFO] Successfully saved " + ckpt + "%04d.pth" % epoch)

        epochs_done += 1
        took = time.time() - epoch_time
        left = (time.time() - run_start) / epochs_done * (args.epochs - epoch - 1)
        m = (ep_sum / max(ep_n, 1)).tolist()
        print("[INFO] Epoch %d time: %.1fs | mean loss %.5f raw_loss %.5f rgb_loss %.5f | "
              "ETA for remaining %d epoch(s): %.1f h | task: %s" % (
                  epoch, took, m[0], m[1], m[2], args.epochs - epoch - 1, left / 3600, args.task))
        # PATCHED (OpenRAW): time budget. Stop BETWEEN epochs -- the checkpoint for
        # this one is already saved -- if the next one (with a 15% margin for an
        # evaluation pass) wouldn't finish in time, instead of being killed mid-save.
        limit = getattr(args, "time_limit_hours", 0) or 0
        stop = False
        if limit and epoch < args.epochs - 1:
            elapsed = time.time() - run_start
            per_epoch = elapsed / epochs_done
            if elapsed + 1.15 * per_epoch > limit * 3600:
                print(f"[INFO] time limit: stopping after epoch {epoch} ({elapsed / 3600:.2f} h used of "
                      f"{limit:.2f} h; next epoch needs ~{per_epoch / 3600:.2f} h). Continue with --resume.")
                stop = True
        # with several GPUs, every process must stop at the same epoch -- one
        # quitting alone would leave the others waiting forever in all_reduce
        if world > 1:
            stop = _stop_flag(stop, rank)
        if stop:
            break


def _stop_flag(stop, rank):
    """Rank 0's stop decision, broadcast to every process."""
    import torch.distributed as dist
    dev = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    t = torch.tensor([1 if stop else 0], device=dev)
    dist.broadcast(t, src=0)
    return bool(t.item())

if __name__ == '__main__':

    torch.set_num_threads(4)
    launch(args)
