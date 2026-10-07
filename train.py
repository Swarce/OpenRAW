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
    parser.add_argument("--start_epoch", type=int, default=None,
                        help="with --resume on a weights-only checkpoint (made before full-state "
                             "checkpoints existed): the epoch to continue from")
    parser.add_argument("--device", default="cuda", help="'cuda' (default), 'cuda:1', ... or 'cpu' (slow; for testing)")
    parser.add_argument("--eval_every", type=int, default=0,
                        help="evaluate on held-out test images every N epochs (default: ~10 times per run, "
                             "plus the last epoch); 0 = auto, -1 = never")
    parser.add_argument("--eval_images", type=int, default=40, help="max test images per evaluation (spread across cameras)")
    parser.add_argument("--eval_crop", type=int, default=512, help="centre crop size for evaluation")
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
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    import fivek_prepare as _fp
    if args.list_cameras:
        _fp.list_cameras(args.data_path)
        raise SystemExit(0)
    _queries = [q.strip() for c in (args.camera or []) for q in c.split(",") if q.strip()]
    if args.all_downloaded:
        _found = _fp.downloaded_cameras(args.data_path)
        if not _found:
            raise SystemExit(f"[data] --all-downloaded: no camera folders with DNGs under {args.data_path}fivek/raw/")
        print(f"[data] found {len(_found)} downloaded camera(s): " + ", ".join(_found))
        _queries += [c for c in _found if c not in _queries]
    args.camera = _fp.prepare_cameras(_queries or ["NIKON_D700"], args.data_path, download=args.download,
                                      jobs=args.download_jobs, use_available=args.all_downloaded,
                                      delete_dngs=args.delete_dngs)
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
    if args.device == "cuda" and torch.cuda.device_count() > 1:
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

    DiffJPEG = DiffJPEG(differentiable=True, quality=90).to(args.device)

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


def main(args):
    # PATCHED (OpenRAW, not upstream) throughout: device-agnostic, parallel
    # data loading, full-state checkpoints + exact resume, honest timing.
    device = torch.device(args.device)
    ckpt = args.out_path + "%s/checkpoint/" % args.task
    os.makedirs(ckpt, exist_ok=True)
    # ======================================define the model======================================
    net = InvISPNet(channel_in=3, channel_out=3, block_num=8).to(device)
    gpu_gb = torch.cuda.get_device_properties(device).total_memory / 2**30 if device.type == "cuda" else None
    use_ckpt = {"on": True, "off": False}.get(getattr(args, "checkpointing", "auto"), gpu_gb is None or gpu_gb < 12)
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

    print("[INFO] Start data loading and preprocessing")
    RAWDataset = FiveKDatasetTrain(opt=args)
    dataloader = DataLoader(RAWDataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.workers, persistent_workers=args.workers > 0,
                            pin_memory=device.type == "cuda")
    # (no worker_init_fn needed: since PyTorch 1.9 each worker gets its own NumPy
    # seed, so augmentations differ across workers -- verified; we require >=1.10)
    if args.resume and step == 0 and start_epoch:  # weights-only resume: estimate the step count
        step = start_epoch * len(dataloader)
    if device.type == "cuda":
        # fixed 256 px input: let cuDNN benchmark and pick the fastest conv algorithms
        torch.backends.cudnn.benchmark = True
    total_steps = len(dataloader) * args.epochs
    print(f"[INFO] LR schedule: x0.5 at epochs {milestones} of {args.epochs}; {total_steps:,} steps in total")
    if len(dataloader) and abs(total_steps / 195_000 - 1) > 0.5:
        print(f"[INFO] note: upstream InvISP trained ~195,000 steps (one camera, 650 images x 300 epochs). "
              f"For a similar amount of training on this dataset: --epochs {max(1, round(195_000 / len(dataloader)))}")
    print(f"[INFO] {len(RAWDataset)} training images, {len(dataloader)} steps/epoch, "
          f"epochs {start_epoch}..{args.epochs - 1}, {args.workers} loader worker(s), device {device}")

    eval_loader, best_raw = None, -1.0
    if args.eval_every >= 0:
        try:
            test_set = FiveKDatasetTest(opt=args)
        except (FileNotFoundError, OSError):
            test_set = None
        if test_set is not None and len(test_set):
            test_set.eval_crop = args.eval_crop
            k = max(1, len(test_set) // max(1, args.eval_images))
            idx = list(range(0, len(test_set), k))[:args.eval_images]  # spread across cameras
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
    for epoch in range(start_epoch, args.epochs):
        epoch_time = time.time()
        data_t0 = time.time()
        for i_batch, sample_batched in enumerate(dataloader):
            data_time = time.time() - data_t0  # waiting for the loader -- upstream didn't count this
            step_time = time.time()

            input, target_rgb, target_raw = (sample_batched['input_raw'].to(device, non_blocking=True),
                                             sample_batched['target_rgb'].to(device, non_blocking=True),
                                             sample_batched['target_raw'].to(device, non_blocking=True))

            reconstruct_rgb = net(input)
            reconstruct_rgb = torch.clamp(reconstruct_rgb, 0, 1)
            rgb_loss = F.l1_loss(reconstruct_rgb, target_rgb)
            reconstruct_rgb = DiffJPEG(reconstruct_rgb)
            reconstruct_raw = net(reconstruct_rgb, rev=True)
            raw_loss = F.l1_loss(reconstruct_raw, target_raw)

            loss = args.rgb_weight * rgb_loss + raw_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("task: %s Epoch: %d Step: %d || loss: %.5f raw_loss: %.5f rgb_loss: %.5f || lr: %f || data %.3fs compute %.3fs" % (
                args.task, epoch, step, loss.detach().cpu().numpy(), raw_loss.detach().cpu().numpy(),
                rgb_loss.detach().cpu().numpy(), optimizer.param_groups[0]['lr'], data_time, time.time() - step_time))
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
        if eval_loader is not None and ((epoch + 1) % eval_every == 0 or epoch == args.epochs - 1):
            t_eval = time.time()
            raw_psnr, rgb_psnr = evaluate(net, eval_loader, device)
            is_best = raw_psnr > best_raw
            if is_best:
                best_raw = raw_psnr
                _save(net.state_dict(), ckpt + "best.pth")
            csv = args.out_path + "%s/eval.csv" % args.task
            new_file = not os.path.exists(csv)
            with open(csv, "a") as f:
                if new_file:
                    f.write("epoch,step,raw_psnr,rgb_psnr\n")
                f.write(f"{epoch},{step},{raw_psnr:.4f},{rgb_psnr:.4f}\n")
            print(f"[EVAL] epoch {epoch}: raw PSNR {raw_psnr:.2f} dB (JPEG -> raw, the OpenRAW direction) | "
                  f"rgb PSNR {rgb_psnr:.2f} dB" + (" | new best -> best.pth" if is_best else f" | best {best_raw:.2f}")
                  + f" | {time.time() - t_eval:.0f}s")
        # weights-only latest.pth stays compatible with `openraw --invisp`;
        # latest_state.pth carries everything --resume needs
        _save(net.state_dict(), ckpt + "latest.pth")
        _save({"net": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "epoch": epoch, "step": step, "best_raw_psnr": best_raw}, ckpt + "latest_state.pth")
        if (epoch + 1) % 10 == 0:
            _save(net.state_dict(), ckpt + "%04d.pth" % epoch)
            print("[INFO] Successfully saved " + ckpt + "%04d.pth" % epoch)

        epochs_done += 1
        took = time.time() - epoch_time
        left = (time.time() - run_start) / epochs_done * (args.epochs - epoch - 1)
        print("[INFO] Epoch %d time: %.1fs | ETA for remaining %d epoch(s): %.1f h | task: %s" % (
            epoch, took, args.epochs - epoch - 1, left / 3600, args.task))

if __name__ == '__main__':

    torch.set_num_threads(4)
    main(args)
