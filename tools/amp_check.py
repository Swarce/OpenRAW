#!/usr/bin/env python3
"""Is mixed precision (train.py --amp) worth it on this GPU? Measures both
sides before any real training time is spent on it:

  speed     -- training-step time in float32 vs --amp, same crops, same GPU
  accuracy  -- on the same model and crops: how far the --amp losses and
               gradients are from float32's, and how exactly the network still
               inverts (raw -> rgb -> raw round trip)
  training  -- two copies from the same weights, trained N steps on the same
               crops, one per precision: do they end up equally good?

    python tools/amp_check.py --data_path data/          # real training crops
    python tools/amp_check.py                            # synthetic crops (no data)

Starts from pretrained/openraw-fivek-e53-best.pth (--checkpoint) so it measures
a trained network, not a random one. A few minutes on a GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _batches(data_path, n, device, crop=256, seed=0):
    """n training crops (input_raw, target_rgb), from real data or synthetic."""
    import torch
    g = torch.Generator().manual_seed(seed)
    if data_path:
        import train
        from dataset.FiveK_dataset import FiveKDatasetTrain
        dp = data_path if data_path.endswith("/") else data_path + "/"
        cams = train.resolve_cameras(SimpleNamespace(camera=None, all_downloaded=True, download=False,
                                                     download_jobs=1, delete_dngs=False, data_path=dp))
        ds = FiveKDatasetTrain(SimpleNamespace(data_path=dp, camera=cams, gamma=True, debug_mode=False))
        idx = torch.randperm(len(ds), generator=g)[:n].tolist()
        idx = (idx * (n // max(1, len(idx)) + 1))[:n]
        out = []
        for i in idx:
            s = ds[i]
            out.append((s["input_raw"][None].to(device), s["target_rgb"][None].to(device)))
        return out
    # synthetic: smooth colour fields plus fine texture, values in [0, 1]
    out = []
    for _ in range(n):
        low = torch.rand(1, 3, 8, 8, generator=g)
        img = torch.nn.functional.interpolate(low, size=(crop, crop), mode="bicubic", align_corners=False)
        img = (img + 0.05 * torch.randn(1, 3, crop, crop, generator=g)).clamp(0, 1)
        out.append((img.to(device), img.pow(1 / 1.5).clamp(0, 1).to(device)))
    return out


def _make(checkpoint, device, amp, blocks=8):
    import torch
    import train
    from openraw.third_party.invisp.model.model import InvISPNet
    net = InvISPNet(channel_in=3, channel_out=3, block_num=blocks).to(device)
    if checkpoint:
        from openraw.invisp_bridge import _load_net
        net.load_state_dict(_load_net("", "", device, checkpoint=checkpoint).state_dict())
    if amp:
        train._enable_amp(net, device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    return net, opt, scaler


def _losses(net, jpeg, x, rgb):
    import torch
    import torch.nn.functional as F
    r = torch.clamp(net(x), 0, 1)
    rgb_loss = F.l1_loss(r, rgb)
    raw_loss = F.l1_loss(net(jpeg(r), rev=True), x)
    return rgb_loss + raw_loss, raw_loss


def _step(net, opt, scaler, jpeg, x, rgb):
    opt.zero_grad()
    loss, _ = _losses(net, jpeg, x, rgb)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return float(loss.detach())


def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _psnr(a, b):
    import train
    return train._psnr(a, b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data_path", default=None, help="prepared training data (default: synthetic crops)")
    ap.add_argument("--checkpoint", default=str(ROOT / "pretrained" / "openraw-fivek-e53-best.pth"))
    ap.add_argument("--timing_steps", type=int, default=30)
    ap.add_argument("--train_steps", type=int, default=150, help="steps for the side-by-side training test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--blocks", type=int, default=8, help="InvISP depth (8 = the real network; less for testing)")
    ap.add_argument("--crop", type=int, default=256, help="crop size for synthetic crops (training uses 256)")
    a = ap.parse_args(argv)
    import torch
    from openraw.third_party.invisp.utils.JPEG import DiffJPEG

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = a.checkpoint if a.checkpoint and Path(a.checkpoint).is_file() and a.blocks == 8 else None
    jpeg = DiffJPEG(differentiable=True, quality=90).to(device)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"[amp] {name}; {'trained weights ' + Path(ckpt).name if ckpt else 'random weights'}; "
          f"{'real 256' if a.data_path else f'synthetic {a.crop}'} px crops", flush=True)
    data = _batches(a.data_path, max(a.timing_steps, a.train_steps) + 8, device, crop=a.crop)
    held_out = data[-8:]
    rows = {}

    # ---- speed
    for amp in (False, True):
        net, opt, scaler = _make(ckpt, device, amp, a.blocks)
        for x, rgb in data[:3]:  # warm-up: cuDNN autotuning, allocator
            _step(net, opt, scaler, jpeg, x, rgb)
        _sync(device)
        t = time.time()
        for x, rgb in data[:a.timing_steps]:
            _step(net, opt, scaler, jpeg, x, rgb)
        _sync(device)
        rows[amp] = (time.time() - t) / a.timing_steps
    speedup = rows[False] / rows[True]
    print(f"[amp] step time: float32 {rows[False] * 1000:.0f} ms, --amp {rows[True] * 1000:.0f} ms "
          f"-> {speedup:.2f}x", flush=True)

    # ---- accuracy on identical weights and crops
    ref, _, _ = _make(ckpt, device, False, a.blocks)
    mix, _, _ = _make(ckpt, device, True, a.blocks)
    mix.load_state_dict(ref.state_dict())
    cos, rel, dl, inv32, inv16 = [], [], [], [], []
    for x, rgb in held_out:
        grads = []
        for net in (ref, mix):
            net.zero_grad()
            loss, _ = _losses(net, jpeg, x, rgb)
            loss.backward()
            grads.append(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]))
            dl.append(float(loss.detach()))
        g32, g16 = grads
        cos.append(float(torch.nn.functional.cosine_similarity(g32, g16, dim=0)))
        rel.append(float((g16 - g32).norm() / g32.norm()))
        with torch.no_grad():
            inv32.append(_psnr(ref(ref(x), rev=True), x))
            inv16.append(_psnr(mix(mix(x), rev=True), x))
    loss_diff = sum(abs(dl[i + 1] - dl[i]) / dl[i] for i in range(0, len(dl), 2)) / (len(dl) / 2)
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    print(f"[amp] same weights, same crops: loss differs by {loss_diff * 100:.2f}% | gradient cosine "
          f"{mean(cos):.5f}, relative difference {mean(rel) * 100:.2f}% | round-trip raw->rgb->raw PSNR "
          f"float32 {mean(inv32):.1f} dB, --amp {mean(inv16):.1f} dB", flush=True)

    # ---- short training side by side
    finals = {}
    for amp in (False, True):
        torch.manual_seed(0)
        net, opt, scaler = _make(ckpt, device, amp, a.blocks)
        for x, rgb in data[:a.train_steps]:
            _step(net, opt, scaler, jpeg, x, rgb)
        clean, _, _ = _make(None, device, False, a.blocks)  # score in float32, like train.py's evaluation
        clean.load_state_dict(net.state_dict())
        with torch.no_grad():
            finals[amp] = mean([_psnr(clean(jpeg(torch.clamp(clean(x), 0, 1)), rev=True), x) for x, _ in held_out])
    print(f"[amp] after {a.train_steps} training steps from the same weights, held-out raw PSNR: "
          f"float32 {finals[False]:.2f} dB, --amp {finals[True]:.2f} dB", flush=True)

    gap = finals[False] - finals[True]
    ok_acc = mean(cos) > 0.99 and gap < 0.1
    verdict = ("worth it" if speedup >= 1.25 and ok_acc else
               "not worth it: too little speed-up" if ok_acc else
               "not worth it: accuracy cost")
    print(f"[amp] VERDICT: {verdict} ({speedup:.2f}x; gradient cosine {mean(cos):.4f}; "
          f"PSNR gap after training {gap:+.2f} dB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
