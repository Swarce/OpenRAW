#!/usr/bin/env python3
"""Check that multi-GPU training can communicate, before spending hours on it.

Spawns one process per GPU and runs the collectives train.py --gpus uses: a
weight broadcast from rank 0, then gradient-sized all-reduces (~6 MB, like
InvISP's 1.4M parameters), checking the results. A broken interconnect --
e.g. GPU peer-to-peer over PCIe on some cloud VMs -- typically makes the first
collective hang forever rather than error, so callers should run this with a
timeout (the Kaggle runner does).

    python tools/multigpu_selftest.py              # all visible GPUs, NCCL
    python tools/multigpu_selftest.py --cpu 2      # two CPU processes over gloo (testing)

Prints "SELFTEST OK" and exits 0 on success. If it hangs or fails on GPUs, try
again with NCCL_P2P_DISABLE=1 set in the environment.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time


def _worker(rank: int, world: int, port: int, cpu: bool):
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    if not cpu:
        torch.cuda.set_device(rank)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo" if cpu else "nccl", rank=rank, world_size=world,
                            timeout=timedelta(seconds=120))
    dev = torch.device("cpu") if cpu else torch.device("cuda", rank)
    try:
        # 1) broadcast, as train.py does with the weights at start
        w = torch.full((1_400_000,), float(rank), device=dev)
        dist.broadcast(w, src=0)
        assert float(w.min()) == float(w.max()) == 0.0, "broadcast delivered wrong values"
        # 2) gradient-sized all-reduces, as every training step does
        for i in range(5):
            g = torch.full((1_400_000,), float(rank + 1 + i), device=dev)
            dist.all_reduce(g)
            expected = sum(r + 1 + i for r in range(world))
            assert float(g.min()) == float(g.max()) == expected, "all_reduce produced wrong values"
        if not cpu:
            torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cpu", type=int, default=0, metavar="N",
                    help="test N CPU processes over gloo instead of GPUs (for testing without GPUs)")
    a = ap.parse_args(argv)
    import torch
    import torch.multiprocessing as mp

    cpu = a.cpu > 0
    world = a.cpu if cpu else torch.cuda.device_count()
    if world < 2:
        print(f"SELFTEST SKIPPED: {world} GPU(s) visible, nothing to test")
        return 0
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    t = time.time()
    p2p = os.environ.get("NCCL_P2P_DISABLE", "0") == "1"
    print(f"selftest: {world} {'CPU processes (gloo)' if cpu else 'GPUs (NCCL' + (', P2P disabled)' if p2p else ')')} ...",
          flush=True)
    mp.spawn(_worker, args=(world, port, cpu), nprocs=world, join=True)
    print(f"SELFTEST OK ({time.time() - t:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
