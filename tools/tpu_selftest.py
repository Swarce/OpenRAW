#!/usr/bin/env python3
"""Check that training on every TPU core can work, before spending hours on it:
one replica per core (as train.py --device xla --gpus 0 runs), each
all-reducing a gradient-sized tensor, results checked.

    python tools/tpu_selftest.py      # prints "TPU SELFTEST OK (N cores)" and exits 0

Needs PyTorch/XLA. Run it in its own process: a TPU can be held by one
process at a time.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PJRT_DEVICE", "TPU")


def _replica(index):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    dev = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
    rank, world = xr.global_ordinal(), xr.world_size()
    for i in range(3):
        g = torch.full((1_400_000,), float(rank + 1 + i), device=dev)
        xm.all_reduce(xm.REDUCE_SUM, [g])
        (torch_xla.sync if hasattr(torch_xla, "sync") else xm.mark_step)()
        expected = sum(r + 1 + i for r in range(world))
        if not (float(g.min().cpu()) == float(g.max().cpu()) == expected):
            raise RuntimeError(f"replica {rank}: all_reduce gave {float(g.min().cpu())}, expected {expected}")
    if rank == 0:
        print(f"TPU SELFTEST OK ({world} cores, {xr.device_type()})", flush=True)


def main() -> int:
    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ImportError as e:
        print(f"TPU SELFTEST FAILED: PyTorch/XLA not installed ({e})", flush=True)
        return 1
    xmp.spawn(_replica, args=())
    return 0


if __name__ == "__main__":
    sys.exit(main())
