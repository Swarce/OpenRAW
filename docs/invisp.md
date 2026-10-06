# InvISP: the optional learned path

`--invisp` replaces the classical deblock/chroma/tone-curve stages with
**InvISP** (Xing, Qian & Chen, *Invertible Image Signal Processing*, CVPR
2021) — an invertible neural network trained to map between RAW and
rendered sRGB. Full attribution in [NOTICE.md](../NOTICE.md).

> **Status: experimental.** It runs end to end and is covered by tests,
> but it has only two camera models and hasn't been evaluated against real
> RAW ground truth on real camera photos. The classical path is the default
> for good reason.

## Running it

```bash
pip install ".[invisp]"     # adds PyTorch
openraw photo.jpg --invisp --invisp-camera NIKON_D700   # or Canon_EOS_5D
```

Weights load from `pretrained/` relative to the current directory
(`--invisp-pretrained-dir` to change it). They are not part of the
installed package — see licensing below.

Expect colors to differ from the classical path: InvISP reconstructs its
learned approximation of the chosen camera's sensor-native response, not
an sRGB-preserving transform. Each checkpoint is camera-specific (upstream
says so explicitly); a JPEG from any other camera is still reconstructed
through that camera's learned behavior.

## What's in the repo

- **`openraw/third_party/invisp/`** — InvISP's own source, vendored
  (MIT): the `InvISPNet` architecture (affine coupling blocks with
  learnable invertible 1×1 convolutions) and its differentiable JPEG
  simulator. One line is patched: `torch.qr` → `torch.linalg.qr`, because
  current PyTorch removed `torch.qr` outright; the value it computes is
  overwritten by the checkpoint on load, so this can't affect results.
  `torch.lu`/`torch.lu_unpack` a few lines below are deprecated (they warn)
  but still work on PyTorch 2.14; if a future release removes them, the
  same `torch.linalg` swap applies, for the same reason.
- **`pretrained/canon.pth`, `pretrained/nikon.pth`** — upstream's official
  checkpoints, verified byte-identical (md5) to upstream.
- **`openraw/invisp_bridge.py`** — our own glue code. The subtle part:
  the checkpoints were trained with upstream's `--gamma` flag, so the
  network's "RAW" output is gamma-compressed, and the bridge undoes that
  before handing linear data to the shared `bitdepth.py`/`dng_writer.py`
  stages.
- **`openraw/ml/`** — superseded: a from-scratch NumPy demo of the same
  architecture class written before the real source was available. Kept
  only for its exact-invertibility tests; don't build on it.

## Open questions before this leaves "experimental"

- **Licensing of weights.** The checkpoints come from an MIT repository but
  were trained on MIT-Adobe FiveK, which has its own usage terms. Check
  those before redistributing weights publicly (including any you train).
- **Evaluation.** Needs a harness comparing reconstructions against real
  RAW files (FiveK's test split) before claims about quality can be made.
- **More cameras.** See [training.md](training.md).
