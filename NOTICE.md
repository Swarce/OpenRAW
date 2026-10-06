# Third-party attribution

## `openraw/third_party/invisp/` — vendored source code (not just cited)

This project vendors a subset of the actual source code from:

> Yazhou Xing\*, Zian Qian\*, Qifeng Chen (\*joint first authors).
> **Invertible Image Signal Processing.** CVPR 2021.
> Code: <https://github.com/yzxing87/Invertible-ISP>
> Paper: <https://arxiv.org/abs/2103.15061>
> License: **MIT** (full text preserved at `openraw/third_party/invisp/LICENSE`,
> unmodified, as the license requires)

```bibtex
@inproceedings{xing21invertible,
  title     = {Invertible Image Signal Processing},
  author    = {Xing, Yazhou and Qian, Zian and Chen, Qifeng},
  booktitle = {CVPR},
  year      = {2021}
}
```

**What was vendored from the upstream repo (pulled 2026-10-03):**
`model/model.py`, `model/modules.py`, `model/utils.py`, `model/loss.py`,
`utils/JPEG.py`, `utils/JPEG_utils.py`, `utils/compression.py`,
`utils/decompression.py`, `utils/commons.py`, and `LICENSE`. Plus the
addition of `openraw/third_party/invisp/__init__.py` and
`openraw/third_party/invisp/utils/__init__.py` (upstream's `utils/` wasn't a
package; it needs to be here so it doesn't collide with any other
top-level `utils` import in this project).

**One line of `model/modules.py` is patched, not pristine**, marked
inline with a `PATCHED (OpenRAW, not upstream)` comment at the call
site: `InvertibleConv1x1.__init__` used `torch.qr(...)`, which current
PyTorch (verified: 2.14.1) has removed outright (raises `RuntimeError`,
not a deprecation warning — this was caught by actually running the
code, not by reading changelogs). Swapped for the equivalent
`torch.linalg.qr(..., mode="reduced")`. This value seeds an initial
orthogonal matrix that `load_state_dict()` immediately overwrites from
the checkpoint in every real use of this class, so the change cannot
affect correctness. `torch.lu`/`torch.lu_unpack` a few lines below are
untouched — deprecated-with-a-warning but still functional on 2.14.1, so
no reason to touch what isn't broken yet (`docs/invisp.md` still
flags them as a future risk).

**Also vendored (added when sourcing training data, not in the initial
pull):** `data/*.txt` (the exact Canon EOS 5D / Nikon D700 image lists
and URLs `canon.pth`/`nikon.pth` were themselves trained+evaluated on),
`data/data_preprocess.py`, `data/data_preprocess.sh`. See
[`data/README.md`](data/README.md) for size estimates, usage, and a real
bug found in `data_preprocess.py` (a camera-name string typo that makes
Canon's black-level subtraction dead code) that was deliberately left
unpatched rather than silently "corrected" — see that file for why.

**Also vendored (added for actual training): `train.py`, `dataset/`
(`FiveK_dataset.py`, `base_dataset.py`), `config/config.py`.** See
[`docs/training.md`](docs/training.md) for the full workflow and three real fixes
applied to get this running at all (not cosmetic): `train.py`'s import
paths patched to match this repo's layout (`model`/`utils` live under
`openraw/third_party/invisp/` here, not top-level as in upstream); `train.py`'s
hard `nvidia-smi` shell-out replaced with an explicit CUDA check plus a
non-fatal fallback instead of a confusing crash on any machine where
`nvidia-smi` isn't on PATH in exactly the form upstream assumed; and
`dataset/FiveK_dataset.py`'s `from scipy.misc import imread` (removed
from scipy years ago, fails outright on any current scipy) replaced with
an equivalent PIL-based read. Two genuinely dead imports in
`FiveK_dataset.py` (`torchvision`, `rawpy` — imported, never referenced)
were deliberately left untouched rather than "cleaned up", consistent
with this project's pattern of patching only what's actually broken.

**What was NOT vendored:** `dataset/` (the PyTorch `Dataset` class
itself — OpenRAW doesn't yet have its own training loop to feed it
into), `config/`, `train.py`, `test_rgb.py`, `test_raw.py`,
`cal_metrics.py` — upstream's FiveK-dataset-specific training/eval
scripts, which pull in extra dependencies (`torchvision`, a now-removed
`scipy.misc.imread`) beyond what this project otherwise needs.
`openraw/invisp_bridge.py` is **our own new code**, not vendored, that
replaces their role for single-image inference — it reuses
`openraw/decode.py` instead of their dataset loader, and was written
by reading their `test_rgb.py` and `test_raw.py` in full to match their
preprocessing (normalization, white-balance handling, the `--gamma`
training flag's effect on what the "RAW" side of the network actually
represents) rather than guessing at it. See that file's docstring for
specifics.

**Pretrained weights:** `pretrained/canon.pth` and `pretrained/nikon.pth`
are upstream's own official checkpoints (verified byte-identical via
md5 against a fresh clone of the upstream repo — not retrained or
modified). Per upstream's own README: each checkpoint is camera-specific
and shouldn't be expected to generalize to other cameras' JPEGs.

InvISP's own README credits the invertible-block design it builds on to:

> Mingqing Xiao, Shuxin Zheng, Chang Liu, Yaolong Wang, Di He, Guolin Ke,
> Jiang Bian, Zhouchen Lin, Tie-Yan Liu.
> **Invertible Image Rescaling.** ECCV 2020.
> <https://github.com/pkuxmq/Invertible-Image-Rescaling>

No source from Invertible-Image-Rescaling was directly vendored here —
only InvISP's own code (which already incorporates/adapts that design)
was pulled in.

## `openraw/ml/` — superseded, NumPy architecture demo (kept for its tests)

Before the real InvISP source and checkpoints were available to this
project, `openraw/ml/` was an independent NumPy re-implementation of
InvISP's general architectural idea (Haar invertible downsampling +
affine coupling blocks) — written from the paper/README description,
not from upstream's code. It is now superseded by `openraw/third_party/invisp/`
+ `openraw/invisp_bridge.py` and should not be extended further; see
its own `__init__.py` docstring for why it's kept around at all (its
exact-invertibility tests remain a legitimate, from-scratch demonstration
of the architecture class).

## Checkpoints NOT part of this project

Some uploaded checkpoints (`FiveK_XYZ.pth`, `FiveK_sRGB.pth`,
`PPR10K_a.pth`, `PPR10K_b.pth`, `PPR10K_c.pth`) are **not InvISP** and
are not included in this repo. Their internal tensor names
(`backbone.model.*`, `gen_3d_lut.weights_generator.*`) identify them as
checkpoints from a different project, **Image-Adaptive-3DLUT** (Zeng et
al., CVPR2020/TPAMI) — a classifier + learned 3D lookup table for photo
retouching, architecturally unrelated to InvISP's invertible network. If
this project ever integrates that approach too, it needs its own
attribution section here, done the same way — not folded into this one.

## Training data (planned, not yet implemented)

When/if training data is added to this project, it should come from
existing paired RAW/JPEG research datasets with their own usage terms
(e.g. MIT-Adobe FiveK, NUS, RAISE — the same datasets InvISP itself
trains on), not scraped web images.
