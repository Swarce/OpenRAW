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
(`FiveK_dataset.py`, `base_dataset.py`), `config/config.py`.** These started
as upstream's files with only what was broken patched (import paths for this
repo's layout; `scipy.misc.imread`, removed from scipy years ago; an
`nvidia-smi | grep` GPU picker that fails on Windows) and have since been
substantially extended for OpenRAW: any FiveK camera and multi-camera
pooling, per-file black/white levels and CFA patterns, compact mosaic
storage, parallel loading, exact resume, held-out evaluation, gradient
checkpointing, time-limited sessions and multi-GPU training. Every change is
marked inline with `PATCHED (OpenRAW, not upstream)`; the reasons are in
[`docs/training.md`](docs/training.md). Two dead imports in
`FiveK_dataset.py` (`torchvision`, `rawpy` — imported, never referenced)
were deliberately left in place. `data/fivek_download.py`,
`data/fivek_prepare.py`, `dataset/mosaic_store.py` and `kaggle/` are
OpenRAW's own code, not vendored.

**What was NOT vendored:** `test_rgb.py`, `test_raw.py` and
`cal_metrics.py` — upstream's evaluation scripts. `openraw/invisp_bridge.py`
is **our own new code** that replaces their role for single-image
inference — it reuses `openraw/decode.py` instead of their dataset loader,
and was written by reading their `test_rgb.py` and `test_raw.py` in full to
match their preprocessing (normalization, white-balance handling, the
`--gamma` training flag's effect on what the "RAW" side of the network
actually represents) rather than guessing at it. See that file's docstring
for specifics.

**Pretrained weights:** `pretrained/canon.pth` and `pretrained/nikon.pth`
are upstream's own official checkpoints (verified byte-identical via
md5 against a fresh clone of the upstream repo — not retrained or
modified). Per upstream's own README: each checkpoint is camera-specific
and shouldn't be expected to generalize to other cameras' JPEGs.

`pretrained/openraw-fivek-e35-best.pth` and
`pretrained/openraw-fivek-e46-latest.pth` are **not** upstream's: they were
trained by OpenRAW with this repository's `train.py` (InvISP's architecture,
vendored MIT code) on images from the MIT-Adobe FiveK dataset
(Bychkovsky, Paris, Chan & Durand, CVPR 2011). FiveK's images are under
its own license terms, which should be checked before these weights are
distributed publicly.

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

## Training data

Training data is not included in this repository; the scripts download it
from each dataset's own server, under that dataset's own terms:

- **MIT-Adobe FiveK** (Bychkovsky, Paris, Chan & Durand, CVPR 2011) —
  `data/fivek_download.py`, `data/fivek_prepare.py`.
- **RAISE** (Dang-Nguyen, Pasquini, Conotter & Boato, "RAISE: A Raw Images
  Dataset for Digital Image Forensics", ACM MMSys 2015) —
  `data/raise_prepare.py`. RAISE's terms: *"to be used for non-commercial
  research and educational purposes"*, and published work using it should
  cite the paper. The list of download links comes from
  <https://loki.disi.unitn.it/RAISE/>, where the terms are accepted. A model
  trained on RAISE images falls under those terms; none of the weights in
  `pretrained/` have been trained on RAISE so far.

## Logo font: Saira (SIL Open Font License 1.1)

The wordmark in `assets/logo/` is set in **Saira** (Copyright 2020 The Saira
Project Authors, https://github.com/Omnibus-Type/Saira), converted to
outlines by `tools/logo/make_logo.py` (the font itself is unmodified). The
font file and its license are included at `tools/logo/fonts/`
(`OFL-Saira.txt`). The aperture artwork is original, constructed
geometrically by the same script.
