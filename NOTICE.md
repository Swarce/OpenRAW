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

`pretrained/openraw-fivek-e35-best.pth`,
`pretrained/openraw-fivek-e46-latest.pth`,
`pretrained/openraw-fivek-e53-best.pth` and
`pretrained/openraw-fivek-raise-e14-best.pth` (FiveK and RAISE, see below)
are **not** upstream's: they were
trained by OpenRAW with this repository's `train.py` (InvISP's architecture,
vendored MIT code) on images from the MIT-Adobe FiveK dataset
(Bychkovsky, Paris, Chan & Durand, CVPR 2011), whose images are licensed
for research use only. These weights are therefore offered for
**non-commercial research use**, unlike the MIT-licensed code; see
"Training data" below.

InvISP's own README credits the invertible-block design it builds on to:

> Mingqing Xiao, Shuxin Zheng, Chang Liu, Yaolong Wang, Di He, Guolin Ke,
> Jiang Bian, Zhouchen Lin, Tie-Yan Liu.
> **Invertible Image Rescaling.** ECCV 2020.
> <https://github.com/pkuxmq/Invertible-Image-Rescaling>

No source from Invertible-Image-Rescaling was directly vendored here —
only InvISP's own code (which already incorporates/adapts that design)
was pulled in.

## Ideas from other papers: implemented, not vendored

The optional training additions in `openraw/conditioning.py`,
`openraw/train_losses.py`, `openraw/camera_params.py` and
`data/camera_params.py` (train.py's `--edit_loss`, `--finish_aug`,
`--cam_lora`, `--exif_cond`; see [`docs/training.md`](docs/training.md)) were
written for OpenRAW from the papers' text. **No source code, weights or data
from these projects is included.** Several of their licences wouldn't allow it
in an MIT project: RawGen's is non-commercial and ParamISP's repository has no
licence. They are credited here because the ideas are theirs. If you use
these features in research, please cite the original papers. All BibTeX
entries are collected in [`docs/references.bib`](docs/references.bib).

**Edit-aware loss** (`--edit_loss`), from:

> Abhijith Punnappurath, Luxi Zhao, Ke Zhao, Hue Nguyen, Radek Grzeszczuk,
> Michael S. Brown (AI Center – Toronto, Samsung Electronics).
> **Edit-aware RAW Reconstruction.** CVPR 2026.
> Paper: <https://arxiv.org/abs/2512.05859>
> No public code found (checked 2026-10-14).

```bibtex
@inproceedings{punnappurath2026editaware,
  title     = {Edit-aware {RAW} Reconstruction},
  author    = {Punnappurath, Abhijith and Zhao, Luxi and Zhao, Ke and Nguyen, Hue and
               Grzeszczuk, Radek and Brown, Michael S.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
(No BibTeX is published with the paper; this entry was assembled from the
CVPR 2026 listing.)

**Photo-finishing augmentation** (`--finish_aug`), after RawGen's training on
many finished variants per target, from:

> Dongyoung Kim, Junyong Lee, Abhijith Punnappurath, Mahmoud Afifi, Sangmin
> Han, Alex Levinshtein, Michael S. Brown (AI Center – Toronto, Samsung
> Electronics; Yonsei University).
> **RawGen: Learning Camera Raw Image Generation.** ECCV 2026.
> Paper: <https://arxiv.org/abs/2604.00093> · Code: <https://github.com/DY112/RawGen>
> Licence: code CC BY-NC-SA 4.0; weights under the FLUX.1 [dev] Non-Commercial License.

```bibtex
@inproceedings{kim2026rawgen,
  title     = {RawGen: Learning Camera Raw Image Generation},
  author    = {Dongyoung Kim and Junyong Lee and Abhijith Punnappurath and
               Mahmoud Afifi and Sangmin Han and Alex Levinshtein and
               Michael S. Brown},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
(BibTeX as given in RawGen's README.)

**Per-camera adapters** (`--cam_lora`), after CamLoRA, from:

> Huanjing Yue, Shangbin Xie, Cong Cao, Qian Wu, Lei Zhang, Lei Zhao,
> Jingyu Yang.
> **SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras.** CVPR 2026.
> Paper: <https://arxiv.org/abs/2603.14885> · Code: <https://github.com/Chuancy-TJU/SpiralDiff>
> Licence: Apache-2.0.

```bibtex
@inproceedings{yue2026spiraldiff,
  title     = {SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras},
  author    = {Yue, Huanjing and Xie, Shangbin and Cao, Cong and Wu, Qian and Zhang, Lei and Zhao, Lei and Yang, Jingyu},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```
(BibTeX as given in SpiralDiff's README.)

**EXIF conditioning** (`--exif_cond`), after ParamNet, from:

> Woohyeok Kim, Geonu Kim, Junyong Lee, Seungyong Lee, Seung-Hwan Baek,
> Sunghyun Cho.
> **ParamISP: Learned Forward and Inverse ISPs using Camera Parameters.** CVPR 2024.
> Paper: <https://arxiv.org/abs/2312.13313> · Code: <https://github.com/woo525/ParamISP>
> Licence: none stated in the repository (checked 2026-10-14).

```bibtex
@inproceedings{kim2024paramisp,
  title     = {ParamISP: Learned Forward and Inverse ISPs using Camera Parameters},
  author    = {Kim, Woohyeok and Kim, Geonu and Lee, Junyong and Lee, Seungyong and Baek, Seung-Hwan and Cho, Sunghyun},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {26067--26076},
  year      = {2024}
}
```
(BibTeX as given in ParamISP's README.)

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

No training images are included in this repository. The scripts download
them from each dataset's own server, under that dataset's own terms. If you
publish results from a model trained with this project, cite the datasets it
was trained on.

### MIT-Adobe FiveK

Used by `data/fivek_download.py`, `data/fivek_prepare.py` and the Kaggle
notebook; `openraw-fivek-*.pth` were trained on it, and so were upstream's
`canon.pth` / `nikon.pth` (its Canon EOS 5D and Nikon D700 subsets).

> Vladimir Bychkovsky, Sylvain Paris, Eric Chan, Frédo Durand.
> **Learning Photographic Global Tonal Adjustment with a Database of
> Input / Output Image Pairs.** CVPR 2011.
> <https://data.csail.mit.edu/graphics/fivek/>

```bibtex
@inproceedings{fivek,
  author    = {Vladimir Bychkovsky and Sylvain Paris and Eric Chan and Fr{\'e}do Durand},
  title     = {Learning Photographic Global Tonal Adjustment with a Database of Input / Output Image Pairs},
  booktitle = {The Twenty-Fourth IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2011}
}
```

**Terms.** FiveK's images are under two licenses, each covering the files
listed with it on the dataset page: [LicenseAdobe](https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobe.txt)
(© 2011 Adobe Systems Incorporated) and
[LicenseAdobeMIT](https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobeMIT.txt)
(© 2011 Adobe Systems Incorporated and Massachusetts Institute of
Technology). Both allow using, copying, modifying and distributing the
images **solely for research purposes**, not for commercial advantage or
monetary compensation; require the copyright notice and license to go with
any copy of the images; and forbid using Adobe's, MIT's or their
contributors' names to endorse or promote derived products. Naming FiveK
here is attribution of the training data, not an endorsement by Adobe or
MIT. This repository contains no FiveK images (the split lists in `data/`
hold image IDs only).

### RAISE

Used by `data/raise_prepare.py`; `pretrained/openraw-fivek-raise-*.pth` were
trained on it (together with FiveK).

> Duc-Tien Dang-Nguyen, Cecilia Pasquini, Valentina Conotter, Giulia Boato.
> **RAISE: A Raw Images Dataset for Digital Image Forensics.** ACM
> Multimedia Systems (MMSys), Portland, Oregon, 2015.
> <https://loki.disi.unitn.it/RAISE/>

```bibtex
@inproceedings{raise,
  author    = {Duc-Tien Dang-Nguyen and Cecilia Pasquini and Valentina Conotter and Giulia Boato},
  title     = {{RAISE}: A Raw Images Dataset for Digital Image Forensics},
  booktitle = {Proceedings of the 6th ACM Multimedia Systems Conference (MMSys)},
  address   = {Portland, Oregon},
  year      = {2015}
}
```

**Terms.** *"The RAISE dataset is to be used for non-commercial research and
educational purposes. If using the dataset in any published work, please
cite its related publication."* The image list (CSV) is obtained from the
RAISE download page, where the terms are accepted; it isn't included here.

### What this means for the weights

The code in this repository is MIT licensed. The trained weights are a
different matter: they're learned from images licensed for research only.
OpenRAW therefore offers every checkpoint in `pretrained/` — upstream's and
its own — for **non-commercial research use**, and a model you train on
FiveK or RAISE with these scripts carries the same expectation.

## Logo font: Saira (SIL Open Font License 1.1)

The wordmark in `assets/logo/` is set in **Saira** (Copyright 2020 The Saira
Project Authors, https://github.com/Omnibus-Type/Saira), converted to
outlines by `tools/logo/make_logo.py` (the font itself is unmodified). The
font file and its license are included at `tools/logo/fonts/`
(`OFL-Saira.txt`). The aperture artwork is original, constructed
geometrically by the same script.
