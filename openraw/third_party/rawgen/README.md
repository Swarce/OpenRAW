# Adapted from RawGen

> Dongyoung Kim, Junyong Lee, Abhijith Punnappurath, Mahmoud Afifi, Sangmin Han,
> Alex Levinshtein, Michael S. Brown (AI Center – Toronto, Samsung Electronics;
> Yonsei University). **RawGen: Learning Camera Raw Image Generation.** ECCV 2026.
> Paper: <https://arxiv.org/abs/2604.00093> · Project page: <https://dy112.github.io/rawgen-page/>
> Code: <https://github.com/DY112/RawGen> (commit `39f959b`) · Licence: CC BY-NC-SA 4.0 ([LICENSE](LICENSE))

RawGen maps sRGB images, finished in many different ways, back to one
camera-independent target. OpenRAW uses two pieces of it:

| File here | From RawGen | What OpenRAW changed |
|---|---|---|
| `finishing.py` | `training/generate_variants_and_anchors.py`: the random photo-finishing parameters (`ISPAugmenter._generate_random_isp_params`) and the finishing applied after the ISP (`_apply_real_isp_pipeline`: tone mapping, contrast, brightness, white-balance multipliers) | Ported from NumPy to PyTorch and batched. Applied to OpenRAW's already rendered training JPEG instead of running RawGen's mini ISP from the DNG. White-balance multipliers applied in linear sRGB and off by default. Parameter ranges and formulas unchanged. |
| `illuminants.csv` | `samples/illuminations.json`: 8,892 real scene illuminants, which RawGen aggregated from the NUS 8-camera dataset and other public colour-constancy datasets | Converted from XYZ to log(R/G), log(B/G) in linear sRGB, rounded to 4 decimals; 71 entries that fall outside the sRGB gamut dropped (8,821 kept). |

Used by `train.py --finish_aug --finish_style rawgen` (finishing) and
`--edit_loss --edit_wb real` (the edit-aware loss's white-balance edits are
drawn between nearby real illuminants). See `docs/training.md`.

The illuminant data comes from datasets with their own terms; the NUS
dataset is Cheng, Prasad and Brown, "Illuminant estimation for color
constancy: why spatial-domain methods work and the role of the color
distribution", JOSA A 31(5), 2014
(<https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html>).

If you use this in research, please cite RawGen:

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
