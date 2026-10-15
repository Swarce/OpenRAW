# Adapted from SpiralDiff

> Huanjing Yue, Shangbin Xie, Cong Cao, Qian Wu, Lei Zhang, Lei Zhao, Jingyu Yang.
> **SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras.** CVPR 2026.
> Paper: <https://arxiv.org/abs/2603.14885> · Code: <https://github.com/Chuancy-TJU/SpiralDiff>
> (commit `3081993`) · Licence: MIT ([LICENSE](LICENSE)). The repository's README badge says
> Apache-2.0; its LICENSE file is MIT, and that file is what this follows.

| File here | From SpiralDiff | What OpenRAW changed |
|---|---|---|
| `losses.py` | `logl1_loss` in `models/gaussian_diffusion.py` (its `eps` follows HDR-CNN, Eilertsen et al., ACM TOG 2017) | Inputs already in [0, 1]; mean over the batch; negative predictions clamped before the log. |

Also from SpiralDiff, as a convention rather than copied code: OpenRAW's
per-camera adapters (`openraw/conditioning.py`, `train.py --cam_lora`) follow
its CamLoRA, including the `alpha / rank` scaling of its `LoRAFFN`
(`--cam_lora_alpha`).

If you use this in research, please cite SpiralDiff:

```bibtex
@inproceedings{yue2026spiraldiff,
  title     = {SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras},
  author    = {Yue, Huanjing and Xie, Shangbin and Cao, Cong and Wu, Qian and Zhang, Lei and Zhao, Lei and Yang, Jingyu},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```
