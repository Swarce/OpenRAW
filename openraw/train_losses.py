"""Extra training objectives for train.py (all optional, all off by default).

Edit-aware loss (train.py --edit_loss W), after "Edit-aware RAW
Reconstruction" (Punnappurath, Zhao, Zhao, Nguyen, Grzeszczuk, Brown; Samsung
AI Center Toronto; CVPR 2026, arXiv:2512.05859). A pixel loss on the raw says
nothing about how the raw holds up once someone edits it -- which is the
whole point of a DNG. So both the predicted and the true raw go through the
same small differentiable "photo editor" with randomly drawn settings, and
the two edited sRGB renders are compared:
    exposure     x 2^e,             e ~ N(0, sigma^2)      (paper: sigma 0.75)
    white balance  red/blue gains   log-gain ~ N(0, 0.15^2)
    colour       saturation         ~ U(0.75, 1.3)
    tone         contrast S-curve   blend ~ U(-0.3, 0.6), then gamma 1/2.2
Differences from the paper: its white balance samples real AsShotNeutral
values and applies a camera-to-XYZ matrix, and its colour and tone stages
are small MLPs fitted to 3D LUTs and Adobe's tone curve. OpenRAW's training
raws are already white-balanced camera RGB with no colour matrix stored, so
this uses simple parametric stand-ins for each stage; the principle -- same
random edit on both, compare the renders -- is the paper's. L1, like
train.py's other losses (the paper uses L2).

Photo-finishing augmentation (train.py --finish_aug W), after RawGen (Kim,
Lee, Punnappurath, Afifi, Han, Levinshtein, Brown; ECCV 2026,
arXiv:2604.00093), which trains on several differently finished sRGB
versions of each photo mapped to ONE target. Here: on a share of steps
(--finish_prob) the real training JPEG gets a random global finish (tone
curve, contrast, saturation), goes through the inverse network, and must
still give the original raw. Camera JPEGs are finished differently from the
LibRaw renders training uses; this teaches the inverse not to put the
finishing into the raw. Only global tone/colour finishes are used -- exposure
and white balance change what the raw should be, so they're left out.
Caveat: InvISP is a bijection, so it can't map two different JPEGs to exactly
the same raw; this pulls the inverse toward the average over finishes.
Measure it against a run without (tools/compare_models.py) before relying on
it. Written for OpenRAW; RawGen's code (CC BY-NC-SA) is not used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_LUMA = (0.2126, 0.7152, 0.0722)


def _luma(x):
    w = x.new_tensor(_LUMA).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def _saturate(x, s):
    y = _luma(x)
    return y + (x - y) * s.view(-1, 1, 1, 1)


def _contrast(x, a):
    """Blend with smoothstep: a > 0 adds an S-curve, a < 0 flattens. Monotone for -2 < a < 1."""
    x = x.clamp(0, 1)
    return x + a.view(-1, 1, 1, 1) * (x * x * (3 - 2 * x) - x)


def _gamma(x, eps=1e-4):
    # x^(1/2.2) has an infinite slope at 0: start the curve at eps instead
    return (x.clamp(0, 1) + eps) ** (1 / 2.2) - eps ** (1 / 2.2)


def sample_edits(n: int, device, sigma: float = 0.75, generator=None) -> dict:
    """Random edit settings for a batch of n images (one setting per image)."""
    def rn(*shape):
        return torch.randn(*shape, device=device, generator=generator)

    def ru(lo, hi):
        return lo + (hi - lo) * torch.rand(n, device=device, generator=generator)

    return {"exposure": rn(n) * sigma, "wb": rn(n, 2) * 0.15, "saturation": ru(0.75, 1.3),
            "contrast": ru(-0.3, 0.6)}


def render_edit(raw_linear: torch.Tensor, e: dict) -> torch.Tensor:
    """White-balanced linear raw (N,3,H,W) -> edited display sRGB in [0, 1]."""
    x = raw_linear.float().clamp(min=0) * (2.0 ** e["exposure"]).view(-1, 1, 1, 1)
    g = torch.exp(e["wb"])  # red and blue gains, green fixed
    gains = torch.stack([g[:, 0], torch.ones_like(g[:, 0]), g[:, 1]], 1).view(-1, 3, 1, 1)
    x = _saturate(x * gains, e["saturation"]).clamp(0, 1)
    return _contrast(_gamma(x), e["contrast"])


def edit_aware_loss(pred_raw: torch.Tensor, true_raw: torch.Tensor, raw_gamma: bool,
                    sigma: float = 0.75, edits: dict | None = None) -> torch.Tensor:
    """L1 between the predicted and true raw after the same random edit.
    raw_gamma: the raws are gamma-encoded (train.py --gamma, as the Kaggle runs
    use): linearised first, as the editor works on linear data."""
    if edits is None:
        edits = sample_edits(true_raw.shape[0], true_raw.device, sigma)
    if raw_gamma:
        pred_raw = pred_raw.float().clamp(min=0) ** 2.2
        true_raw = true_raw.float().clamp(min=0) ** 2.2
    return F.l1_loss(render_edit(pred_raw, edits), render_edit(true_raw, edits))


def sample_finish(n: int, device, generator=None) -> dict:
    def ru(lo, hi):
        return lo + (hi - lo) * torch.rand(n, device=device, generator=generator)
    return {"gamma": torch.exp(ru(-0.25, 0.25)), "contrast": ru(-0.25, 0.4), "saturation": ru(0.8, 1.25)}


def apply_finish(rgb: torch.Tensor, f: dict) -> torch.Tensor:
    """A random global photo finish on display sRGB (N,3,H,W) in [0, 1]."""
    x = rgb.float().clamp(0, 1) ** f["gamma"].view(-1, 1, 1, 1)
    x = _contrast(x, f["contrast"])
    return _saturate(x, f["saturation"]).clamp(0, 1)
