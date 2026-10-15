"""Extra training objectives for train.py (all optional, all off by default).

Edit-aware loss (train.py --edit_loss W), after "Edit-aware RAW
Reconstruction" (Punnappurath, Zhao, Zhao, Nguyen, Grzeszczuk, Brown; Samsung
AI Center Toronto; CVPR 2026, arXiv:2512.05859). A pixel loss on the raw says
nothing about how the raw holds up once someone edits it -- which is the
whole point of a DNG. So both the predicted and the true raw go through the
same small differentiable "photo editor" with randomly drawn settings, and
the two edited sRGB renders are compared:
    exposure     x 2^e,             e ~ N(0, sigma^2)      (paper: sigma 0.75)
    white balance  red/blue gains   log-gain ~ N(0, 0.15^2), or with
                                    --edit_wb real: the change between two
                                    nearby real illuminants (RawGen's set)
    colour       saturation         ~ U(0.75, 1.3)
    tone         contrast S-curve   blend ~ U(-0.3, 0.6), then gamma 1/2.2
Differences from the paper: its white balance samples real AsShotNeutral
values near the image's own (--edit_wb real is the closest stand-in) and
applies a camera-to-XYZ matrix, and its colour and tone stages
are small MLPs fitted to 3D LUTs and Adobe's tone curve. OpenRAW's training
raws are already white-balanced camera RGB with no colour matrix stored, so
this uses simple parametric stand-ins for each stage; the principle -- same
random edit on both, compare the renders -- is the paper's. L1, like
train.py's other losses (the paper uses L2).

Photo-finishing augmentation (train.py --finish_aug W), after RawGen (Kim,
Lee, Punnappurath, Afifi, Han, Levinshtein, Brown; ECCV 2026,
arXiv:2604.00093), which trains on several differently finished sRGB
versions of each photo mapped to ONE target. Here: on a share of steps
(--finish_prob) the real training JPEG gets a random global finish, goes
through the inverse network, and must still give the original raw. Two
finishes: --finish_style rawgen uses RawGen's own photo-finishing (tone
mapping, contrast, optionally its white-balance multipliers with
--finish_wb; adapted in openraw/third_party/rawgen/finishing.py, CC BY-NC-SA
4.0); --finish_style simple is OpenRAW's earlier stand-in (gamma, contrast
S-curve, saturation). Camera JPEGs are finished differently from the
LibRaw renders training uses; this teaches the inverse not to put the
finishing into the raw. Exposure changes are never used and white balance
only with --finish_wb, since both change what the raw should be.
Caveat: InvISP is a bijection, so it can't map two different JPEGs to exactly
the same raw; this pulls the inverse toward the average over finishes.
Measure it against a run without (tools/compare_models.py) before relying on
it.

Log-domain L1 (train.py --logl1_loss W), from SpiralDiff (Yue et al., CVPR
2026, arXiv:2603.14885; code in openraw/third_party/spiraldiff/, MIT): the
predicted and true raw compared as log(x + 1/255), so errors in the shadows
count as much as in the highlights.
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


_ILLUMINANTS = None


def illuminants() -> torch.Tensor:
    """(K, 2) log(R/G), log(B/G) of real scene illuminants, from RawGen's
    illumination set (openraw/third_party/rawgen/illuminants.csv, CC BY-NC-SA 4.0)."""
    global _ILLUMINANTS
    if _ILLUMINANTS is None:
        from pathlib import Path
        import numpy as np
        f = Path(__file__).resolve().parent / "third_party" / "rawgen" / "illuminants.csv"
        _ILLUMINANTS = torch.from_numpy(np.loadtxt(f, delimiter=",", comments="#", skiprows=3,
                                                   usecols=(0, 1), dtype=np.float32))
    return _ILLUMINANTS


def real_wb_shifts(n: int, device, neighbours: int = 200, generator=None) -> torch.Tensor:
    """(n, 2) log white-balance gain changes (red, blue) a person might make:
    from one real illuminant to another nearby one (among the `neighbours`
    closest in chromaticity), the way the edit-aware loss paper samples white
    balances near the image's own. Illuminants from RawGen's set."""
    ill = illuminants()
    i = torch.randint(len(ill), (n,), generator=generator)
    d = torch.cdist(ill[i], ill)                                   # (n, K)
    near = d.topk(min(neighbours, len(ill)), largest=False).indices  # (n, k)
    j = near[torch.arange(n), torch.randint(near.shape[1], (n,), generator=generator)]
    # correcting for illuminant L means gains 1/L: the change from i's correction to j's
    return (ill[i] - ill[j]).to(device)


def sample_edits(n: int, device, sigma: float = 0.75, generator=None, wb: str = "gaussian") -> dict:
    """Random edit settings for a batch of n images (one setting per image).
    wb: "gaussian" (log-gains ~ N(0, 0.15^2)) or "real" (between nearby real
    illuminants, real_wb_shifts)."""
    def rn(*shape):
        return torch.randn(*shape, device=device, generator=generator)

    def ru(lo, hi):
        return lo + (hi - lo) * torch.rand(n, device=device, generator=generator)

    shifts = real_wb_shifts(n, device) if wb == "real" else rn(n, 2) * 0.15
    return {"exposure": rn(n) * sigma, "wb": shifts, "saturation": ru(0.75, 1.3),
            "contrast": ru(-0.3, 0.6)}


def render_edit(raw_linear: torch.Tensor, e: dict) -> torch.Tensor:
    """White-balanced linear raw (N,3,H,W) -> edited display sRGB in [0, 1]."""
    x = raw_linear.float().clamp(min=0) * (2.0 ** e["exposure"]).view(-1, 1, 1, 1)
    g = torch.exp(e["wb"])  # red and blue gains, green fixed
    gains = torch.stack([g[:, 0], torch.ones_like(g[:, 0]), g[:, 1]], 1).view(-1, 3, 1, 1)
    x = _saturate(x * gains, e["saturation"]).clamp(0, 1)
    return _contrast(_gamma(x), e["contrast"])


def edit_aware_loss(pred_raw: torch.Tensor, true_raw: torch.Tensor, raw_gamma: bool,
                    sigma: float = 0.75, edits: dict | None = None, wb: str = "gaussian") -> torch.Tensor:
    """L1 between the predicted and true raw after the same random edit.
    raw_gamma: the raws are gamma-encoded (train.py --gamma, as the Kaggle runs
    use): linearised first, as the editor works on linear data."""
    if edits is None:
        edits = sample_edits(true_raw.shape[0], true_raw.device, sigma, wb=wb)
    if raw_gamma:
        pred_raw = pred_raw.float().clamp(min=0) ** 2.2
        true_raw = true_raw.float().clamp(min=0) ** 2.2
    return F.l1_loss(render_edit(pred_raw, edits), render_edit(true_raw, edits))


def finish(rgb: torch.Tensor, style: str = "simple", white_balance: bool = False) -> torch.Tensor:
    """A random photo finish on a batch of display sRGB images.
    style "rawgen": RawGen's finishing (openraw/third_party/rawgen/finishing.py:
    tone mapping, contrast, optional white balance); "simple": the parametric
    stand-in below (gamma, contrast S-curve, saturation)."""
    if style == "rawgen":
        from .third_party.rawgen import finishing
        return finishing.apply(rgb, finishing.sample_params(rgb.shape[0], rgb.device, white_balance=white_balance))
    if style != "simple":
        raise ValueError(f"unknown finish style {style!r}")
    return apply_finish(rgb, sample_finish(rgb.shape[0], rgb.device))


def logl1(pred_raw: torch.Tensor, true_raw: torch.Tensor, raw_gamma: bool) -> torch.Tensor:
    """SpiralDiff's log-domain L1 (openraw/third_party/spiraldiff/losses.py) on linear raw."""
    from .third_party.spiraldiff.losses import logl1_loss
    if raw_gamma:
        pred_raw = pred_raw.float().clamp(min=0) ** 2.2
        true_raw = true_raw.float().clamp(min=0) ** 2.2
    return logl1_loss(pred_raw.float(), true_raw.float())


def sample_finish(n: int, device, generator=None) -> dict:
    def ru(lo, hi):
        return lo + (hi - lo) * torch.rand(n, device=device, generator=generator)
    return {"gamma": torch.exp(ru(-0.25, 0.25)), "contrast": ru(-0.25, 0.4), "saturation": ru(0.8, 1.25)}


def apply_finish(rgb: torch.Tensor, f: dict) -> torch.Tensor:
    """A random global photo finish on display sRGB (N,3,H,W) in [0, 1]."""
    x = rgb.float().clamp(0, 1) ** f["gamma"].view(-1, 1, 1, 1)
    x = _contrast(x, f["contrast"])
    return _saturate(x, f["saturation"]).clamp(0, 1)
