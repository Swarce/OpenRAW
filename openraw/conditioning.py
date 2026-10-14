"""Per-camera adapters and EXIF conditioning for the InvISP network.

Two optional additions to InvISP, both trained with train.py and both off
unless asked for. They change only the dense sub-networks inside each
coupling block, never the coupling itself, so the network stays exactly
invertible: whatever F, G and H compute, y1 = x1 + F(x2), y2 = x2*exp(s(y1))
+ G(y1) can be undone given the same F, G, H (and the same conditioning).

Per-camera adapters (train.py --cam_lora RANK), after CamLoRA in SpiralDiff
(Yue et al., CVPR 2026, arXiv:2603.14885). One shared network plus, for each
training camera, a small low-rank correction to every convolution:
    conv(x) + B_c(A_c(x)),   A_c: k x k conv to RANK channels, B_c: 1x1 back,
B_c starting at zero, so a freshly attached adapter changes nothing. A photo
from a camera the model was trained on (matched by EXIF Make/Model) gets its
camera's adapter; any other photo runs the shared network alone. Training
drops the adapter for a share of steps (--cam_dropout) so the shared network
stays good on its own for those unknown cameras.

EXIF conditioning (train.py --exif_cond), after ParamNet in ParamISP (Kim et
al., CVPR 2024, arXiv:2312.13313). The photo's exposure time, f-number, ISO
and focal length are embedded (log scale, a few sine/cosine features each),
summed, passed through a small MLP, and modulate the first four convolutions
of every dense sub-network (FiLM: out * (1 + gamma) + beta, starting at the
identity). Like ParamISP, each setting is dropped at random in training
(--param_dropout) so a JPEG missing some EXIF still converts sensibly; one
with no EXIF at all uses the learned "nothing known" state. ParamISP also
feeds the raw's white balance and colour matrix to ParamNet: a JPEG doesn't
have those, so this uses only the four settings a JPEG does carry.

Both are written for OpenRAW from the papers' descriptions; no code from
either project is included (see NOTICE.md).

Checkpoints: the extra weights live under "openraw_cond." in the state dict,
with a plain-dict description under "openraw_meta" (cameras, rank, ...).
load_into() builds the matching network for any checkpoint, with or without.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn

from .camera_params import PARAMS

META_KEY = "openraw_meta"
PREFIX = "openraw_cond."
VERSION = 1

# log2 centre and spread of each setting, so typical values land in about [-1, 1]
_CENTRE = {"exposure_time": math.log2(1 / 100), "f_number": math.log2(5.6), "iso": math.log2(400),
           "focal_length": math.log2(50)}
_SPREAD = {"exposure_time": 4.0, "f_number": 1.5, "iso": 2.5, "focal_length": 1.5}
_FREQS = 4
_FEATS = 1 + 2 * _FREQS


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def match_camera(cameras: list[str], make: str | None, model: str | None) -> int:
    """Index of the training camera folder (e.g. 'NIKON_D700', 'RAISE_NIKON_D90')
    a photo's EXIF Make/Model belongs to, or -1. EXIF models usually repeat the
    make ('NIKON D700', 'Canon EOS 5D'); some don't ('M8' for a Leica M8)."""
    m = _norm(model or "")
    if not m:
        return -1
    mk = _norm((make or "").split()[0]) if make else ""
    best, best_len = -1, 0
    for i, name in enumerate(cameras):
        n = _norm(re.sub(r"^RAISE_", "", name))
        if n in (m, mk + m) or (n.endswith(m) and len(m) >= 2 and n[: len(n) - len(m)] in (mk, "")) \
                or (m.endswith(n) and m[: len(m) - len(n)] in (mk, "")):
            if len(n) > best_len:
                best, best_len = i, len(n)
    return best


class ParamEmbed(nn.Module):
    """ParamNet-style embedding of the four EXIF settings into one vector."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.inp = nn.ModuleList(nn.Linear(_FEATS, hidden) for _ in PARAMS)
        self.mlp = nn.Sequential(nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, hidden))
        self.bias = nn.Parameter(torch.zeros(hidden))

    @staticmethod
    def features(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """values: (N, 4) in PARAMS order, NaN/<=0 = unknown -> (features (N,4,F), known mask (N,4))."""
        known = torch.isfinite(values) & (values > 0)
        v = torch.where(known, values, torch.ones_like(values)).log2()
        c = values.new_tensor([_CENTRE[k] for k in PARAMS])
        s = values.new_tensor([_SPREAD[k] for k in PARAMS])
        v = ((v - c) / s).clamp(-3, 3)
        f = [v] + [fn(v * (2 ** k) * math.pi / 2) for k in range(_FREQS) for fn in (torch.sin, torch.cos)]
        return torch.stack(f, -1), known

    def forward(self, values: torch.Tensor, dropout: float = 0.0) -> torch.Tensor:
        feats, known = self.features(values.float())
        keep = known.float()
        if dropout > 0:
            keep = keep * (torch.rand_like(keep) >= dropout).float()
        e = self.bias.expand(values.shape[0], -1)
        for i, lin in enumerate(self.inp):
            e = e + keep[:, i:i + 1] * lin(feats[:, i])
        return self.mlp(e)


class _LoRA(nn.Module):
    def __init__(self, conv: nn.Conv2d, rank: int):
        super().__init__()
        self.a = nn.Conv2d(conv.in_channels, rank, conv.kernel_size, conv.stride, conv.padding, bias=False)
        self.b = nn.Conv2d(rank, conv.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)  # starts as no change

    def forward(self, x):
        return self.b(self.a(x))


class Conditioning(nn.Module):
    """Attached to an InvISPNet as `net.openraw_cond`; see the module docstring."""

    def __init__(self, net: nn.Module, cameras=(), lora_rank: int = 0, exif: bool = False, hidden: int = 64):
        super().__init__()
        from .third_party.invisp.model.model import DenseBlock
        self.cameras = list(cameras) if lora_rank > 0 else []
        self.lora_rank, self.exif, self.hidden = int(lora_rank), bool(exif), int(hidden)
        dense = [m for m in net.modules() if isinstance(m, DenseBlock)]
        self._convs = [[d.conv1, d.conv2, d.conv3, d.conv4, d.conv5] for d in dense]
        if self.lora_rank > 0 and not self.cameras:
            raise ValueError("per-camera adapters need the list of training cameras")
        self.lora = nn.ModuleList(
            nn.ModuleList(_LoRA(c, self.lora_rank) for convs in self._convs for c in convs) for _ in self.cameras)
        if self.exif:
            self.embed = ParamEmbed(hidden)
            gc = dense[0].conv1.out_channels if dense else 32
            self.film = nn.ModuleList(nn.Linear(hidden, 4 * 2 * gc) for _ in dense)
            for lin in self.film:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        self.cam_ids = None   # (N,) long, -1 = shared network only
        self.hyper = None     # (N, hidden)
        self._films = None    # per dense block: (N, 4*2*gc), computed once per context
        for d, convs in enumerate(self._convs):
            for j, conv in enumerate(convs):
                conv.register_forward_hook(self._hook(d, j))
        # a context holds values computed from the weights: loading new ones drops it
        self.register_load_state_dict_post_hook(lambda module, keys: module.clear())

    def meta(self) -> dict:
        return {"version": VERSION, "cameras": self.cameras, "lora_rank": self.lora_rank, "exif": self.exif,
                "hidden": self.hidden, "params": list(PARAMS)}

    # ---- context: what the next forward/inverse passes condition on
    def set_context(self, cam_ids=None, params=None, param_dropout: float = 0.0):
        """cam_ids: (N,) camera indices (-1 = none) or None; params: (N, 4) EXIF
        values (NaN = unknown) or None. Kept until the next call -- the forward
        and inverse passes of one training step (and gradient checkpointing's
        recomputation in backward) must see the same context."""
        dev = next(self.parameters()).device if any(True for _ in self.parameters()) else None
        self.cam_ids = None
        if self.lora_rank > 0 and cam_ids is not None:
            ids = torch.as_tensor(cam_ids, dtype=torch.long, device=dev).reshape(-1)
            self.cam_ids = ids if bool((ids >= 0).any()) else None
        self.hyper = None
        if self.exif:
            if params is None:
                params = torch.full((1, len(PARAMS)), float("nan"))
            p = torch.as_tensor(params, dtype=torch.float32, device=dev).reshape(-1, len(PARAMS))
            self.hyper = self.embed(p, param_dropout)
        # computed here, once, outside the network's (possibly checkpointed) blocks
        self._films = [lin(self.hyper) for lin in self.film] if self.hyper is not None else None

    def clear(self):
        self.cam_ids, self.hyper, self._films = None, None, None

    def _hook(self, d, j):
        k = d * 5 + j

        def hook(mod, inputs, out):
            x = inputs[0]
            n = out.shape[0]
            if self.cam_ids is not None:
                ids = self.cam_ids
                if ids.numel() == 1 and n > 1:
                    ids = ids.expand(n)
                present = [int(c) for c in torch.unique(ids).tolist() if c >= 0]
                if len(present) == 1 and bool((ids == present[0]).all()):
                    out = out + self.lora[present[0]][k](x).to(out.dtype)
                else:
                    for c in present:
                        idx = (ids == c).nonzero(as_tuple=True)[0]
                        out = out.index_add(0, idx, self.lora[c][k](x[idx]).to(out.dtype))
            if self.exif and j < 4:
                if self._films is None:  # no context set: settings unknown
                    self.set_context(self.cam_ids)
                gb = self._films[d]
                gc = out.shape[1]
                gb = gb.view(gb.shape[0], 4, 2, gc)[:, j]
                if gb.shape[0] == 1 and n > 1:
                    gb = gb.expand(n, -1, -1)
                g, b = gb[:, 0, :, None, None].to(out.dtype), gb[:, 1, :, None, None].to(out.dtype)
                out = out * (1 + g) + b
            return out
        return hook


def attach(net: nn.Module, cameras=(), lora_rank: int = 0, exif: bool = False, hidden: int = 64) -> Conditioning | None:
    """Add per-camera adapters and/or EXIF conditioning to an InvISPNet (in
    place). Returns the Conditioning module (also net.openraw_cond), or None
    when neither is asked for."""
    if lora_rank <= 0 and not exif:
        return None
    cond = Conditioning(net, cameras, lora_rank, exif, hidden)
    net.openraw_cond = cond
    try:
        cond.to(next(p for n, p in net.named_parameters() if not n.startswith(PREFIX)).device)
    except StopIteration:
        pass
    return cond


def attach_from_meta(net: nn.Module, meta: dict | None) -> Conditioning | None:
    if not meta:
        return None
    if int(meta.get("version", 1)) > VERSION:
        raise ValueError(f"checkpoint made by a newer OpenRAW (conditioning v{meta['version']})")
    return attach(net, meta.get("cameras", ()), int(meta.get("lora_rank", 0)), bool(meta.get("exif")),
                  int(meta.get("hidden", 64)))


def get(net: nn.Module) -> Conditioning | None:
    return getattr(net, "openraw_cond", None)


def split_meta(state_dict: dict) -> tuple[dict, dict | None]:
    """(state dict without the description, the description or None)."""
    if isinstance(state_dict, dict) and META_KEY in state_dict:
        sd = {k: v for k, v in state_dict.items() if k != META_KEY}
        return sd, state_dict[META_KEY]
    return state_dict, None


def with_meta(state_dict: dict, net: nn.Module) -> dict:
    """The state dict to save: plus the description when the net has conditioning."""
    c = get(net)
    return {**state_dict, META_KEY: c.meta()} if c is not None else state_dict


def set_context(net: nn.Module, cam_ids=None, params=None, param_dropout: float = 0.0) -> None:
    c = get(net)
    if c is not None:
        c.set_context(cam_ids, params, param_dropout)


def remap_cameras(state_dict: dict, old: list[str], new: list[str]) -> dict:
    """Re-index per-camera adapter weights saved for camera list `old` to the
    order of `new`: a camera in both keeps its adapter, a camera only in `new`
    gets none from the file (starts as 'no change'), one only in `old` is dropped."""
    out, pre = {}, PREFIX + "lora."
    for k, v in state_dict.items():
        if not k.startswith(pre):
            out[k] = v
            continue
        i, rest = k[len(pre):].split(".", 1)
        i = int(i)
        if i < len(old) and old[i] in new:
            out[f"{pre}{new.index(old[i])}.{rest}"] = v
    return out


def conditioned_parameters(net: nn.Module):
    return [p for n, p in net.named_parameters() if n.startswith(PREFIX)]


def describe(meta: dict | None) -> str:
    if not meta:
        return "none"
    parts = []
    if meta.get("lora_rank"):
        parts.append(f"per-camera adapters (rank {meta['lora_rank']}) for {', '.join(meta.get('cameras', []))}")
    if meta.get("exif"):
        parts.append("EXIF conditioning (" + ", ".join(meta.get("params", PARAMS)) + ")")
    return "; ".join(parts) or "none"


__all__ = ["attach", "attach_from_meta", "get", "split_meta", "with_meta", "set_context", "match_camera",
           "conditioned_parameters", "describe", "remap_cameras", "Conditioning", "ParamEmbed", "META_KEY", "PREFIX"]
