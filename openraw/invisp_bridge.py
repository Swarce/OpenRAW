"""
invisp_bridge.py — OUR OWN integration code (not vendored) wiring the
actual InvISP network (openraw/third_party/invisp/, vendored upstream source)
into openraw, using the real pretrained checkpoints in pretrained/.

IMPORTANT, stated plainly: this module requires PyTorch, which is not
installable in the sandbox this project was built in (no disk headroom --
see NOTICE.md). It is written directly against upstream's own
test_rgb.py / test_raw.py / dataset/FiveK_dataset.py logic (read in full
to get the preprocessing exactly right -- see the citations below), but
it has NOT been executed anywhere. Treat it as a carefully-reviewed but
unrun port until it's been run once on a real machine and that's reflected
here. If something doesn't match upstream's behavior, upstream is right
and this file has the bug.

What this reproduces from upstream, stated because it's unintuitive and
easy to get wrong silently:

- The provided checkpoints (pretrained/canon.pth, pretrained/nikon.pth)
  were trained with upstream's --gamma flag (see their train.sh/test.sh).
  That means the "RAW" side of the network is NOT scene-linear: it's
  gamma-compressed (raw_sensor ** (1/2.2)) sensor data, white-balance-
  premultiplied, bilinearly demosaiced to 3 channels, and normalized by
  the camera's bit-depth max (4095 for Canon 12-bit, 16383 for Nikon
  14-bit sensors -- see dataset/FiveK_dataset.py in the vendored copy's
  upstream history / openraw/third_party/invisp/UPSTREAM_README.md for context).
  net(x, rev=True) therefore returns that gamma-compressed
  representation. We undo the gamma (** 2.2) before handing the result
  to the rest of openraw, which expects scene-linear data from this
  point on (matching what tonecurve.srgb_to_linear produces on the
  classical path).
- The "RGB" side is plain sRGB-gamma pixels normalized by 255 -- exactly
  what openraw.decode.load_jpeg(...).rgb already is. No extra
  preprocessing needed on that side.
- Each checkpoint is camera-specific (upstream's own README says so
  explicitly: "one trained model can only be applied for a specific
  camera"). Picking the wrong camera for a JPEG from an unrelated camera
  is still expected to run, but the reconstructed "RAW" will reflect
  that camera's learned color/tone behavior, not the JPEG's actual
  source camera -- there is no way to avoid this without a model trained
  per-camera on data we don't have, which is exactly the limitation
  upstream's own README names.
"""

from __future__ import annotations

import os

import numpy as np

CAMERA_CHOICES = ("NIKON_D700", "Canon_EOS_5D")

_CKPT_FILENAMES = {
    "NIKON_D700": "nikon.pth",
    "Canon_EOS_5D": "canon.pth",
}

_GAMMA = 2.2


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "invisp_bridge needs PyTorch, which isn't installed. "
            "pip install 'openraw[invisp]' (torch is kept out of the "
            "base install since the classical deterministic "
            "pipeline in cli.py doesn't need it)."
        ) from e
    return torch


def _load_net(camera: str, pretrained_dir: str, device, checkpoint: str | None = None):
    """checkpoint: path to any InvISP checkpoint (e.g. a best.pth from your own
    training run) -- overrides camera/pretrained_dir. Weights-only state dicts
    and train.py's full-state latest_state.pth (weights under "net") both work."""
    torch = _require_torch()
    # Import deferred to here so importing this module doesn't require
    # torch unless you actually call something that needs it.
    from .third_party.invisp.model.model import InvISPNet

    if checkpoint is not None:
        ckpt_path = checkpoint
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    else:
        if camera not in CAMERA_CHOICES:
            raise ValueError(f"camera must be one of {CAMERA_CHOICES}, got {camera!r}")
        ckpt_path = os.path.join(pretrained_dir, _CKPT_FILENAMES[camera])
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"checkpoint not found: {ckpt_path} "
                f"(expected pretrained/{_CKPT_FILENAMES[camera]})"
            )

    # block_num=8, channel_in/out=3: matches upstream's test_rgb.py /
    # test_raw.py exactly -- these aren't tunable per-checkpoint, they're
    # baked into how canon.pth/nikon.pth were trained (and train.py's default).
    net = InvISPNet(channel_in=3, channel_out=3, block_num=8)
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)  # also train.py's dicts
    except TypeError:  # torch < 1.13
        state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and "net" in state_dict and "optimizer" in state_dict:
        state_dict = state_dict["net"]  # train.py's full-state latest_state.pth
    # strict=False only to tolerate upstream's checkpoints, which carry 16
    # leftover 'actnorm' entries from a layer the released model doesn't use.
    # Anything MISSING is an error: strict=False alone loaded nothing at all
    # from a full-state checkpoint -- 0 of 280 weights, silently, leaving a
    # randomly initialised network.
    result = net.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        raise ValueError(
            f"{ckpt_path} is not an InvISP checkpoint matching this network: "
            f"{len(result.missing_keys)} of {len(net.state_dict())} weights missing "
            f"(e.g. {result.missing_keys[0]})"
        )
    unexpected = [k for k in result.unexpected_keys if ".actnorm." not in k]
    if unexpected:
        raise ValueError(f"{ckpt_path} has weights this network doesn't use (e.g. {unexpected[0]})")
    net.to(device)
    net.eval()
    return net


def reconstruct_pseudo_raw(
    srgb_rgb: np.ndarray,
    camera: str = "NIKON_D700",
    pretrained_dir: str = "pretrained",
    device: str = "cpu",
    checkpoint: str | None = None,
) -> np.ndarray:
    """
    srgb_rgb: float32 HxWx3 in [0, 1] -- e.g. decode.load_jpeg(path).rgb directly.
    camera: one of CAMERA_CHOICES -- must match a checkpoint in pretrained_dir.
    checkpoint: path to a specific checkpoint instead (e.g. your own trained
        best.pth); overrides camera and pretrained_dir.
    returns: float32 HxWx3 in [0, 1], scene-linear camera-native pseudo-RAW
        (gamma undone -- see module docstring). Feed this into
        bitdepth.expand_to_16bit() and dng_writer.write_linear_dng() the
        same way tonecurve.srgb_to_linear()'s output is used on the
        classical path.
    """
    torch = _require_torch()

    if srgb_rgb.dtype != np.float32:
        srgb_rgb = srgb_rgb.astype(np.float32)

    h, w = srgb_rgb.shape[:2]
    # Not strictly required by the architecture, but we haven't verified
    # odd-dimension behavior against upstream (their own pipeline always
    # crops to even patch sizes), so we crop defensively rather than risk
    # silently wrong output on an edge case nobody's checked.
    h2, w2 = h - (h % 2), w - (w % 2)
    if (h2, w2) != (h, w):
        srgb_rgb = srgb_rgb[:h2, :w2, :]

    tensor = torch.from_numpy(srgb_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    net = _load_net(camera, pretrained_dir, device, checkpoint)
    with torch.no_grad():
        reconstructed = net(tensor, rev=True)
        reconstructed = torch.clamp(reconstructed, 0.0, 1.0)

    raw_gamma = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    raw_linear = np.power(np.clip(raw_gamma, 0.0, 1.0), _GAMMA).astype(np.float32)
    return raw_linear
