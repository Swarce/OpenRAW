"""Before/after images for the compare view: the camera JPEG, and the DNG as
a raw converter renders it (LibRaw via rawpy: camera white balance, no
auto-brightening, no tone curve -- "flat", the way a raw editor first opens
it before its own default look), optionally gain-matched to the JPEG.
Runs off the UI thread; returns plain numpy arrays."""

from __future__ import annotations

import numpy as np

FLAT, MATCHED = 0, 1


def _enc(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _dec(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_jpeg(path: str) -> np.ndarray:
    """uint8 HxWx3, with the photo's EXIF rotation NOT applied -- the DNG
    stores pixels unrotated too, and both are shown the same way."""
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def render_dng(path: str) -> np.ndarray:
    """float32 HxWx3 linear sRGB, unrotated, DefaultCrop applied."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        out = r.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1), user_flip=0,
                            output_color=rawpy.ColorSpace.sRGB)
        s = r.sizes
    if s.crop_width and s.crop_height and (s.crop_width, s.crop_height) != (out.shape[1], out.shape[0]):
        out = out[s.crop_top_margin:s.crop_top_margin + s.crop_height,
                  s.crop_left_margin:s.crop_left_margin + s.crop_width]
    return out.astype(np.float32) / 65535.0


def match_gains(lin: np.ndarray, ref_lin: np.ndarray) -> np.ndarray:
    """One gain per channel mapping lin onto ref (least squares on a
    subsample of unclipped pixels): exposure and white balance matched."""
    h, w = min(lin.shape[0], ref_lin.shape[0]), min(lin.shape[1], ref_lin.shape[1])
    step = max(1, int(np.sqrt(h * w / 250_000)))
    a, b = lin[:h:step, :w:step].reshape(-1, 3), ref_lin[:h:step, :w:step].reshape(-1, 3)
    ok = (a.max(axis=1) < 0.98) & (b.max(axis=1) < 0.98)
    if ok.sum() < 100:
        return np.ones(3, np.float32)
    a, b = a[ok], b[ok]
    return ((a * b).sum(0) / np.maximum((a * a).sum(0), 1e-12)).astype(np.float32)


def dng_display(lin: np.ndarray, mode: int, jpeg_u8: np.ndarray | None = None) -> np.ndarray:
    """uint8 sRGB image of a rendered DNG for display."""
    if mode == MATCHED and jpeg_u8 is not None:
        lin = lin * match_gains(lin, _dec(jpeg_u8.astype(np.float32) / 255))
    return (_enc(lin) * 255 + 0.5).astype(np.uint8)


def available() -> bool:
    try:
        import rawpy  # noqa: F401
        return True
    except ImportError:
        return False
