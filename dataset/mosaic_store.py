"""
Compact storage for training raws: the sensor MOSAIC (one value per pixel),
not a demosaiced 3-channel image, demosaiced at load time -- only the crop a
training step actually uses.

Why: storing the demosaiced float32 image (upstream's format) costs ~216 MB
for an 18 MP photo -- 20x+ its DNG -- and filled disks fast. The mosaic,
split into its four CFA planes (R, G, G, B at half resolution, so lossless
JPEG's predictor compares like with like) and stored as 4 lossless-JPEG
streams: ~18 MB for the same photo, bit-exact, and faster to read than zlib.

Format ("mosaic-ljpeg-v1", inside a .npz):
    format       'mosaic-ljpeg-v1'
    shape        (H, W) of the mosaic, already rotated to display orientation
    cfa_pattern  'RGGB' / 'GRBG' / 'BGGR' / 'GBRG' at pixel (0, 0)
    p0..p3       lossless-JPEG bytes of the planes [0::2,0::2], [0::2,1::2],
                 [1::2,0::2], [1::2,1::2]
    bits         bits per sample used for the encoding
    white_level  black-subtracted white level (normalization range)
    wb           camera white balance (4 values, as rawpy reports them)
"""

from __future__ import annotations

import numpy as np

FORMAT = "mosaic-ljpeg-v1"
PATTERNS = ("RGGB", "GRBG", "BGGR", "GBRG")
_CH = {"R": 0, "G": 1, "B": 2}


def _index_map(pattern: str, h: int, w: int) -> np.ndarray:
    tile = np.array([_CH[c] for c in pattern]).reshape(2, 2)
    return np.tile(tile, ((h + 1) // 2, (w + 1) // 2))[:h, :w]


def rotate_mosaic(mosaic: np.ndarray, pattern: str, flip: int):
    """Rotate a mosaic to display orientation (rawpy sizes.flip: 3=180,
    5=90 CCW, 6=90 CW). Rotating a Bayer mosaic yields another Bayer layout,
    found by rotating the colour-index map identically (handles odd sizes)."""
    k = {3: 2, 5: 1, 6: 3}.get(flip, 0)
    if k == 0:
        return mosaic, pattern
    idx = np.rot90(_index_map(pattern, *mosaic.shape), k)
    new = "".join("RGB"[i] for i in idx[:2, :2].flatten())
    return np.ascontiguousarray(np.rot90(mosaic, k)), new


def pack(mosaic: np.ndarray, pattern: str, white_level: float, wb) -> dict:
    """mosaic: 2-D uint16, black-subtracted, clipped to white_level."""
    import imagecodecs
    if pattern not in PATTERNS:
        raise ValueError(f"unsupported CFA pattern {pattern!r}")
    mosaic = np.ascontiguousarray(mosaic, dtype=np.uint16)
    bits = max(2, int(mosaic.max()).bit_length())
    planes = [mosaic[0::2, 0::2], mosaic[0::2, 1::2], mosaic[1::2, 0::2], mosaic[1::2, 1::2]]
    out = {f"p{i}": np.frombuffer(imagecodecs.ljpeg_encode(np.ascontiguousarray(p), bitspersample=bits), np.uint8)
           for i, p in enumerate(planes)}
    out.update(format=FORMAT, shape=np.array(mosaic.shape), cfa_pattern=pattern, bits=bits,
               white_level=np.float32(white_level), wb=np.asarray(wb, np.float32))
    return out


def is_mosaic(npz) -> bool:
    return "format" in npz.files and str(npz["format"]) == FORMAT


def unpack(npz):
    """-> (mosaic uint16 HxW, pattern)"""
    import imagecodecs
    h, w = (int(v) for v in npz["shape"])
    m = np.empty((h, w), np.uint16)
    for i, (sy, sx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        m[sy::2, sx::2] = imagecodecs.ljpeg_decode(npz[f"p{i}"].tobytes())
    return m, str(npz["cfa_pattern"])


def demosaic(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    import colour_demosaicing
    return colour_demosaicing.demosaicing_CFA_Bayer_bilinear(mosaic.astype(np.float32), pattern).astype(np.float32)


def demosaic_region(mosaic, pattern, y, x, h, w, margin=2):
    """Bilinear-demosaic only [y:y+h, x:x+w], with a margin so the region's
    edges see the same neighbours as a full-image demosaic would. y and x
    must be even (keeps the CFA phase, so `pattern` stays valid)."""
    if y % 2 or x % 2:
        raise ValueError("region origin must be even to keep the CFA phase")
    H, W = mosaic.shape
    m = margin + (margin % 2)  # even margin keeps the phase too
    y0, x0 = max(0, y - m), max(0, x - m)
    y1, x1 = min(H, y + h + m), min(W, x + w + m)
    de = demosaic(mosaic[y0:y1, x0:x1], pattern)
    return de[y - y0:y - y0 + h, x - x0:x - x0 + w]


def _remosaic_with(demosaiced, pattern):
    h, w, _ = demosaiced.shape
    idx = _index_map(pattern, h, w)
    m = np.take_along_axis(demosaiced, idx[..., None], 2)[..., 0]
    # At the outermost pixel ring, the filter's reflect mode folds each sample
    # back onto itself (an edge red sample comes out 1.5x, a corner one 2.25x).
    # It's linear and self-only, so demosaicing a mosaic of ones gives each
    # site's multiplier exactly -- divide it back out.
    gain = np.take_along_axis(demosaic(np.ones((h, w), np.float32), pattern), idx[..., None], 2)[..., 0]
    return np.clip(np.round(m / gain), 0, 65535).astype(np.uint16)


def _reproduction_error(demosaiced, pattern, region=None):
    d = demosaiced if region is None else demosaiced[region]
    re = demosaic(_remosaic_with(d, pattern).astype(np.float32), pattern)
    # skip a 2-px border: the outermost ring may hold values the old format
    # clipped, and re-demosaicing spreads them one pixel inward
    return float(np.abs(re[2:-2, 2:-2] - d[2:-2, 2:-2]).max())


def remosaic(demosaiced: np.ndarray, hint: str | None = None):
    """Recover the original mosaic from a BILINEAR-demosaiced image (the
    previous storage format) -- exact: bilinear demosaicing keeps every
    original sample at its own site (scaled by a known factor at the outermost
    pixel ring, undone in _remosaic_with). One caveat for the previous format:
    values it clipped at the white level at that ring can't be un-clipped, so
    a migrated file can differ from a fresh one within 2 px of the image edge
    (where clipping happened); everything else is bit-exact.

    The pattern is identified as the one whose re-demosaic reproduces the
    image. A small corner probe decides quickly, but a FLAT corner (clear sky,
    blown highlights) fits every pattern equally -- found by a test where it
    silently picked the wrong one -- so ties are settled on the full image.
    hint: a stored pattern, preferred among exact ties.
    -> (mosaic uint16, pattern)."""
    h, w, _ = demosaiced.shape
    probe = (slice(0, min(h, 64) // 2 * 2), slice(0, min(w, 64) // 2 * 2))
    errs = {p: _reproduction_error(demosaiced, p, probe) for p in PATTERNS}
    cands = [p for p in PATTERNS if errs[p] <= 0.5]
    if len(cands) > 1:
        errs = {p: _reproduction_error(demosaiced, p) for p in cands}
        best_err = min(errs.values())
        cands = [p for p in cands if errs[p] <= max(0.5, best_err)]
    if not cands:
        raise ValueError(f"not a bilinear-demosaiced image (min error {min(errs.values()):.2f})")
    best = hint if hint in cands else cands[0]
    return _remosaic_with(demosaiced, best), best
