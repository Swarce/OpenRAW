"""
decode.py — load a JPEG and pull out what we can about how lossy it was,
plus its real camera metadata.

We use Pillow for the actual decode (robust, handles every JPEG variant
in the wild), but we also read: the embedded quantization tables (how
aggressively each DCT frequency was crushed -- drives deblock.py's
strength and, since 2.0, which quant table chroma actually used, not
just luma); the real per-component chroma sampling factors (so chroma.py
can tell a genuinely-subsampled 4:2:0 JPEG apart from a 4:4:4 one that
never needed chroma upsampling in the first place); and camera EXIF
(Make/Model/lens/exposure/etc, via exif_transfer.py) to carry into the
output DNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .exif_transfer import extract_exif

# Standard JPEG zigzag order for an 8x8 block, used to unpack the
# quantization tables Pillow hands back in zigzag form.
_ZIGZAG = np.array(
    [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ]
)


@dataclass
class DecodedJpeg:
    rgb: np.ndarray  # float32, HxWx3, range [0, 1]
    width: int
    height: int
    quant_tables: dict = field(default_factory=dict)  # {table_idx: 8x8 int array}
    quality_estimate: float = 85.0  # rough 0-100 estimate, for deblock strength
    chroma_quality_estimate: float | None = None  # same, from the chroma quant table specifically
    is_chroma_subsampled: bool = True  # False for 4:4:4 JPEGs -- see chroma.py
    exif: dict = field(default_factory=dict)  # legacy flat IFD0 dict, kept for back-compat
    exif_fields: dict = field(default_factory=dict)  # structured: exif_transfer.extract_exif() output
    source_path: str = ""


def _unzigzag(flat_table: list[int]) -> np.ndarray:
    block = np.zeros(64, dtype=np.float64)
    block[_ZIGZAG] = np.array(flat_table, dtype=np.float64)
    return block.reshape(8, 8)


def _estimate_quality(quant_table: np.ndarray | None) -> float | None:
    """
    Rough JPEG quality estimate from a quant table, using the same logic
    libjpeg uses in reverse (sum of table values maps monotonically to
    the quality factor). Works for luma or chroma tables -- just pass
    the right one. This is only used to scale deblocking/chroma-refine
    strength -- it does not need to be exact, just monotonic.

    Note on the reference baseline: the IJG standard LUMA table at
    quality=50 sums to 2504, which is what this is calibrated against.
    The standard CHROMA table at quality=50 sums differently (it's a
    different base table), so this estimate is only meaningfully
    comparable across tables of the SAME kind (luma-vs-luma or
    chroma-vs-chroma) -- not directly between a luma estimate and a
    chroma estimate for the same image. Each is still individually
    useful as "how aggressively was *this* channel quantized."
    """
    if quant_table is None:
        return None
    total = float(quant_table.sum())
    baseline_sum_q50 = 2504.0
    if total <= 0:
        return 95.0
    ratio = baseline_sum_q50 / total
    if ratio >= 1.0:
        quality = min(100.0, 50.0 + (ratio - 1.0) * 50.0)
    else:
        quality = max(1.0, 50.0 * ratio)
    return float(np.clip(quality, 1.0, 100.0))


def load_jpeg(path: str) -> DecodedJpeg:
    img = Image.open(path)
    img.load()  # forces decode, populates img.quantization/.layer/exif

    # Capture every JPEG-specific attribute BEFORE any convert() call.
    # img.convert() returns a plain Image that does not carry forward
    # .quantization, .layer, or (reliably) EXIF -- reading these after a
    # convert() silently loses them for any non-RGB-mode source (CMYK,
    # grayscale). This used to happen in the wrong order; fixed here.
    quant_tables = {}
    raw_q = getattr(img, "quantization", None)
    if raw_q:
        for idx, table in raw_q.items():
            quant_tables[idx] = _unzigzag(list(table))

    # img.layer: list of (component_id, h_sampling, v_sampling, quant_idx)
    # per JPEG component, in encode order (typically Y, Cb, Cr). All
    # sampling factors equal (e.g. all (1,1)) means 4:4:4 -- chroma was
    # never subsampled, so chroma.py's upsample-correction has nothing
    # to correct and should get out of the way rather than risk
    # softening/altering chroma that was already full-resolution.
    layer = getattr(img, "layer", None)
    is_chroma_subsampled = True
    chroma_quant_idx = 1
    if layer and len(layer) >= 1:
        sampling_factors = {(h, v) for (_cid, h, v, _qidx) in layer}
        is_chroma_subsampled = len(sampling_factors) > 1
        # Quant table index actually used by the chroma components
        # (component index 1 by position, typically Cb) -- usually 1,
        # but don't assume; read it from what the encoder actually did.
        if len(layer) >= 2:
            chroma_quant_idx = layer[1][3]

    quality_estimate = _estimate_quality(quant_tables.get(0))
    if quality_estimate is None:
        quality_estimate = 85.0
    chroma_quality_estimate = _estimate_quality(quant_tables.get(chroma_quant_idx))

    exif_obj = img.getexif()
    exif_fields = {}
    exif_flat = {}
    try:
        exif_fields = extract_exif(exif_obj)
        exif_flat = dict(exif_obj)
    except Exception:
        pass

    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.asarray(img).astype(np.float32) / 255.0

    return DecodedJpeg(
        rgb=arr,
        width=arr.shape[1],
        height=arr.shape[0],
        quant_tables=quant_tables,
        quality_estimate=quality_estimate,
        chroma_quality_estimate=chroma_quality_estimate,
        is_chroma_subsampled=is_chroma_subsampled,
        exif=exif_flat,
        exif_fields=exif_fields,
        source_path=path,
    )
