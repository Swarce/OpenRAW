"""
decode.py — load a JPEG and pull out what we can about how lossy it was.

We use Pillow for the actual decode (robust, handles every JPEG variant
in the wild), but we also read the embedded quantization tables. These
tell us, per 8x8 block, how aggressively each DCT frequency was crushed —
that's the signal the deblocking stage uses to decide how hard to work,
instead of applying one fixed amount of smoothing to every JPEG regardless
of its actual quality level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


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
    exif: dict = field(default_factory=dict)
    source_path: str = ""


def _unzigzag(flat_table: list[int]) -> np.ndarray:
    block = np.zeros(64, dtype=np.float64)
    block[_ZIGZAG] = np.array(flat_table, dtype=np.float64)
    return block.reshape(8, 8)


def _estimate_quality(quant_tables: dict) -> float:
    """
    Rough JPEG quality estimate from the luma quant table, using the same
    logic libjpeg uses in reverse (sum of table values maps monotonically
    to the quality factor). This is only used to scale deblocking strength
    — it does not need to be exact, just monotonic.
    """
    if 0 not in quant_tables:
        return 85.0
    luma = quant_tables[0]
    # The IJG standard luma table at quality=50 sums to 2504. Scale is
    # roughly linear around that in the 0-100 range for typical encoders.
    total = float(luma.sum())
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
    img.load()  # forces decode, populates img.quantization

    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.asarray(img).astype(np.float32) / 255.0

    quant_tables = {}
    raw_q = getattr(img, "quantization", None)
    if raw_q:
        for idx, table in raw_q.items():
            quant_tables[idx] = _unzigzag(list(table))

    quality_estimate = _estimate_quality(quant_tables) if quant_tables else 85.0

    exif = {}
    try:
        exif = dict(img.getexif())
    except Exception:
        pass

    return DecodedJpeg(
        rgb=arr,
        width=arr.shape[1],
        height=arr.shape[0],
        quant_tables=quant_tables,
        quality_estimate=quality_estimate,
        exif=exif,
        source_path=path,
    )
