"""
Shared test helpers. Since dng_writer.py switched to writing a preview
(IFD0) + main image (SubIFD) structure, tf.pages[0] is now the PREVIEW,
not the main reconstructed data -- these centralize "get me the real
main image" so test files don't each re-derive it slightly differently.
"""

from __future__ import annotations

import tifffile


def main_page(tf: tifffile.TiffFile):
    """The main (full-resolution LinearRaw) TiffPage, i.e. the SubIFD
    off of IFD0's preview -- NOT tf.pages[0], which is the preview."""
    p0 = tf.pages[0]
    return p0.pages[0] if p0.pages else p0  # single-IFD layout: IFD0 IS the main image


def main_array(path: str):
    """Pixel data of the main (full-resolution) image, not the preview."""
    with tifffile.TiffFile(path) as tf:
        return main_page(tf).asarray()


def ifd0_tags(tf: tifffile.TiffFile):
    """IFD0's tags -- where the DNG spec puts DNGVersion, Make/Model,
    ColorMatrix1, EXIF etc. (NOT the raw SubIFD; see test_dng_identity_tags_live_in_ifd0)."""
    return tf.pages[0].tags


def decoded_linear(path: str):
    """Main image as a real raw reader sees it: stored values passed through
    the DNG LinearizationTable (tag 50712) when present, else as-is."""
    import numpy as np
    with tifffile.TiffFile(path) as tf:
        page = main_page(tf)
        arr = page.asarray()
        if 50712 in page.tags:
            table = np.asarray(page.tags[50712].value, dtype=np.uint16)
            arr = table[np.minimum(arr, len(table) - 1)]
    return arr
