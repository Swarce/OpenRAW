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
    return tf.pages[0].pages[0]


def main_array(path: str):
    """Pixel data of the main (full-resolution) image, not the preview."""
    return tifffile.imread(path, series=1)
