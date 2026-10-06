"""
openraw — reconstruct a pseudo-RAW (linear DNG) from a JPEG.

This is NOT real RAW recovery. JPEG is lossy and irreversible (8-bit,
tone-curved, chroma-subsampled, DCT-quantized). What this pipeline produces
is a plausible *linear, high-bit-depth* reconstruction that gives raw
converters more room to grade than the baked JPEG did — nothing more, and
the code is written to be honest about that at every stage.
"""

from .pipeline import OpenRawPipeline, PipelineConfig

__all__ = ["OpenRawPipeline", "PipelineConfig"]
from ._version import __version__
