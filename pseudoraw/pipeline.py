"""
pipeline.py — wires the stages together in order:

    JPEG --(decode)--> sRGB pixels + quant tables
         --(deblock)--> cleaned sRGB
         --(chroma refine)--> cleaned sRGB
         --(tonecurve: srgb_to_linear [+ optional generic S-curve undo])--> linear
         --(bitdepth: dither + deband, expand to 16-bit)--> uint16 linear
         --(write_linear_dng)--> .dng file

Each stage is independently testable/importable; this module just owns
the ordering and the config surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decode import load_jpeg, DecodedJpeg
from .deblock import deblock
from .chroma import refine_chroma
from .tonecurve import srgb_to_linear, remove_generic_s_curve
from .colormatrix import apply_experimental_gamut_widen
from .bitdepth import expand_to_16bit
from .dng_writer import write_linear_dng

_PIPELINE_VERSION = "0.0.1-poc"


@dataclass
class PipelineConfig:
    deblock_enabled: bool = True
    chroma_refine_enabled: bool = True
    chroma_refine_strength: float = 0.6
    generic_s_curve_strength: float = 0.0  # off by default -- see tonecurve.py
    experimental_gamut_widen: float = 0.0  # off by default -- see colormatrix.py
    dither: bool = True
    deband: bool = True
    seed: int | None = 0

    # Real InvISP network path (pseudoraw/invisp_bridge.py), as an
    # alternative to the classical deblock/chroma/tonecurve stages above.
    # Requires torch + a matching checkpoint in pretrained/ -- see
    # invisp_bridge.py's module docstring for exactly what this does and
    # does not reproduce from upstream. Off by default: the classical
    # path has actually been run and tested in this environment; this
    # one hasn't (no torch available here -- see NOTICE.md).
    use_invisp: bool = False
    invisp_camera: str = "NIKON_D700"
    invisp_pretrained_dir: str = "pretrained"
    invisp_device: str = "cpu"


@dataclass
class PipelineResult:
    decoded: DecodedJpeg
    linear_rgb: np.ndarray  # float32, pre-16bit-expansion, for previews/debugging
    rgb16: np.ndarray  # uint16, final data written to DNG


class PseudoRawPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def run(self, jpeg_path: str) -> PipelineResult:
        cfg = self.config
        decoded = load_jpeg(jpeg_path)
        srgb_u8 = (decoded.rgb * 255.0 + 0.5).astype(np.uint8)

        if cfg.use_invisp:
            # Real InvISP network path -- see invisp_bridge.py docstring
            # for exactly what this does and doesn't reproduce from
            # upstream, and for the (untested-in-this-environment) caveat.
            from .invisp_bridge import reconstruct_pseudo_raw

            linear = reconstruct_pseudo_raw(
                decoded.rgb,
                camera=cfg.invisp_camera,
                pretrained_dir=cfg.invisp_pretrained_dir,
                device=cfg.invisp_device,
            )
            # source_srgb_u8 must match linear's possibly-cropped shape
            # (reconstruct_pseudo_raw crops to even dimensions).
            srgb_u8 = srgb_u8[: linear.shape[0], : linear.shape[1], :]
        else:
            working = decoded.rgb
            if cfg.deblock_enabled:
                working = deblock(working, decoded.quality_estimate)
            if cfg.chroma_refine_enabled:
                working = refine_chroma(working, strength=cfg.chroma_refine_strength)

            linear = srgb_to_linear(working)
            if cfg.generic_s_curve_strength > 0:
                linear = remove_generic_s_curve(linear, strength=cfg.generic_s_curve_strength)
            if cfg.experimental_gamut_widen > 0:
                linear = apply_experimental_gamut_widen(linear, strength=cfg.experimental_gamut_widen)

        rgb16 = expand_to_16bit(
            linear,
            source_srgb_u8=srgb_u8,
            dither=cfg.dither,
            deband=cfg.deband,
            seed=cfg.seed,
        )

        return PipelineResult(decoded=decoded, linear_rgb=linear, rgb16=rgb16)

    def run_to_dng(self, jpeg_path: str, out_path: str) -> PipelineResult:
        result = self.run(jpeg_path)
        write_linear_dng(
            out_path,
            result.rgb16,
            source_jpeg_path=jpeg_path,
            pipeline_version=_PIPELINE_VERSION,
        )
        return result
