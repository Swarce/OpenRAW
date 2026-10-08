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

from ._version import __version__ as _PIPELINE_VERSION


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

    # Output DNG: lossless-JPEG tiles + 12-bit LinearizationTable by default
    # (~half the size of uncompressed, verified in Adobe's SDK and libraw).
    # See dng_writer.write_linear_dng for the measured size/precision table.
    dng_compression: str = "ljpeg"  # or "none"
    dng_bit_depth: int = 12  # 16 (bit-exact), 14, 12, 10
    encode_threads: int | None = None  # LJPEG tile-encoding threads; None = all cores
    dng_layout: str = "linear"  # or "cfa": Bayer mosaic, demosaiced by the raw editor
    preserve_exif: bool = True  # no GPS ever -- dropped entirely, see exif_transfer.py
    write_preview: bool = True  # small JPEG preview + SubIFD main image -- see dng_writer.py
    preview_max_dim: int = 1024
    preview_quality: int = 90

    # Real InvISP network path (openraw/invisp_bridge.py), as an
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
    invisp_checkpoint: str | None = None  # any trained checkpoint; overrides camera/dir
    invisp_tile: int = 512  # tiled inference (same result, ~1 GB peak); 0 = whole image


@dataclass
class PipelineResult:
    decoded: DecodedJpeg
    linear_rgb: np.ndarray  # float32, pre-16bit-expansion, for previews/debugging
    rgb16: np.ndarray  # uint16, final data written to DNG


class OpenRawPipeline:
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
                checkpoint=cfg.invisp_checkpoint,
                tile=cfg.invisp_tile,
            )
            # source_srgb_u8 must match linear's possibly-cropped shape
            # (reconstruct_pseudo_raw crops to even dimensions).
            srgb_u8 = srgb_u8[: linear.shape[0], : linear.shape[1], :]
        else:
            working = decoded.rgb
            if cfg.deblock_enabled:
                working = deblock(working, decoded.quality_estimate)
            if cfg.chroma_refine_enabled:
                working = refine_chroma(
                    working,
                    strength=cfg.chroma_refine_strength,
                    is_chroma_subsampled=decoded.is_chroma_subsampled,
                    chroma_quality_estimate=decoded.chroma_quality_estimate,
                )

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
        cfg = self.config
        result = self.run(jpeg_path)
        write_linear_dng(
            out_path,
            result.rgb16,
            source_jpeg_path=jpeg_path,
            pipeline_version=_PIPELINE_VERSION,
            compression=cfg.dng_compression,
            bit_depth=cfg.dng_bit_depth,
            threads=cfg.encode_threads,
            layout=cfg.dng_layout,
            exif_fields=result.decoded.exif_fields if cfg.preserve_exif else None,
            write_preview=cfg.write_preview,
            preview_max_dim=cfg.preview_max_dim,
            preview_quality=cfg.preview_quality,
        )
        return result
