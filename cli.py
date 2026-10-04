#!/usr/bin/env python3
"""
pseudoraw CLI — POC.

    python3 cli.py input.jpg output.dng
    python3 cli.py input.jpg output.dng --s-curve 0.2 --gamut-widen 0.3
    python3 cli.py input.jpg output.dng --no-deblock --no-deband
"""

from __future__ import annotations

import argparse
import sys
import time

from pseudoraw import PseudoRawPipeline, PipelineConfig


def main() -> int:
    p = argparse.ArgumentParser(description="Reconstruct a pseudo-RAW (linear DNG) from a JPEG.")
    p.add_argument("input", help="input .jpg/.jpeg path")
    p.add_argument("output", help="output .dng path")
    p.add_argument("--no-deblock", action="store_true", help="disable deblocking stage")
    p.add_argument("--no-chroma-refine", action="store_true", help="disable chroma refinement")
    p.add_argument("--chroma-strength", type=float, default=0.6)
    p.add_argument(
        "--s-curve",
        type=float,
        default=0.0,
        help="EXPERIMENTAL generic contrast-curve removal strength (0 disables; try 0.1-0.4)",
    )
    p.add_argument(
        "--gamut-widen",
        type=float,
        default=0.0,
        help="EXPERIMENTAL gamut widening strength, a look not a reconstruction (0 disables)",
    )
    p.add_argument("--no-dither", action="store_true")
    p.add_argument("--no-deband", action="store_true")
    p.add_argument(
        "--invisp",
        action="store_true",
        help=(
            "Use the real InvISP network (third_party/invisp/ + pretrained/*.pth) "
            "instead of the classical deblock/chroma/tonecurve stages. Requires "
            "torch (pip install -r requirements-invisp.txt) and has NOT been "
            "executed in the environment this project was built in -- see "
            "pseudoraw/invisp_bridge.py's docstring before trusting it."
        ),
    )
    p.add_argument(
        "--invisp-camera",
        choices=["NIKON_D700", "Canon_EOS_5D"],
        default="NIKON_D700",
        help="Which checkpoint to use with --invisp (must match a file in pretrained/).",
    )
    p.add_argument("--invisp-pretrained-dir", default="pretrained")
    p.add_argument("--invisp-device", default="cpu", help="'cpu' or 'cuda:0' etc.")
    p.add_argument("--no-compress", action="store_true", help="write fully uncompressed DNG (not recommended, see dng_writer.py)")
    p.add_argument("--compression-level", type=int, default=9, help="1 (fastest) - 9 (smallest), default 9")
    p.add_argument("--no-exif", action="store_true", help="don't carry camera metadata (Make/Model/lens/exposure/etc) from the source JPEG into the DNG")
    p.add_argument(
        "--preserve-gps",
        action="store_true",
        help=(
            "Currently a no-op (see exif_transfer.py: GPS needs a proper sub-IFD "
            "writer, not implemented yet) -- reserved for when that's done. "
            "GPS is off by default even once implemented: it's capture location, "
            "worth opting into deliberately, not forwarding silently."
        ),
    )
    args = p.parse_args()

    config = PipelineConfig(
        deblock_enabled=not args.no_deblock,
        chroma_refine_enabled=not args.no_chroma_refine,
        chroma_refine_strength=args.chroma_strength,
        generic_s_curve_strength=args.s_curve,
        experimental_gamut_widen=args.gamut_widen,
        dither=not args.no_dither,
        deband=not args.no_deband,
        use_invisp=args.invisp,
        invisp_camera=args.invisp_camera,
        invisp_pretrained_dir=args.invisp_pretrained_dir,
        invisp_device=args.invisp_device,
        dng_compress=not args.no_compress,
        dng_compression_level=args.compression_level,
        preserve_exif=not args.no_exif,
        preserve_gps=args.preserve_gps,
    )

    pipeline = PseudoRawPipeline(config)

    t0 = time.time()
    result = pipeline.run_to_dng(args.input, args.output)
    dt = time.time() - t0

    d = result.decoded
    print(f"input:           {args.input}  ({d.width}x{d.height})")
    print(f"luma quality:    {d.quality_estimate:.1f} / 100  (drives deblock strength)")
    if d.chroma_quality_estimate is not None:
        subsampled = "4:2:0/4:2:2-ish, subsampled" if d.is_chroma_subsampled else "4:4:4, NOT subsampled"
        print(f"chroma quality:  {d.chroma_quality_estimate:.1f} / 100  ({subsampled})")
    if d.exif_fields and (d.exif_fields.get("ifd0") or d.exif_fields.get("exif_sub")):
        make = d.exif_fields.get("ifd0", {}).get(271, "")
        model = d.exif_fields.get("ifd0", {}).get(272, "")
        cam = f"{make} {model}".strip()
        n_fields = len(d.exif_fields.get("ifd0", {})) + len(d.exif_fields.get("exif_sub", {}))
        print(f"camera metadata: {n_fields} field(s) found" + (f" ({cam})" if cam else "") + (" -- carried into output" if not args.no_exif else " -- NOT carried (--no-exif)"))
    else:
        print("camera metadata: none found in source JPEG")
    print(f"output:          {args.output}")
    print(f"elapsed:         {dt:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
