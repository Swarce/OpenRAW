# DNG format notes

How `pseudoraw/dng_writer.py` writes files, why each choice was made, and —
most importantly for contributors — **how to test changes**, because every
real bug this writer has had passed a lenient reader first.

## File structure

```
IFD0          small JPEG preview (NewSubfileType=1)
              + ALL identity/color/EXIF tags: DNGVersion, UniqueCameraModel,
                ColorMatrix1, CalibrationIlluminant1, AsShotNeutral,
                BaselineExposure, Make/Model/Orientation, EXIF fields
  └ SubIFD    main image (NewSubfileType=0): PhotometricInterpretation =
              LinearRaw (34892), 3 samples, 256×256 lossless-JPEG tiles
              + raw-data tags only: BlackLevel, WhiteLevel, LinearizationTable
```

With `--no-preview`, everything lives in a single IFD0.

**Tag placement is not cosmetic.** Readers built on Adobe's DNG SDK
(Android's Skia, Luminar, most commercial tools) identify a DNG by finding
`DNGVersion` *in IFD0*. When the preview was first added, these tags stayed
attached to the main image and moved into the SubIFD; libraw still opened
the file (it scans every IFD), but Skia and Luminar rejected it.

`UniqueCameraModel` is always `pseudoraw virtual sensor`, even when the real
Make/Model is copied from EXIF, so no reader mistakes the data for that
camera's sensor output.

## Compression

Each tile is a 3-component **lossless JPEG** (ITU T.81 process 14,
predictor 1, no color transform) — the codec real cameras use for raw, and
the layout Adobe's DNG Converter writes for linear DNGs. Encoded with
libjpeg-turbo ≥ 3 via `imagecodecs`, on a thread pool (output is
byte-identical at any thread count).

Below 16 bits, values are stored sRGB-gamma-encoded with a DNG
**LinearizationTable** (tag 50712) that readers use to expand them back to
linear — the same idea as the tone curves inside Nikon and Leica raw files.
Gamma spends precision evenly across tones, so fewer bits suffice.

Measured on a 4896×3672 photo (error in units of the source JPEG's own
8-bit steps):

| mode | size | error |
|---|---|---|
| `--compression none` | 108 MB | none |
| ljpeg 16-bit | 76 MB | none (bit-exact) |
| ljpeg 14-bit | 64 MB | ≤ 0.02 |
| ljpeg 12-bit (default) | 50 MB | ≤ 0.06 |
| ljpeg 10-bit | 37 MB | ≤ 0.16 |

The floor for reference: the source's own 8-bit pixels need ~24 MB stored
losslessly. Dither noise accounts for only ~2 MB of the default size.

### Dead ends (don't repeat these)

| tried | result |
|---|---|
| Deflate / LZW / PackBits | libraw rejects all three for this structure |
| lossy (DCT) JPEG for main data | ~30000/65535 mean error on 16-bit data |
| single-component "W×3 wide" LJPEG | passes Adobe's SDK, but libraw scrambles every tile |
| letting tifffile infer LinearRaw layout | varies by tifffile version (see below) |

## tifffile version-proofing

tifffile doesn't know LinearRaw (34892), so it must *guess* how the 3
samples relate to it — and versions guess differently. One version wrote a
spurious `ExtraSamples` tag; 2026.9.20 miscounts tiles and raises
`StopIteration`. So the main image is written as plain RGB (which every
version understands exactly) and `_finalize_main_ifd` patches it in place:
PhotometricInterpretation → 34892, and the YCbCr tags tifffile adds to any
JPEG-compressed RGB IFD are removed. Verified byte-identical output on
tifffile 2025.5.10, 2026.3.3, and 2026.9.20.

## Testing changes to the writer

Three readers, from lenient to strict. **A change isn't verified until it
passes the strict one.**

1. **tifffile** — wrote the file, so of course it reads it back. Necessary,
   never sufficient.
2. **libraw** (`pip install rawpy`) — what darktable, RawTherapee and many
   open tools use. Lenient about structure: scans all IFDs.
3. **Adobe's `dng_validate`** — the reference implementation; strict. Build it:

   ```bash
   tools/build_dng_validate.sh          # Linux/macOS: g++ and zlib
   export PSEUDORAW_DNG_VALIDATE=$PWD/.dng_validate_build/dng_validate
   python -m pytest                     # Adobe tests now run instead of skipping
   ```

   It clones a public mirror of Adobe's DNG SDK (not vendored — Adobe's own
   license) and stubs out the XMP toolkit, which doesn't affect structural
   validation or decoding. CI runs this on every push.

Decode checks should compare pixels, not just "it opened": the test suite
compares libraw's decode of each compression mode against the uncompressed
file, and `dng_validate -tif out.tif file.dng` renders through Adobe's SDK.
