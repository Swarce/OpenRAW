# How it works

OpenRAW turns an 8-bit, gamma-encoded, lossy JPEG into a 16-bit **linear**
DNG — or, with `--layout cfa`, a Bayer mosaic of the same linear data that
your raw editor demosaics itself. Each stage below is labelled by how much it can be trusted — the same
labels the module docstrings use — because the honest answer differs a lot
between stages.

- **Exact** — mathematically correct inversion; adds no error.
- **Defensible** — standard signal processing with a bounded, measured effect.
- **Experimental** — a guess; off by default.

```
JPEG
 │ decode.py        Pillow decode + quantization tables, chroma sampling, EXIF
 ▼
 │ deblock.py       edge-aware deblocking, strength from the JPEG's real quality
 ▼
 │ chroma.py        re-sharpen chroma edges against luma (skipped for 4:4:4)
 ▼
 │ tonecurve.py     undo sRGB gamma -> scene-linear  (+ optional S-curve undo)
 ▼
 │ colormatrix.py   declare the color space honestly (no invented sensor)
 ▼
 │ bitdepth.py      8 -> 16 bit: flatness-gated dither + clamped debanding
 ▼
 │ dng_writer.py    lossless-JPEG tiled DNG (Linear, or CFA mosaic) + preview + EXIF
 ▼
DNG
```

## decode.py — read the JPEG *and* how it was made

Besides pixels, it reads the embedded quantization tables (how aggressively
each frequency was crushed), the per-component chroma sampling factors
(4:2:0 vs 4:4:4), and camera EXIF. These drive the later stages: a JPEG
saved at quality 98 gets almost no deblocking, one saved at 40 gets a lot.

## deblock.py — *defensible*

An edge-aware guided filter, strength scaled by the JPEG's estimated
quality, with extra smoothing on the 8×8 block grid where blocking lives.
On a high-quality JPEG it's effectively a no-op — correctly, since there's
nothing to fix.

*Limitation:* it works on decoded pixels. Proper deblocking would happen on
the DCT coefficients before decoding; that needs a decoder exposing them.

## chroma.py — *defensible*

JPEGs usually store color at half resolution (4:2:0). Pillow has already
upsampled it by the time we see it, so this stage re-sharpens chroma
against the full-resolution luma rather than truly reconstructing it.
Skipped entirely for 4:4:4 JPEGs, which have nothing to correct. Strength
scales with how hard the chroma channels themselves were quantized.

## tonecurve.py — *exact* (sRGB) / *experimental* (S-curve)

Removing the sRGB transfer function is exact math: it adds no error, it
just stops hiding the 8-bit quantization already present (which is why
`bitdepth.py` comes next).

Cameras also apply their own contrast S-curve ("picture style") before
encoding. That curve is camera-specific and **not recoverable from the
JPEG**; `--s-curve` applies a mild generic inverse and is off by default.
This is the biggest remaining quality lever and the main target for
learned reconstruction (see [invisp.md](invisp.md)).

## colormatrix.py — *exact*, by not pretending

There's no real sensor to invert to, and inventing a "sensor gamut" would
be fabrication. Instead the DNG declares its native space to be what the
data honestly is — linear sRGB — with the exact standard matrix, so raw
editors render colors correctly. `--gamut-widen` is an experimental *look*,
off by default.

## bitdepth.py — *defensible, bounded*

Linearizing 8-bit data stretches shadow steps wide apart — exactly where
you'd want headroom. Two mitigations, both redistributing existing error
rather than inventing detail:

- **Dither**: triangular noise sized to the local quantization step, full
  strength only in flat areas (where banding would show), 25% elsewhere.
- **Deband**: smoothing in flat regions, **clamped to ±1 source quantization
  step per pixel** — banding is made of steps that size, so it can smooth
  banding but physically cannot remove real detail.

The clamp matters: an earlier unclamped version, which also judged
flatness in linear space (where shadows look artificially flat), removed
~40% of measured detail from a real photo. Now every pixel stays within ±1
level of the source JPEG; on a shadow gradient the distinct tonal levels go
from 26 to 338.

## dng_writer.py

See [dng-format.md](dng-format.md).

## The optional learned path

`--invisp` replaces deblock/chroma/tonecurve with the InvISP invertible
network — see [invisp.md](invisp.md).
