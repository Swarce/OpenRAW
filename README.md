# pseudoraw (POC)

Reconstructs a **linear DNG** from a JPEG: not real sensor RAW data (that's
impossible — see below), but a 16-bit linear reconstruction with deblocking,
chroma refinement, tone-curve inversion, and dithered/debanded bit-depth
expansion, giving raw-capable editors (Lightroom, darktable, Capture One,
RawTherapee) more grading headroom than the baked JPEG had.

## Why this can't be "real RAW recovery," and what it is instead

JPEG throws away information in ways that are genuinely unrecoverable:
8-bit quantization, baked tone curve, chroma subsampling, DCT quantization,
and (for real cameras) a sensor-specific demosaic + color pipeline that's
long gone by the time you have a JPEG. No amount of processing — classical
or ML — gets that back. Any tool claiming to "restore your RAW" from a JPEG
is overselling.

What *is* real and useful: redistributing the error that's already there
instead of hiding it. An 8-bit JPEG pushed hard in post shows ugly banding
and blocking because the baked tone curve and lossy compression concentrated
all its precision in the midtones and threw the rest away. Decompressing
that into a dithered, debanded, deblocked 16-bit linear file doesn't add
back missing highlight/shadow detail, but it does mean the degradation you
already have shows up as fine grain instead of hard edges when you push it —
which is a legitimate, measurable improvement (see `tests/` for a literal
"more distinguishable tonal levels than the 8-bit source" check).

**Every stage is written to say, in its own docstring, exactly how
confident it is** — which steps are exact invertible math (sRGB gamma
removal) vs. defensible-but-approximate signal processing (deblocking,
debanding) vs. openly speculative/experimental and off by default (generic
contrast-curve removal, gamut widening). Read `pseudoraw/*.py` — the
docstrings are as much the documentation of this project's honesty as the
code is of its mechanism.

## Pipeline

```
JPEG
  │  decode.py      — Pillow decode + read the embedded quant tables
  │                    (used to estimate how lossy this specific JPEG was)
  ▼
deblock.py           — edge-aware guided-filter deblocking, strength driven
  │                     by the real quality estimate, not a fixed amount
  ▼
chroma.py             — re-sharpen chroma planes against the luma guide
  │                      (see its docstring for an important limitation:
  │                      Pillow hands us already-upsampled chroma, so this
  │                      is refinement, not true subsampled-plane recovery)
  ▼
tonecurve.py           — exact sRGB EOTF inversion -> scene-linear
  │                       (+ optional, off-by-default generic S-curve undo)
  ▼
colormatrix.py          — NOT a fabricated sensor-gamut inverse. Declares
  │                        the "virtual sensor" native space to honestly be
  │                        linear sRGB, and writes the exact matching matrix.
  ▼
bitdepth.py               — 8->16 bit expansion: amplitude-matched triangular
  │                          dithering + gradient-gated debanding smoothing
  ▼
dng_writer.py               — real Linear DNG tags via tifffile (not a
                               renamed TIFF) — DNGVersion, ColorMatrix1,
                               CalibrationIlluminant1, WhiteLevel/BlackLevel,
                               AsShotNeutral, PhotometricInterpretation=
                               LinearRaw(34892)
```

(`pseudoraw/ml/` — the InvISP-based invertible network — is groundwork
for replacing the deblock/chroma stages above; it is not wired into this
pipeline yet. See "ML groundwork" below.)

## Install & run

```bash
pip install -r requirements.txt   # opencv-contrib-python is required, not
                                   # plain opencv-python -- we need ximgproc

python3 cli.py input.jpg output.dng
```

Options:

```
--no-deblock              disable the deblocking stage
--no-chroma-refine        disable chroma refinement
--chroma-strength FLOAT   default 0.6
--s-curve FLOAT           EXPERIMENTAL generic contrast-curve removal,
                           off by default (0). try 0.1-0.4 if you want it.
--gamut-widen FLOAT       EXPERIMENTAL gamut widening -- a creative look,
                           not a reconstruction step. off by default (0).
--no-dither / --no-deband
```

## Real InvISP integration: `third_party/invisp/` + `pseudoraw/invisp_bridge.py`

This project vendors actual source code from **InvISP (CVPR 2021)** —
not a reimplementation — and wires it up with InvISP's own official
pretrained checkpoints. Full attribution, exactly what was vendored vs.
written fresh, and checkpoint provenance (md5-verified against upstream)
are in [`NOTICE.md`](NOTICE.md). Summary:

- `third_party/invisp/` — vendored, unmodified: `model/model.py`
  (`InvISPNet`, `InvBlock` — the real 1-vs-2-channel affine coupling with
  a *learnable* invertible 1×1 conv per block, not the Haar-doubling
  scheme an earlier version of this README described before the real
  source was available), `model/modules.py`, the differentiable JPEG
  simulator (`utils/JPEG*.py`, `utils/compression.py`,
  `utils/decompression.py`), and upstream's own `LICENSE`.
- `pretrained/canon.pth`, `pretrained/nikon.pth` — InvISP's own official
  checkpoints, one per camera (Canon EOS 5D / Nikon D700, both from
  MIT-Adobe FiveK).
- `pseudoraw/invisp_bridge.py` — **our own new code**, not vendored:
  loads a checkpoint, runs `net(x, rev=True)` on a decoded JPEG, and
  undoes the gamma compression upstream's `--gamma` training flag bakes
  into the "RAW" side of the network (see the module docstring — this
  part is unintuitive and easy to get wrong silently, so it's spelled
  out there in full) before handing scene-linear data to the same
  `bitdepth.py` / `dng_writer.py` stages the classical path uses.

Run it: `python3 cli.py input.jpg output.dng --invisp --invisp-camera NIKON_D700`
(needs `pip install -r requirements-invisp.txt`).

**Update: this has now actually been run**, end to end, on real torch
(2.14.1, CPU) — construct the real `InvISPNet`, load the real
`nikon.pth`, run `net(x, rev=True)` on a decoded JPEG, undo the gamma,
write a compressed Linear DNG. It was not a clean first run: constructing
the model hit `torch.qr`, which current PyTorch has removed outright (not
just deprecated — see `NOTICE.md` for the one-line patch, which doesn't
touch anything `load_state_dict()` doesn't immediately overwrite anyway).
With that fixed, output is finite, non-degenerate, visibly coherent
(see `examples/` for a rendered preview), and pinned by
`tests/test_pipeline.py::test_invisp_path_runs_real_network_and_writes_valid_dng`,
which actually exercises the real network and real checkpoint rather than
mocking around them. `requirements-invisp.txt` still flags `torch.lu`/
`torch.lu_unpack` (used a few lines below the patched call) as a future
risk — deprecated-with-a-warning but functional on 2.14.1, left alone
since there's no reason to patch what isn't broken yet.

One thing worth being honest about even with it running: colors come out
visibly different from the classical path's output — more muted/shifted,
since this is the network's learned approximation of Nikon D700 sensor-
native color response, not an sRGB-preserving transform. That's expected
behavior, not a bug, but it means the two `cli.py` paths are not
drop-in equivalents of each other, just two different approaches to the
same goal.

A separate, now-superseded NumPy reimplementation of InvISP's general
architectural idea (built before the real source was available) still
lives in `pseudoraw/ml/` — see its docstring and `NOTICE.md` for why it's
kept around (its invertibility tests are still a legitimate from-scratch
demo) despite not being the path forward anymore.

## What's genuinely demonstrated by this POC

- A real, valid Linear DNG that round-trips through `tifffile` with
  correct tags (verified: photometric interpretation, color matrix,
  calibration illuminant, white/black level, hue identity of saturated
  test patches survives the full pipeline).
- Measurable deblocking on a low-quality (q15) synthetic JPEG — see
  `examples/deblock_comparison.png`, generated from the test image in
  this repo.
- Measurable increase in distinguishable tonal levels across a gradient
  after dithering (`tests/test_pipeline.py::test_bitdepth_expansion_increases_tonal_resolution`).

## What's explicitly NOT yet done (the honest backlog)

1. **No learned model yet.** `deblock.py` and `chroma.py` are classical
   heuristics, clearly marked with `TODO` blocks describing exactly what
   they'd be replaced with (a small ARCNN-style deblocking net; a decode
   path that exposes true subsampled chroma planes). This was the explicit
   ground-truth-first ordering: get the deterministic pipeline and DNG
   container correct before anything learned sits on top of it, since a
   model trained against a buggy container would learn to compensate for
   the bug.
2. **No per-camera color science.** `colormatrix.py` deliberately does not
   try to guess a sensor's native gamut — see its docstring for why that
   would be fabrication, not reconstruction.
3. **No synthetic-Bayer-mosaic mode.** This POC only writes Linear DNG
   (non-mosaiced). A mode that re-mosaics into a fake CFA pattern so a raw
   converter's own demosaic/denoise engages is a plausible v2 feature, not
   implemented here.
4. **The real InvISP path now runs** (see above) but has only been
   validated on one small synthetic test image on CPU (~36s for 768x512
   -- untested at real photo resolutions or on GPU, and untested on an
   actual camera JPEG rather than a synthetic gradient image). Next real
   step here: run it against real Nikon D700 / Canon EOS 5D photos (ideally
   ones with known ground-truth RAW, e.g. from FiveK itself) and look at
   the result critically, not just confirm it doesn't crash.
5. **Only two cameras.** `canon.pth`/`nikon.pth` cover Canon EOS 5D and
   Nikon D700 only (what InvISP itself trained on). A JPEG from any other
   camera gets reconstructed through one of those two learned color/tone
   behaviors regardless of its real source camera — upstream's own README
   states this limitation plainly, it's not specific to this integration.
6. **No training pipeline of our own yet.** If/when retraining or
   fine-tuning on more cameras becomes the goal, upstream's
   differentiable JPEG simulator (vendored, see `NOTICE.md`) means
   training directly against JPEG-compressed inputs is already available
   in principle — just not wired into a training script here yet. Would
   need MIT-Adobe FiveK (or NUS/RAISE) and a GPU.

## License

MIT (see `LICENSE`). Contributions welcome — this is meant to be a
community project, not a solo tool.
