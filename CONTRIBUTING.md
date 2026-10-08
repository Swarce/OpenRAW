# Contributing to OpenRAW

Thanks for helping! OpenRAW is young, so bug reports from real photos and
real editors are as valuable as code.

## Reporting a bug

Please include:

- the output of `openraw --version` (almost every bug so far has depended
  on a library version — tifffile, imagecodecs, OpenCV)
- the exact command you ran
- which app failed to open the DNG, and its exact error message
- if you can share it: the source JPEG, or at least its dimensions and
  camera model

"Opens in X but not in Y" reports are especially useful — different raw
readers are strict about different things.

## Development setup

```bash
git clone https://github.com/Swarce/OpenRAW.git
cd OpenRAW
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # add ,training for the PyTorch/InvISP tests
python -m pytest
```

Tests needing optional pieces skip cleanly when those are missing: PyTorch +
torchvision (InvISP, training-loop and multi-process tests), `rawpy` (real
libraw decode tests), and Adobe's `dng_validate` (see below). Training tests
use a small network on the CPU, so no GPU is needed; multi-GPU tests run two
CPU processes over gloo. The training-loop tests take a few minutes.

## The one rule for DNG changes

**A change to how DNGs are written isn't verified until it passes the
strictest reader** — for both layouts, linear and CFA. Every real DNG bug
in this project's history passed a lenient reader first — tifffile reads
back whatever it wrote, and libraw
scans all IFDs, so neither catches what Adobe-SDK-based apps (Lightroom,
Luminar, Android) reject. Build Adobe's validator once:

```bash
tools/build_dng_validate.sh                     # Linux/macOS; needs g++ and zlib
export OPENRAW_DNG_VALIDATE=$PWD/.dng_validate_build/dng_validate
python -m pytest                                # Adobe tests now actually run
```

And compare decoded pixels, not just "it opened". See
[docs/dng-format.md](docs/dng-format.md) for the structure, the reasoning
behind it, and the dead ends already explored.

## Principles the code follows

- **Never fabricate.** Missing EXIF stays missing; no invented sensor
  profiles, noise models, or lens data. If the output claims something, the
  input supported it.
- **Say how much to trust each stage.** Modules label themselves exact,
  defensible, or experimental (see [docs/how-it-works.md](docs/how-it-works.md)).
  Experimental features are off by default.
- **Measure, don't assume.** Quality claims come with numbers (sharpness,
  error in source quantization steps, file size), and a test pins them.
- **Patch vendored code minimally and visibly.** Changes under
  `openraw/third_party/` get a `PATCHED (OpenRAW, not upstream)`
  comment and a note in [NOTICE.md](NOTICE.md). Fix what's broken; leave
  the rest as upstream wrote it.
- **Privacy by default.** GPS is never copied, nor the source file's folder path.
- **Fail loudly.** A wrong input, missing dependency or mismatched
  checkpoint is a clear error, never a silent fallback to garbage.

## Pull requests

- Keep each PR focused; include tests for behavior changes and bug fixes
  (a regression test that fails without the fix is ideal).
- CI runs the suite on Linux, Windows and macOS, a job with Adobe's
  `dng_validate`, a job with PyTorch (CPU) so the InvISP and training tests
  actually run, and a clean-venv install of the built wheel. Docs-only
  changes skip those and get a Markdown link check instead — run it locally
  with `python tools/check_links.py`.
- Explain *why* in the commit message, not just what — especially for
  anything touching DNG structure.
- Don't commit sample photos you don't have the rights to share.

## License

Contributions are released under the project's MIT license.
