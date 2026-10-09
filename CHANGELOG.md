# Changelog

All notable changes. Development history before the first release is
summarized; commit messages carry the full detail and measurements.

## [Unreleased] — 0.1.0.dev0

### Fixed
- `tools/raw_pair_eval.py` scored CFA DNGs 4 px out of alignment (LibRaw
  reports the DefaultCrop that removes the CFA padding but doesn't apply it),
  so CFA looked 0.9–2.3 dB worse than linear against the NEX-7 raws. Fixed:
  CFA scores the same as linear. Results and comparison images regenerated.

### Changed
- **Renamed to OpenRAW**: package `pseudoraw` → `openraw`, command
  `pseudoraw` → `openraw`, `PseudoRawPipeline` → `OpenRawPipeline`,
  `PSEUDORAW_DNG_VALIDATE` → `OPENRAW_DNG_VALIDATE`. DNGs now identify as
  `OpenRAW` (Make/Software) and `OpenRAW virtual sensor` (UniqueCameraModel).

### Added — conversion
- **Batch conversion**: files and folders, `-r` recursive with mirrored
  output tree, `--jobs` for parallel files. Re-runs skip finished outputs
  (`--overwrite` to redo), a failing file never stops the batch, and
  interrupted runs never leave truncated DNGs.
- Camera EXIF passthrough (make, model, lens, exposure, ISO, focal length,
  orientation, capture time). Never fabricated when absent; GPS never copied.
- `openraw --version` reports key library versions for bug reports.

### Added — DNG output
- **`--cfa-aa STRENGTH`** (with `--layout cfa`): emulates a camera's optical
  low-pass (anti-aliasing) filter before the Bayer mosaic. At 0.5: about half
  the false-colour moiré on fine detail, a closer demosaic round trip on real
  photos (51.9 → 53.1 dB through AHD), ~15% less edge energy. Off by default.
- **Lossless-JPEG DNG compression** (tiled, 3-component) with a DNG
  LinearizationTable at 12-bit by default: ~50 MB instead of ~108 MB for an
  18 MP photo. `--bit-depth 16/14/12/10`, `--compression none`.
- **`--layout cfa`**: Bayer (RGGB) mosaic DNGs, demosaiced by the raw editor
  like a camera raw; ~18 MB vs ~50 MB for 18 MP. Round trip ~48 dB PSNR
  through libraw's AHD/DCB/PPG on a full-resolution photo (median 42 dB on 14
  downscaled web photos; quality tracks pixel-level detail). 4 px padding +
  DefaultCrop; 2-component lossless-JPEG tiles. Passes Adobe `dng_validate`
  in every mode. Best for photographs; keep linear for graphics.
- `tools/cfa_quality.py`: measure `--layout cfa` round-trip quality on your
  own photos across every libraw demosaic (+ Adobe's reference renderer),
  with zoomed side-by-side crops.
- Multithreaded tile encoding (`--threads`), byte-identical at any count.
- Embedded JPEG preview in the standard preview + SubIFD layout.

### Added — learned reconstruction (InvISP)
- Optional InvISP (CVPR 2021) learned path, `--invisp`, with vendored
  upstream code and official checkpoints.
- **OpenRAW-trained models** in `pretrained/`: `openraw-fivek-e35-best.pth`
  (held-out raw PSNR 39.08 dB), `openraw-fivek-e46-latest.pth` and
  `openraw-fivek-e53-best.pth` (39.47 dB, the run's best), snapshots
  of the first pooled-FiveK training run (stopped at epoch 139 of 182, past its plateau). Upstream's `nikon.pth` /
  `canon.pth` stay alongside.
- **`--invisp-checkpoint PATH`**: use a model you trained (`best.pth`,
  `latest.pth`, or the full-state `latest_state.pth`).

### Added — training
- **FiveK training by camera**: `train.py --camera "Nikon D70" --download`
  fetches only that camera's DNGs, preprocesses them, and writes the split
  lists; `--list-cameras`, `--prepare-only`, multi-camera pooling. Built on
  `data/fivek_download.py`, sharing its `data/fivek/raw/<Make_Model>/`
  layout; `--all-downloaded` trains on every camera found there.
- **RAISE as a second training source** (`data/raise_prepare.py`): reads
  the CSV from RAISE's download page, downloads NEFs (by camera, category or
  range; resumable, `--delete-nefs`), preprocesses them like FiveK DNGs into
  `RAISE_<Make_Model>/` folders with a fixed hash-based test split. Picked up
  by `--all-downloaded` and the Kaggle runner (`RAISE_CSV = True` prepares it
  within the output budget, in parts). Non-commercial research use only.
- Compact training storage: the sensor mosaic as 4 lossless-JPEG planes,
  demosaiced per training crop at load time: ~18 MB instead of ~216 MB per
  18 MP photo. Existing pairs are shrunk in place. `--delete-dngs` removes
  DNGs after preprocessing, in small batches to keep peak disk use low.
- Parallel data loading (`--workers`), exact resume from full-state
  checkpoints (`latest_state.pth`: optimizer, LR schedule, epoch/step) with
  atomic saves, `--start_epoch` for weights-only checkpoints, data/compute
  timing and per-epoch ETA, `--epochs`, `--device`.
- Held-out evaluation (`--eval_every`, `--eval_images`, `--eval_crop`): raw
  PSNR (real JPEG → raw, the OpenRAW direction) and rgb PSNR on test-split
  centre crops, logged and written to `eval.csv`; `best.pth` keeps the best
  model by raw PSNR across resumes.
- Gradient checkpointing (`--checkpointing auto`, on below 12 GB): fits the
  8-block network on 6 GB GPUs, identical gradients.
- `--time_limit_hours`: stop cleanly between epochs before a session limit.
- **`openraw-fivek-raise-e14-best.pth`**: the FiveK + RAISE run's best so
  far (from `e53-best`, ~1,930 images from 5 cameras, weight averaging).
  Highest of all models on the shared held-out set (39.01 dB vs 38.55 for
  `e53-best`, 37.48 for `nikon.pth`), and 39.42 dB on the Sony NEX-7 real-raw
  test (`e53-best` 38.95, `nikon.pth` 40.32). Now the suggested checkpoint.
  The model table in `docs/invisp.md` scores every model on the same test set.
- **Ground truth against real raws** (`tools/raw_pair_eval.py`): converts
  the JPEG of RAW+JPEG pairs with every method and layout, renders each DNG
  and the camera's real raw with the same LibRaw settings, aligns them and
  measures PSNR (as converted, gain-matched, display). First results, four
  Sony NEX-7 photos: InvISP 39–40 dB vs classical 36.5 dB gain-matched;
  upstream `nikon.pth` ahead of `openraw-fivek-e53-best`. Comparison images
  in `examples/` replace the earlier Coolpix parrot crop.
- **`tools/compare_models.py`**: scores any set of checkpoints on exactly
  the held-out images, crops and metric of training's evaluation, with a
  per-camera breakdown, into `compare.md` / `compare.csv`. The Kaggle
  notebook runs it after every training session, or alone with
  `COMPARE_ONLY = True`. `train.py`'s camera resolution and test-image
  selection are shared functions now, so the two can't drift apart.
- Kaggle: `INIT_FROM` starts a new `TASK` from another run's `best.pth`
  (e.g. a bigger dataset continuing from the FiveK-only model), with its own
  schedule, test set and best score.
- `--invisp-checkpoint` on a full-state `latest_state.pth` uses its weight
  average when it has one.
- **`--amp` mixed precision** (experimental, off by default): float16 only in
  the dense sub-networks, float32 for the invertible coupling, JPEG
  simulation, losses and evaluation; loss scaling, kept consistent across
  GPUs. `tools/amp_check.py` (Kaggle: `AMP_CHECK_ONLY = True`) measures the
  speed-up and accuracy cost on the actual GPU before using it.
- **Weight averaging (`--ema`, default 0.999)**: evaluation and the saved
  `best.pth` / `latest.pth` / `NNNN.pth` use an exponential moving average of
  the weights, which doesn't jitter between snapshots the way the live
  weights do late in training (held-out raw PSNR moved 39.47 → 38.80 dB
  between two snapshots). Resumes continue the average; older runs start it
  on resume.
- `--log_every N` (default 50): one step line per N steps with averaged
  losses, and mean losses on each epoch line. Per-step printing made long
  runs' logs (~50,000 lines per Kaggle session) slow to view.
- **`--gpus N`** multi-GPU data parallelism (one process per GPU,
  hand-averaged gradients, disjoint sampling; rank 0 evaluates, saves and
  decides when to stop). Exactly equivalent to batch size N.
- **Kaggle training** (`kaggle/openraw_kaggle.ipynb` + `kaggle/kaggle_runner.py`):
  one notebook that prepares a FiveK dataset within Kaggle's 20 GB, then
  trains across 12 h GPU sessions on every GPU the session has, stopping
  cleanly and resuming automatically. First real run: ~1,070 pooled images,
  ~0.8 s/step on a T4, held-out raw PSNR 39.1 dB at epoch 35.

### Added — project
- **Logo**: aperture with a Bayer RGGB sensor tile in the lens opening,
  wordmark in Saira; light/dark SVGs, mono mark, avatar and social
  preview PNGs, all generated by `tools/logo/make_logo.py`.
- Packaging: `pyproject.toml`, `openraw` command, extras `[invisp]`,
  `[dataprep]`, `[training]`, `[dev]`.
- CI on Linux/Windows/macOS, with Adobe's `dng_validate` and a clean-venv
  wheel install; `tools/build_dng_validate.sh` to run the validator locally.
- CI: a PyTorch (CPU) job, so InvISP, training-loop, multi-process and
  Kaggle-runner tests run (they were skipped in every other job); job
  timeouts; docs-only changes skip the test jobs and get a Markdown link
  check (`tools/check_links.py`) instead. Dependabot for GitHub Actions.
- Dataset credits: FiveK and RAISE citations (BibTeX) and license terms in
  `NOTICE.md`, summarized in the README; the trained weights are marked for
  non-commercial research use, since both datasets are research-only.
- Repository: `.gitattributes` (LF line endings on every OS, binary files
  marked, vendored code excluded from language statistics), `.gitignore`
  covering training data/outputs and editor files, issue forms and a PR
  template.

### Fixed — conversion and DNG output
- DNGs rejected by Adobe-SDK readers (Android/Skia, Luminar): `DNGVersion`
  and other identity tags had ended up outside IFD0.
- Privacy: DNG ImageDescription embedded the full source path (e.g. your
  user account name and folder layout); now only the filename.
- Debanding softened whole images (~40% measured detail loss); its
  correction is now clamped to one source quantization step.
- Deflate-compressed DNGs were unreadable by libraw (replaced by
  lossless JPEG).
- Spurious `ExtraSamples` tag and a `StopIteration` crash, both caused by
  tifffile-version-dependent handling of LinearRaw; output is now
  byte-identical across tifffile versions.
- Missing `imagecodecs` crashed the whole run; it's now a declared
  dependency and the writer falls back to uncompressed output if absent.
- Decoder read quantization tables/EXIF after color conversion, losing them
  for non-RGB JPEGs.
- Rational EXIF values (e.g. f-number) were corrupted when given as plain
  integers.

### Fixed — InvISP and training
- The first two-GPU Kaggle session produced no output for hours — consistent
  with GPU-to-GPU communication hanging. The Kaggle runner now self-tests
  communication first (`tools/multigpu_selftest.py`), retries with
  `NCCL_P2P_DISABLE=1`, then falls back to one GPU; training collectives time
  out after 15 minutes; a failed multi-GPU run is retried on one GPU; and
  training output is unbuffered so logs stream live.
- `--invisp` ran the whole photo through the network at once, needing ~1.8 GB
  per megapixel (~32 GB for 18 MP) and getting killed on ordinary machines.
  It now runs in 512 px tiles with a 96 px overlap (`--invisp-tile`): same
  result to float rounding, ~1 GB peak at any size.
- InvISP checkpoints loaded with `strict=False`, so a file that didn't match
  (e.g. a full-state checkpoint) silently loaded 0 of 280 weights and ran a
  random network. Missing weights are now an error.
- Training preprocessing assumed InvISP's two cameras: hardcoded RGGB
  pattern, no black-level subtraction, hardcoded white levels. Now read per
  file; demosaic border overshoot (~1.5× white) clipped.
- Training on 6 GB GPUs crawled (7-21 s/step on an RTX 3050): a 256 px step
  needs ~6.2 GB, and Windows silently spills the overflow into system RAM.
  Fixed by gradient checkpointing; the log reports GPU memory and warns when
  it's nearly full.
- Upstream's LR milestones (epochs 50/80 of 300, tuned for one ~650-image
  camera) now scale with `--epochs`; the log suggests an `--epochs` giving
  upstream's training budget. cuDNN autotuning on CUDA.
- `train.py` reported "no GPU" when the real cause was a CPU-only PyTorch
  build (PyPI's default on Windows/macOS); it now says which of CPU build /
  missing driver / old driver it is, with the fix.
- Upstream's GPU auto-select shelled out to `grep`/`rm` (absent on Windows,
  so it failed every run) and set `CUDA_VISIBLE_DEVICES` after CUDA was
  already touched; now queries `nvidia-smi` directly and selects by index.
- `train.py` ran all setup at import time, which breaks multiprocessing's
  spawn mode (always used on Windows).
- colour-science's "matplotlib not available" warning printed once per
  preprocessing worker; silenced (only that message).
