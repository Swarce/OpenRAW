# Training InvISP on FiveK — quickstart

The full workflow, start to finish. The upstream scripts were patched only
where they were actually broken (listed below), not rewritten for style.
The data pipeline and a real training step (on CPU) are covered by tests;
full GPU training runs are not yet verified — see the bottom of this page.

## Quick start

> **Windows / macOS + NVIDIA GPU:** install PyTorch's CUDA build first —
> `pip install torch` from PyPI gives a **CPU-only** build there, which can't
> see your GPU even though it imports fine:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
> ```
> New Pythons (3.13/3.14) need a current CUDA index like `cu130`; older ones
> (`cu118`, `cu121`, ...) have no wheels for them. Linux gets CUDA by default.
> `train.py` diagnoses this if it can't use your GPU.

```bash
pip install -e ".[training]"            # torch, rawpy, colour-demosaicing
python train.py --list-cameras           # every FiveK camera + image count
python train.py --task d70 --camera "Nikon D70" --download --gamma --aug
```

`train.py` resolves the camera, downloads **only that camera's** missing
DNGs from MIT's server, preprocesses them into training pairs, writes the
train/test lists, then trains. Re-running skips everything already done.

- **Everything you've downloaded**: `--all-downloaded` scans
  `data/fivek/raw/`, maps each `<Make_Model>` folder back to its FiveK
  camera, and pools all of them into one model, using whatever DNGs are
  present (partly downloaded cameras train on what's there):

  ```bash
  python train.py --task all_cameras --all-downloaded --gamma --aug
  ```
- **Save disk**: add `--delete-dngs` to delete each DNG once its training
  pair is written. Downloads and preprocessing then run in small batches, so
  peak usage stays at a few hundred MB rather than a whole camera's DNGs.
  Cameras stay detectable by `--all-downloaded` after their DNGs are gone.
- **Several specific cameras**: repeat `--camera` (or use commas) to pool them into
  one model — e.g. `--camera "Nikon D700" --camera "Canon EOS 5D"`. A
  multi-camera model is arguably the more useful one for OpenRAW, since a
  JPEG's source camera is often unknown.
- **No GPU here?** `--prepare-only` downloads and preprocesses, then exits —
  prepare on one machine, train on another (copy `data/`).
- Names are matched loosely: `"D70"`, `"Nikon D70"` and `"NIKON_D70"` all
  work; ambiguous ones are reported with suggestions.
- `--debug_mode` loads 10 images per camera: a fast smoke test of the whole
  pipeline before a real run.

The standalone tools work too: `data/fivek_download.py` (download by camera,
optionally expert TIFFs) and `data/fivek_prepare.py --camera ... --download`.

## Data layout

```
data/fivek/raw/<Make_Model>/<name>.dng   downloaded originals, e.g. data/fivek/raw/Canon_EOS_10D/
                                         (fivek_download.py's layout with --out data/fivek;
                                         deletable after preprocessing)
data/<CameraDir>/RAW/<name>.npz       sensor mosaic (compact, lossless) + white balance + levels
data/<CameraDir>/RGB/<name>.jpg       rendered training target
data/<CameraDir>_train.txt, _test.txt
data/fivek/_metadata/*.json           FiveK camera/split metadata (cached)
```

DNGs left in the older `data/<CameraDir>/DNG/` location are still found.
If a split list names images that were never prepared, the loader skips
them and reports how many.

Splits: InvISP's two cameras (Nikon D700 → `NIKON_D700`, Canon EOS 5D →
`Canon_EOS_5D`) keep InvISP's published lists, so results stay comparable
with the shipped `canon.pth`/`nikon.pth`. Every other camera uses FiveK's
official split (train + validation → train, test → test).

## Preprocessing: what changed from upstream, and why

Upstream's `data/data_preprocess.py` was written for InvISP's two cameras.
Three assumptions in it silently break every other camera, so
`data/fivek_prepare.py` reads them from each DNG instead:

| | upstream | now |
|---|---|---|
| Bayer pattern | hardcoded RGGB | read from the file (RGGB/GRBG/BGGR/GBRG); a wrong pattern swaps colors |
| black level | never subtracted | subtracted per CFA channel |
| white level | 4095 for Canon EOS 5D, 16383 for all others | stored per image; the loader normalizes by it |

Also new: demosaiced values are clipped to the white level (bilinear
demosaicing overshoots to ~1.5× at image borders, physically impossible
values upstream also produced), non-Bayer sensors are skipped rather than
mis-decoded, and pairs are written atomically so an interrupted run never
leaves half a pair.

**Canon EOS 5D note:** because black level is now subtracted, Canon data
prepared this way differs from what `canon.pth` was trained on (upstream's
subtraction was dead code — see `data/README.md`). To reproduce upstream
exactly, preprocess with `data/data_preprocess.py`; the loader still reads
those files with upstream's original normalization.

**Expert edits** (FiveK's retouched TIFFs) can be downloaded with
`data/fivek_download.py --experts`, but aren't used as training targets:
they're artistic retouches in ProPhoto RGB, while OpenRAW's input is camera
JPEGs, so a model trained to invert them would learn the wrong mapping.

## Storage format

Training pairs store the **sensor mosaic** (one value per pixel, black-level
subtracted, rotated upright) as four lossless-JPEG colour planes, and are
demosaiced at load time — only the 256 px crop each training step uses, so
it's cheap and gives exactly what a full-frame demosaic would. For an 18 MP
photo that's **~18 MB instead of ~216 MB** for upstream's demosaiced float32
arrays (format details: `dataset/mosaic_store.py`).

Pairs in the previous, larger format are shrunk in place automatically the
next time their camera is prepared — no re-download. That conversion is
bit-exact except within 2 px of the image edge, where the old format had
clipped values that can't be recovered.

## Training options

`--gamma` matches how the shipped checkpoints were trained; `--aug` enables
random crop/flip/rotate; also `--batch_size`, `--lr`, `--loss`, `--epochs`
(default 300).

**How long to train.** Upstream's 300 epochs were chosen for one camera
(~650 images, ~195,000 steps). Pooling many cameras at 300 epochs multiplies
that -- ~4,000 images is ~1.2M steps. One 256 px step is ~1.5 TFLOP (measured,
with checkpointing): ~1.2 s on a laptop RTX 3050, so 1.2M steps is ~16 days.
Pick `--epochs` for the step budget you want; the startup log suggests a
value matching upstream's (~195k steps). The learning-rate drops (upstream:
epochs 50 and 80 of 300) scale with `--epochs`, so they always land at the
same fraction of training.

**Evaluation.** Every `--eval_every` epochs (default: ~10 times per run,
plus the last epoch) the model is scored on held-out test images
(`--eval_images`, default 40, spread across cameras; deterministic
`--eval_crop` 512 px centre crops):

- **raw PSNR** -- the real JPEG through the *inverse* network vs the true raw.
  This is what OpenRAW actually does, and it picks `best.pth`.
- **rgb PSNR** -- raw through the forward network vs the JPEG.

Results go to the log (`[EVAL]` lines) and `exps/<task>/eval.csv`; `best.pth`
(weights-only, usable with `openraw --invisp`) always holds the best model so
far, and survives `--resume`. Use the curve to decide when to stop: once raw
PSNR flattens, more epochs are mostly polishing. Per-step training loss is
too noisy at batch size 1 to judge this.

**Speed.** Images are loaded by parallel worker processes (`--workers`,
default: up to 8, one less than your CPU cores). Preparing one 18 MP sample
takes ~0.3-0.5 s of CPU, far longer than the GPU step, so with upstream's
single loader the GPU mostly waited. Each step logs `data` (time spent
waiting for the loader) and `compute` separately -- if `data` stays well
above zero, raise `--workers`. Each epoch logs its time and an ETA.

**Resuming.** Every epoch saves `latest.pth` (weights only, usable with
`openraw --invisp`) and `latest_state.pth` (weights + optimizer +
learning-rate schedule + epoch/step). `--resume` continues exactly where it
stopped. Checkpoints are written atomically, so a crash mid-save can't
corrupt them. `NNNN.pth` snapshots are kept every 10 epochs.

Runs started before `latest_state.pth` existed only have weights: resume
them with `--resume --start_epoch N`, where N is one more than the last
`Epoch:` number in the log. The epoch count and learning-rate schedule
continue from there; only the optimizer's momentum restarts, which settles
within a few hundred steps.

**Memory.** A 256 px training step of the full 8-block network needs ~5.7 GB
of activations (+ ~0.5 GB CUDA context) -- more than a 6 GB GPU. On Windows the
NVIDIA driver then doesn't fail: it silently spills into system RAM, and steps
get 10-100x slower (measured on an RTX 3050 6 GB: 7-21 s per step). So
**gradient checkpointing** is on by default for GPUs under 12 GB
(`--checkpointing auto|on|off`): each block recomputes its activations during
backward, cutting a step to ~2.5 GB for ~30-40% extra compute, with identical
loss and gradients (verified for every parameter). The log shows GPU memory
after the first step and warns if it's nearly full. To make any future
overflow fail loudly instead of crawling: NVIDIA Control Panel > Manage 3D
settings > CUDA - Sysmem Fallback Policy > Prefer No Sysmem Fallback.

## Real fixes applied to get here (not cosmetic — these were blockers)

Same pattern as everywhere else in this project: patch only what's
necessary to run, document inline, never silently.

- **`scipy.misc.imread`** (`dataset/FiveK_dataset.py`): removed from
  scipy years ago, this import fails outright on any current scipy.
  Replaced with an equivalent PIL-based read (the file already imports
  PIL for other things). Dead-simple, behavior-preserving — it was only
  ever used to load a standard RGB JPEG.
- **Import paths** (`train.py`): upstream's repo has `model/` and
  `utils/` as top-level packages; in OpenRAW they live under
  `openraw/third_party/invisp/` instead. Patched the two import lines
  accordingly; `dataset/` and `config/` ARE top-level here (matching
  upstream), so those imports are untouched.
- **GPU auto-select** (`train.py`): upstream shells out to `nvidia-smi`
  and parses its output to pick the GPU with the most free memory --
  reasonable on a known multi-GPU box, but it crashes confusingly
  (`FileNotFoundError`, or an empty-list `np.argmax`) anywhere
  `nvidia-smi` isn't on PATH in exactly the expected form. Now: an
  explicit, clear `RuntimeError` up front if `torch.cuda.is_available()`
  is False (this script trains GPU-only, full stop), and the nvidia-smi
  auto-select itself is wrapped so a failure there falls back to
  whatever `CUDA_VISIBLE_DEVICES`/default device is already set, with a
  warning instead of a crash.
- **Two genuinely dead imports** (`torchvision`, `rawpy` in
  `dataset/FiveK_dataset.py`) left completely untouched in the vendored
  file (neither is actually referenced anywhere in it) — just documented
  in the `[training]` extra so they get installed anyway, since the
  import itself still needs to succeed even if nothing uses them.

## What's verified vs not, stated plainly

Verified by the test suite (offline, with synthetic Bayer DNGs served from a
local HTTP server behind fake FiveK metadata):
- camera selection → download of only the missing files → preprocessing →
  split lists, including resume/skip behaviour and deleted-DNG handling;
- CFA pattern detection for all four Bayer layouts, black/white levels;
- the loader pooling several cameras and normalizing per image;
- one real InvISP forward + backward + optimizer step on CPU.

Not yet verified: a full multi-epoch GPU training run, loss curves, and
checkpoint quality — those need a GPU and the real dataset. Start any real
run with `--debug_mode` first.
