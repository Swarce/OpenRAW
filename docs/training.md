# Training InvISP on FiveK — quickstart

The full workflow, start to finish. The upstream scripts were patched only
where they were actually broken (listed below), not rewritten for style.
An actual training step has not yet been run end to end — see "What's
verified vs not" at the bottom.

## 1. Get the data

Download MIT-Adobe FiveK from Adobe/MIT (the full set, or just the
subset in `data/`). Worth knowing: `train.py` (below) only ever trains ONE camera at a time, and
out of the box only has train/test splits for the same two cameras
`canon.pth`/`nikon.pth` already cover (Canon EOS 5D, Nikon D700) — see
`data/README.md`. Downloading the full 5,000-image set is still useful:
it's a strict superset, and if you ever want to train a third camera
model later, you'd build your own `<CAMERA>_train.txt`/`_test.txt` list
(same bare-filename-per-line format as the existing ones) from whatever
other camera subsets are in the full set — not wired up here, but the
data would already be on disk if you want to go there.

Either way, `data_preprocess.py` (already vendored, see `data/README.md`)
needs the DNGs sitting at `data/NIKON_D700/DNG/*.dng` and
`data/Canon_EOS_5D/DNG/*.dng` specifically — point your full-dataset
download there, or symlink the relevant files in, matching those two
folder names exactly (train.py's `--camera` flag and the dataset loader
both key off them).

## 2. Preprocess: DNG -> training pairs

```bash
pip install ".[dataprep]"
cd data
python3 data_preprocess.py --camera NIKON_D700
python3 data_preprocess.py --camera Canon_EOS_5D
cd ..
```

Writes `data/<camera>/RAW/*.npz` (demosaiced RAW + white balance) and
`data/<camera>/RGB/*.jpg` (rawpy's own default render — see
`data/README.md`'s caveat about this not being the real in-camera JPEG).

Remember the real bug noted in `data/README.md`: Canon's black-level
subtraction is dead code in this script (`'Canon EOD 5D'` typo), left
unpatched deliberately since it's a training-data-content decision, not
an environment fix. Decide if you want to fix that string before running
this, or match what `canon.pth` itself was (apparently unintentionally)
trained on.

## 3. Train

```bash
pip install ".[training]"
python3 train.py --task my_nikon_run --camera NIKON_D700 --gamma --aug
```

`--gamma` matches how `canon.pth`/`nikon.pth` were trained (see
`invisp_bridge.py`'s docstring — the gamma-compression detail that
module already has to undo at inference time). `--aug` enables upstream's
random crop/flip/rotate augmentation. Useful flags from `config/config.py`
and `train.py` itself: `--debug_mode` (loads only 10 images — a real
smoke test, not a toy), `--resume` (continues from
`<out_path>/<task>/checkpoint/latest.pth`), `--batch_size`, `--lr`,
`--loss` (L1/L2).

Checkpoints land at `./exps/<task>/checkpoint/latest.pth` (every epoch)
and `./exps/<task>/checkpoint/<epoch>.pth` (every 10 epochs). Once you
have one you like, copy it to `pretrained/` and point
`invisp_bridge.py`/`cli.py --invisp` at it the same way as the stock
checkpoints — nothing else in the inference path needs to change, it was
already written generically against "a checkpoint matching this
architecture," not specifically the stock weights.

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

Verified so far (torch installed, no GPU):
- Every import in `train.py`'s chain resolves correctly, including the
  patched paths — confirmed by running it and watching it fail exactly
  and only at the explicit CUDA check, nothing earlier or cryptic.
- `DiffJPEG` (the differentiable JPEG simulator used in the training
  loss) actually constructs and runs a forward pass correctly on CPU.
- `dataset/FiveK_dataset.py` imports cleanly post-patch (the imread fix).

NOT verified yet, because it needs a real GPU and downloaded data:
- An actual training step (forward + backward + optimizer step) has
  never run.
- `FiveKDatasetTrain`/`FiveKDatasetTest`'s `__getitem__` (the actual
  `.npz`/`.jpg` loading and augmentation) has never run against real
  preprocessed data.
- Multi-epoch training stability, loss curves, checkpoint quality —
  none of that can be assessed without actually training.

First thing to do when training: run with `--debug_mode`
first (10 images, fast) to confirm the whole pipeline actually executes
end to end on your machine before committing to a full run.
