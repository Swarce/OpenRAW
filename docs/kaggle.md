# Training on Kaggle (free GPUs, unattended)

`kaggle/openraw_kaggle.ipynb` trains InvISP on Kaggle's free GPUs across as
many 12-hour sessions as it takes, resuming automatically. One notebook, two
modes picked from what's attached to it.

Kaggle limits it's built around: **12 h per GPU session**, **~30 GPU hours per
week** (TPU quota is separate, but InvISP isn't ported to TPUs), **20 GB** of saved output per run (other disk space is wiped).

## One-time setup

1. **Import the notebook**: Kaggle → Code → New Notebook → File → Import
   Notebook → upload `kaggle/openraw_kaggle.ipynb`.
2. **Internet on**: notebook Settings → Internet (needs a phone-verified
   Kaggle account).
3. **GitHub token** (optional now that the repo is public): to clone with a
   token anyway, create a fine-grained GitHub token with *read-only
   Contents* access to the repo, then in the notebook: Add-ons → Secrets →
   add it named `GITHUB_TOKEN`. It's read at runtime, never shown or saved;
   the notebook removes it from the clone afterwards.
4. Edit the config cell if you like (`CAMERAS`, `DATA_BUDGET_GB`, `EPOCHS`).

## Run 1 — prepare the data (CPU, no GPU quota)

- Settings → Accelerator **None**.
- **Save Version → Save & Run All (Commit)**. It runs in the background:
  downloads the cameras (largest first), preprocesses them into the compact
  format, deletes each DNG right away, and stops adding cameras before the
  data would exceed `DATA_BUDGET_GB` (default 18 GB of the 20 GB Kaggle keeps).
- When it finishes: open the run's **Output** → **New Dataset**.

## Runs 2, 3, … — train (GPU)

- **Add Input** → your new dataset.
- Settings → Accelerator **GPU T4 x2** (or P100 if your account offers it).
  Training uses **every GPU the session has** -- both T4s, roughly 1.7-1.9x
  faster than one -- after first checking that the GPUs can communicate (see
  troubleshooting). To force a single GPU, call `kr.run(..., gpus=1)` in the
  last cell.
- **Save & Run All (Commit)**. It trains until ~11¼ h into the session, stops
  cleanly *between* epochs, and saves checkpoints, `eval.csv` and `best.pth`.
- To continue: **Add Input → this notebook's own latest output** (if it's
  already attached, make sure it points at the newest version), then Save &
  Run All again. It resumes exactly where it stopped.
- `STATUS.txt` in the output says **DONE** or **RUN AGAIN** with progress and
  the best held-out score so far.

`EPOCHS = None` picks the epoch count that matches upstream InvISP's
training amount (~195k steps) for however many images fit the budget.

## More data: RAISE (optional)

RAISE (8,156 Nikon raws; **non-commercial research use only**, see
`NOTICE.md`) is prepared by the same notebook in a separate run, into its own
dataset:

1. Get the CSV from the [RAISE download page](https://loki.disi.unitn.it/RAISE/download.html)
   (pick a package, accept the terms). Upload it as a small Kaggle dataset
   and attach it to a **new copy** of the notebook (its output must not mix
   with the training notebook's).
2. In the config cell set `RAISE_CSV = True` (optionally `RAISE_CAMERAS = ["D90"]`).
   Accelerator **None**, Internet on, Save & Run All.
3. It downloads and preprocesses images until the output reaches
   `DATA_BUDGET_GB` (~1,100-1,400 images in 18 GB), deleting each NEF as it
   goes. If images are left, the log says so and gives the `RAISE_START`
   value for preparing the next part in another copy of the notebook.
4. Output → **New Dataset**, then attach it to the training notebook next to
   the FiveK dataset. Training uses both.

Attach it when starting a **new** `TASK`, not in the middle of a run: more
data changes the test set (so raw PSNR is no longer comparable with earlier
evaluations) and the automatic epoch count.

## What to expect

Measured on the first real run: the 18 GB budget held ~1,070 training images
from the pooled FiveK cameras, which auto-selected 182 epochs; one T4 did
~0.8 s per step, ~14 minutes per epoch, so a 12 h session covered ~45 epochs.
Held-out raw PSNR was 39.1 dB at epoch 35.

**Evaluation cadence.** By default the model is evaluated ~10 times per run
(every 18 epochs for 182), so a session can pass with few or no `[EVAL]`
lines. Each evaluation takes ~30 s; to evaluate more often, pass extra
arguments in the notebook's last cell:
`kr.run(..., extra_train_args=["--eval_every", "6"])`. Every result is also
in `exps/<task>/eval.csv` in the output, which Kaggle keeps even when its log
viewer trims old lines.

**Log size.** Training prints a step line every 50 steps (averaged over
them), so a 12 h session logs around a thousand lines instead of ~50,000.
Kaggle's log viewer gets very slow with long logs; to find an `[EVAL]` line
or an epoch boundary, search for `[EVAL]` or `Epoch 47 time`. Sessions
started before this change keep the per-step output; this applies from the
next session, since the notebook clones the repository at the start.

- On a 16 GB Kaggle GPU, gradient checkpointing turns off automatically (a step
  needs ~6.2 GB), so steps are ~25% cheaper than on a 6 GB laptop GPU.
- Measured on one T4: ~0.8 s per step. With both, each GPU takes its own
  image per step (effective batch 2), so an epoch takes roughly half the steps.
- A checkpoint from a single-GPU session resumes fine on two GPUs, and the
  other way round.
- If the step logs show `data` well above 0 s with two GPUs, the 4 CPUs can't
  prepare images fast enough; that's the next bottleneck to look at.
- The total depends on the GPU Kaggle assigns and how many images fit;
  probably a few sessions spread over one to two weeks of free quota. Each
  session's log prints epoch times and an ETA.
- Use the result: download `best.pth` from the output and run
  `openraw photo.jpg --invisp-checkpoint best.pth`. (Kaggle may save a
  `.pth` with a `.zip` extension — it is already the checkpoint, since a
  `.pth` file is a zip internally; just rename it back.)

## How it's tested

The orchestration is in `kaggle/kaggle_runner.py`, tested on a simulated Kaggle
machine (`tests/test_kaggle_runner.py`). An unmocked rehearsal ran all three
stages with the real 8-block network: prepare → train until the time budget →
resume from the attached previous output to completion. Not tested: Kaggle's
own UI steps and an actual Kaggle GPU session.

## Troubleshooting

**No log output for a long time, GPUs idle.** On some machines, communication
between the two GPUs hangs instead of failing — the first two-GPU session did
exactly this, silently, for hours. The runner now tests GPU-to-GPU
communication before training (`tools/multigpu_selftest.py`, killed after 4
minutes if it hangs), retries with `NCCL_P2P_DISABLE=1` (the usual fix when
GPUs can't talk directly over PCIe), and otherwise trains on one GPU. During
training, a stalled GPU exchange errors out after 15 minutes, and a failed
multi-GPU run is retried on one GPU from the last saved epoch. Output is
unbuffered, so the log shows progress as it happens. Check the GPU usage
graphs in the session panel: idle GPUs with no new log lines mean it's stuck.

**`ERRORED_MOUNTING_DATASET` at session start.** Kaggle couldn't attach an
input — usually the notebook's own output from a version that saved nothing
(a run that was stopped or failed). Remove that input and re-add it, picking
the last version that finished normally (its `STATUS.txt` shows progress). If
it already is that version, it's a Kaggle-side problem: retry.

**`No usable CUDA GPU`.** Check the accelerator setting. Don't install `torch`
through a dependency setting: Kaggle's preinstalled build matches its GPU
drivers, and a different build may not.

