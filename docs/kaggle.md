# Training on Kaggle (free GPUs, unattended)

`kaggle/openraw_kaggle.ipynb` trains InvISP on Kaggle's free GPUs across as
many 12-hour sessions as it takes, resuming automatically. One notebook, two
modes picked from what's attached to it.

Kaggle limits it's built around: **12 h per GPU session**, **~30 GPU hours per
week**, **20 GB** of saved output per run (other disk space is wiped).

## One-time setup

1. **Import the notebook**: Kaggle → Code → New Notebook → File → Import
   Notebook → upload `kaggle/openraw_kaggle.ipynb`.
2. **Internet on**: notebook Settings → Internet (needs a phone-verified
   Kaggle account).
3. **GitHub token** (the repo is private): create a fine-grained GitHub token
   with *read-only Contents* access to the repo, then in the notebook: Add-ons
   → Secrets → add it named `GITHUB_TOKEN`. It's read at runtime, never shown
   or saved; the notebook removes it from the clone afterwards.
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
- Settings → Accelerator **GPU P100** (one GPU; with "T4 x2" the second GPU
  would sit idle).
- **Save & Run All (Commit)**. It trains until ~11¼ h into the session, stops
  cleanly *between* epochs, and saves checkpoints, `eval.csv` and `best.pth`.
- To continue: **Add Input → this notebook's own latest output** (if it's
  already attached, make sure it points at the newest version), then Save &
  Run All again. It resumes exactly where it stopped.
- `STATUS.txt` in the output says **DONE** or **RUN AGAIN** with progress and
  the best held-out score so far.

`EPOCHS = None` picks the epoch count that matches upstream InvISP's
training amount (~195k steps) for however many images fit the budget.

## What to expect

- On a 16 GB Kaggle GPU, gradient checkpointing turns off automatically (a step
  needs ~6.2 GB), so steps are ~25% cheaper than on a 6 GB laptop GPU.
- The total depends on the GPU Kaggle assigns and how many images fit;
  probably a few sessions spread over one to two weeks of free quota. Each
  session's log prints epoch times and an ETA.
- Use the result: download `best.pth` from the output, put it in `pretrained/`,
  and run `openraw photo.jpg --invisp`.

## How it's tested

The orchestration is in `kaggle/kaggle_runner.py`, tested on a simulated Kaggle
machine (`tests/test_kaggle_runner.py`). An unmocked rehearsal ran all three
stages with the real 8-block network: prepare → train until the time budget →
resume from the attached previous output to completion. Not tested: Kaggle's
own UI steps and an actual Kaggle GPU session.
