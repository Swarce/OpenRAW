# Training data: MIT-Adobe FiveK, Canon EOS 5D + Nikon D700 subset

This is the exact subset InvISP's own `canon.pth`/`nikon.pth` were trained
and evaluated on — not the full 5,000-image FiveK dataset (47.34 GB).
Vendored from the upstream repo (MIT License, see `/NOTICE.md`): the four
`.txt` URL lists, `data_preprocess.py`, `data_preprocess.sh`.

| Camera        | Total | Train | Test | Est. size |
|---------------|------:|------:|-----:|----------:|
| Canon EOS 5D  |   777 |   650 |  127 |  ~7.4 GB  |
| Nikon D700    |   487 |   414 |   73 |  ~4.6 GB  |
| **Combined**  | **1,264** | **1,064** | **200** | **~11.7 GB** |

Size estimate from FiveK's own known total (47.34 GB / 5,000 images =
9.47 MB/DNG average) × this subset's count — not independently verified
per-file in this environment (no network access to data.csail.mit.edu
from here). Treat it as a solid estimate, confirm actual size once you
start downloading.

## Download

```bash
cd data
bash data_preprocess.sh   # wget -i against Canon_EOS_5D.txt / NIKON_D700.txt
```

This pulls the FULL per-camera set (both train+test — test DNGs are
needed too, for eval) into `./Canon_EOS_5D/DNG/` and `./NIKON_D700/DNG/`.
Want to start smaller first? Just the test split is 200 images / ~1.9 GB:

```bash
cd data
mkdir -p NIKON_D700/DNG Canon_EOS_5D/DNG
wget -P NIKON_D700/DNG -i NIKON_D700_test.txt
wget -P Canon_EOS_5D/DNG -i Canon_EOS_5D_test.txt
```

## Preprocess (RAW DNG -> training pairs)

```bash
pip install "..[dataprep]"      # i.e. pip install ".[dataprep]" from the repo root
cd data
python3 data_preprocess.py --camera NIKON_D700
python3 data_preprocess.py --camera Canon_EOS_5D
```

For each DNG, this writes an `.npz` (bilinearly-demosaiced RAW + camera
white balance) to `<camera>/RAW/` and a `.jpg` (quality 90, rawpy's own
default render of the DNG — see caveat below) to `<camera>/RGB/`.

**Important, stated plainly:** the "JPEG" side of each training pair is
NOT each image's real in-camera JPEG. It's `rawpy.postprocess()`'s own
default demosaic+render of the DNG. That's consistent with what the
existing `canon.pth`/`nikon.pth` checkpoints already learned to invert
(same script produced their training data), so it's not a new problem —
but it does mean the network is learning to invert *rawpy's* rendering
pipeline specifically, not an arbitrary real-world camera JPEG. Worth
remembering if a real photo from one of these cameras doesn't invert as
cleanly as the training pairs would suggest.

## A real bug in this vendored script, left unpatched (unlike modules.py)

Line 50: `if camera_name == 'Canon EOD 5D':` — note "EOD", not "EOS", and
a space instead of an underscore. Every actual invocation of this script
passes `--camera Canon_EOS_5D` (see `data_preprocess.sh` and
`FiveK_dataset.py` in the upstream repo), so this string never matches
and the Canon-specific black-level subtraction
(`raw_img = np.maximum(raw_img - 127.0, 0)`) is dead code — it has
apparently never actually run, including for the official pretrained
`canon.pth`.

Unlike the `torch.qr` fix in `openraw/third_party/invisp/model/modules.py`, this
is NOT patched here: that fix was pure environment-compatibility with a
value immediately overwritten by checkpoint loading, provably harmless.
This one changes what numbers the Canon training data actually contains
(whether a 127-level black-point offset gets subtracted before
demosaicing) -- a real algorithmic choice, not a compatibility shim, and
not one to make silently. Fixing the string means changing the
training distribution slightly from what `canon.pth` itself was (likely
unintentionally) trained on. Worth deciding deliberately, not inheriting
by accident.
