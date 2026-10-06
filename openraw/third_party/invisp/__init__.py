"""
Vendored subset of yzxing87/Invertible-ISP (CVPR 2021), MIT License.
See LICENSE in this directory (upstream's own license, preserved verbatim)
and /NOTICE.md at the repo root for full attribution and what was changed.

Vendored as-is: model/ (model.py, modules.py, utils.py, loss.py) and
utils/ (JPEG.py, JPEG_utils.py, compression.py, decompression.py,
commons.py) -- the differentiable-JPEG simulator and the InvISPNet
architecture itself, unmodified except for this __init__.py and the
addition of utils/__init__.py (upstream's utils/ wasn't a package; ours
needs to be, to live inside openraw without colluding with any other
top-level "utils" import).

NOT vendored: dataset/, config/, train.py, test_rgb.py, test_raw.py,
cal_metrics.py -- these are FiveK-dataset-specific training/eval
scripts with heavy extra dependencies (rawpy, torchvision, a deprecated
scipy.misc.imread). openraw/invisp_bridge.py is OUR OWN new code that
replaces their role for single-image inference, reusing decode.py
instead of their dataset loader.
"""
