"""
pseudoraw.ml — SUPERSEDED. This was an independent NumPy re-
implementation of InvISP's general architecture (Haar invertible
downsampling + affine coupling blocks), built before the actual InvISP
source and pretrained checkpoints were available to this project.

It is no longer the path forward: `third_party/invisp/` now vendors
InvISP's real PyTorch source, and `pseudoraw/invisp_bridge.py` wires it
up with the real pretrained checkpoints in `pretrained/` (canon.pth,
nikon.pth). Use `cli.py --invisp` for that, not this package.

This package is kept around because its exact-invertibility tests
(tests/test_invnet_invertibility.py) are still a legitimate, checkable
demonstration of why this class of architecture is invertible by
construction -- useful as a from-scratch explanation separate from
reading upstream's actual (more complex: learnable invertible 1x1 convs,
1-vs-2 channel splits rather than Haar-doubling) implementation. It is
not wired into pipeline.py and should not be extended further; put new
work into invisp_bridge.py / third_party/invisp/ instead.

Original docstring, for reference:

Architecture credit, in full: this package's design (Haar-wavelet
invertible downsampling feeding a stack of affine coupling blocks) is
directly based on:

    Yazhou Xing*, Zian Qian*, Qifeng Chen.
    "Invertible Image Signal Processing." CVPR 2021.
    https://github.com/yzxing87/Invertible-ISP
    https://arxiv.org/abs/2103.15061
    (MIT License)

InvISP's own README credits its invertible-block design to:

    Mingqing Xiao, Shuxin Zheng, Chang Liu, Yaolong Wang, Di He,
    Guolin Ke, Jiang Bian, Zhouchen Lin, Tie-Yan Liu.
    "Invertible Image Rescaling." ECCV 2020.
    https://github.com/pkuxmq/Invertible-Image-Rescaling

and its differentiable JPEG simulator (not yet implemented in this
package -- see README backlog) to:

    https://github.com/mlomnitz/DiffJPEG

This package is an independent NumPy re-implementation of the general
*architecture* described in those works (Haar invertible downsampling +
affine coupling blocks) -- it does not copy InvISP's source code. It is
NOT trained, and the coupling-block subnets here are small fixed/random
channel-wise MLPs that exist only to prove the architecture's exact-
invertibility property numerically (see tests/test_invnet_invertibility.py).
Producing actually-useful output requires training the subnets against
paired RAW/JPEG data (the next milestone -- see README), which needs
PyTorch and real compute this scaffold does not assume.
"""

from .invnet import InvISPLiteNet

__all__ = ["InvISPLiteNet"]
