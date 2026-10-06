#!/usr/bin/env bash
# Build Adobe's reference DNG validator (dng_validate) WITHOUT the XMP Toolkit.
#
# Why: libraw is lenient (scans every IFD), while Adobe's DNG SDK -- used by
# Android's Skia, Luminar, and most commercial tools -- is strict. A real bug
# (DNGVersion missing from IFD0) passed libraw but broke those apps; this
# validator caught it immediately. Run it on output before trusting a change
# to dng_writer.py.
#
# The DNG SDK is NOT vendored (Adobe's own license); this clones a public
# mirror at build time. XMP is stubbed out (tools/dng_xmp_sdk_stub.cpp) --
# structural DNG/TIFF validation and full decoding are unaffected.
#
# Usage:  tools/build_dng_validate.sh [build_dir]    (Linux/macOS, needs g++, zlib)
#         then: PSEUDORAW_DNG_VALIDATE=<build_dir>/dng_validate pytest tests/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
B="${1:-$HERE/../.dng_validate_build}"
mkdir -p "$B" && cd "$B"
[ -d dng_sdk ] || git clone --depth 1 https://github.com/yanburman/dng_sdk.git dng_sdk
S=dng_sdk/source; LJ=dng_sdk/libjpeg
(cd "$LJ" && { [ -f jconfig.h ] || ./configure -q >/dev/null; })
for f in $(cd "$LJ" && ls j*.c | grep -v -E 'jmemansi|jmemdos|jmemmac|jmemname|jpegtran|jmemnobs') jmemnobs.c; do
  gcc -O1 -w -c "$LJ/$f" -I"$LJ" -o "lj_${f%.c}.o"; done
FLAGS="-O1 -w -std=gnu++11 -fpermissive -DqLinux=1 -DqMacOS=0 -DqWinOS=0 -DqDNGValidateTarget=1 -DqDNGXMPDocOps=0 -DqDNGXMPFiles=0 -DqDNGThreadSafe=1 -DUNIX_ENV=1 -I$S -I$LJ"
for f in $(ls $S/*.cpp | grep -v dng_xmp_sdk.cpp) "$HERE/dng_xmp_sdk_stub.cpp"; do
  g++ $FLAGS -c "$f" -o "sdk_$(basename "${f%.cpp}").o"; done
g++ -o dng_validate lj_*.o sdk_*.o -lpthread -lz
echo "built: $B/dng_validate"
