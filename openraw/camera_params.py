"""Camera settings from EXIF: exposure time, f-number, ISO and focal length.

The EXIF-conditioned model (train.py --exif_cond, after ParamISP, Kim et al.,
CVPR 2024) reads these four values, which every camera JPEG carries and which
say a lot about how the camera processed the photo (high ISO -> stronger noise
reduction, long exposures -> darker scenes pushed up, ...). Also Make and
Model, which pick the per-camera adapter (train.py --cam_lora).

One small reader for both places the values come from:
  - a JPEG (APP1 "Exif" segment) -- what users convert;
  - a TIFF-based raw (DNG, NEF, ARW, CR2, ...) -- the training data. Only the
    first few hundred KB of a raw file are needed, so data/camera_params.py
    can fetch them with an HTTP range request instead of the whole file.

Values that are missing, zero or unreadable come back as None. Never raises
on bad input: a photo without EXIF just converts without conditioning.
"""

from __future__ import annotations

import math
import struct

PARAMS = ("exposure_time", "f_number", "iso", "focal_length")

_TAGS = {0x829A: "exposure_time", 0x829D: "f_number", 0x8827: "iso", 0x8833: "iso_alt", 0x920A: "focal_length",
         0x010F: "make", 0x0110: "model"}
_EXIF_IFD, _SUB_IFDS = 0x8769, 0x014A
_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8}


class _Truncated(Exception):
    """The bytes end before a value we need (read more of the file)."""


def _tiff(buf: bytes, base: int = 0) -> dict:
    """Parse the TIFF structure starting at buf[base:]: IFD0 and the EXIF IFD."""
    if len(buf) < base + 8:
        raise _Truncated
    bo = {b"II": "<", b"MM": ">"}.get(buf[base:base + 2])
    if bo is None or struct.unpack(bo + "H", buf[base + 2:base + 4])[0] not in (42, 0x4F52, 0x5352, 0x55):
        return {}  # not TIFF (0x4F52/0x5352: Olympus ORF; 0x55: Panasonic RW2)
    out: dict = {}

    def need(off, n):
        if base + off + n > len(buf):
            raise _Truncated

    def value(typ, cnt, raw, off_field):
        size = _SIZES.get(typ, 1) * cnt
        if size <= 4:
            data = raw[:size]
        else:
            off = struct.unpack(bo + "I", off_field)[0]
            need(off, size)
            data = buf[base + off: base + off + size]
        if typ == 2:
            return data.split(b"\0", 1)[0].decode("latin-1", "replace").strip()
        if typ in (3, 8):
            return struct.unpack(bo + ("H" if typ == 3 else "h") * cnt, data)
        if typ in (4, 9, 13):
            return struct.unpack(bo + ("I" if typ != 9 else "i") * cnt, data)
        if typ in (5, 10):
            v = struct.unpack(bo + ("I" if typ == 5 else "i") * (2 * cnt), data)
            return tuple(v[i] / v[i + 1] if v[i + 1] else 0.0 for i in range(0, len(v), 2))
        return None

    def ifd(off, depth=0):
        need(off, 2)
        n = struct.unpack(bo + "H", buf[base + off: base + off + 2])[0]
        need(off + 2, 12 * n)
        for i in range(n):
            e = base + off + 2 + 12 * i
            tag, typ, cnt = struct.unpack(bo + "HHI", buf[e:e + 8])
            field = buf[e + 8:e + 12]
            if tag == _EXIF_IFD and depth < 2:
                ifd(struct.unpack(bo + "I", field)[0], depth + 1)
            elif tag in _TAGS and _TAGS[tag] not in out:
                v = value(typ, cnt, field, field)
                if v not in (None, (), ""):
                    out[_TAGS[tag]] = v if isinstance(v, str) else v[0]

    ifd(struct.unpack(bo + "I", buf[base + 4: base + 8])[0])
    return out


def _jpeg(buf: bytes) -> dict:
    """Find the APP1 Exif segment of a JPEG and parse its TIFF block."""
    i = 2
    while i + 4 <= len(buf):
        if buf[i] != 0xFF:
            return {}
        marker = buf[i + 1]
        if marker in (0xD9, 0xDA):  # end of image / start of scan: no EXIF before the image data
            return {}
        length = struct.unpack(">H", buf[i + 2:i + 4])[0]
        if marker == 0xE1 and buf[i + 4:i + 10] == b"Exif\0\0":
            if i + 2 + length > len(buf):
                raise _Truncated
            return _tiff(buf[: i + 2 + length], base=i + 10)
        i += 2 + length
    raise _Truncated


def _clean(raw: dict) -> dict:
    out = {}
    for k in PARAMS:
        v = raw.get(k)
        if k == "iso" and not v:
            v = raw.get("iso_alt")
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = None
        out[k] = v if v is not None and math.isfinite(v) and v > 0 else None
    for k in ("make", "model"):
        if isinstance(raw.get(k), str) and raw[k]:
            out[k] = raw[k]
    return out


def from_bytes(buf: bytes) -> dict:
    """Camera settings from the start of a JPEG or TIFF-based raw file.
    Raises _Truncated (a ValueError subclass would hide it) if more bytes are
    needed -- see data/camera_params.py; returns all-None for anything else."""
    if buf[:2] == b"\xff\xd8":
        return _clean(_jpeg(buf))
    return _clean(_tiff(buf))


def from_file(path) -> dict:
    """Camera settings from a JPEG or raw file on disk (all None if unreadable)."""
    try:
        with open(path, "rb") as f:
            head = f.read(1 << 20)
            try:
                return from_bytes(head)
            except _Truncated:
                return from_bytes(head + f.read())
    except (OSError, _Truncated, struct.error):
        return {k: None for k in PARAMS}


def from_exif_fields(fields: dict) -> dict:
    """Camera settings from decode.load_jpeg()'s exif_fields (exif_transfer.extract_exif)."""
    raw = {}
    groups = [g for g in (fields or {}).values() if isinstance(g, dict)]
    for tag, key in _TAGS.items():
        for group in groups:
            v = group.get(tag)
            if v is None:
                continue
            if isinstance(v, (tuple, list)):
                if not v:
                    continue
                v = v[0]
            raw[key] = float(v) if not isinstance(v, str) else v.strip("\0 ")
            break
    return _clean(raw)


def as_vector(params: dict | None) -> list[float]:
    """The four settings in PARAMS order, NaN where unknown (what the model takes)."""
    params = params or {}
    return [float(params[k]) if params.get(k) else float("nan") for k in PARAMS]
