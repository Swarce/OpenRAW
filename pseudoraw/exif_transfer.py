"""
exif_transfer.py — carry real camera metadata (lens, aperture, exposure,
ISO, focal length, camera make/model, orientation, capture time) from the
source JPEG into the output DNG. Missing fields stay missing: a
point-and-shoot's auto-mode JPEG that has no lens info gets a DNG with no
lens info, not a fabricated one. Never invented, only copied.

What's copied, and why these specifically: a curated allowlist of
standard baseline-TIFF and Exif-sub-IFD tags that describe the capture
(not the file) -- Make/Model/Orientation/DateTime live in IFD0;
ExposureTime/FNumber/ISO/FocalLength/LensModel/LensMake/
FocalLengthIn35mmFilm/ExposureProgram/MeteringMode/Flash/WhiteBalance/
LensSpecification live in the Exif sub-IFD per the EXIF spec.

What's deliberately NOT copied:
- MakerNote (tag 37500): manufacturer-proprietary, often containing
  offsets/references tied to the original file's exact byte layout.
  Blindly copying it into a restructured file risks copying garbage or
  actively misleading data. Not parsed, not copied.
- GPS: not extracted, not copied, no opt-in flag. An earlier version of
  this module extracted GPS fields with an off-by-default preserve_gps
  flag, but the flag had no effect yet (writing GPS tags correctly needs
  a real GPSInfo sub-IFD, which wasn't implemented) -- decided to just
  drop GPS handling entirely rather than carry a half-built, currently-
  inert feature around. If GPS preservation becomes a real requirement,
  rebuild it properly (real sub-IFD, explicit opt-in) rather than
  resurrecting this.
- The embedded JPEG thumbnail some cameras store in EXIF: irrelevant to
  a DNG, not copied. (dng_writer.py generates its own preview thumbnail
  from the reconstructed pixel data instead -- see that file.)

Known simplification, stated plainly: these tags are written directly
into the DNG's main IFD0 via the same extratags mechanism dng_writer.py
already uses for its own tags, not into a separate Exif sub-IFD pointed
to by tag 34665 the way the strict EXIF spec prefers. tifffile has no
built-in support for writing a nested sub-IFD, and hand-constructing one
by patching raw TIFF offsets wasn't something that could be verified
correct in this environment without real reader testing. In practice
this isn't unusual -- plenty of real-world DNG writers place these tags
directly in IFD0 -- but a strict EXIF-spec reader that only looks in the
sub-IFD location would miss them. If that turns out to matter for a
real reader, revisit this.
"""

from __future__ import annotations

from PIL.ExifTags import Base, IFD
from PIL.TiffImagePlugin import IFDRational

# (tag_id, tifffile dtype code, friendly name) -- IFD0-resident tags.
_IFD0_FIELDS = [
    (Base.Make, "s", "Make"),
    (Base.Model, "s", "Model"),
    (Base.Orientation, "H", "Orientation"),
    (Base.DateTime, "s", "DateTime"),
    (Base.Artist, "s", "Artist"),
    (Base.Copyright, "s", "Copyright"),
]

# Exif-sub-IFD-resident tags per spec, written into IFD0 here (see
# module docstring's "Known simplification").
_EXIF_SUB_FIELDS = [
    (Base.ExposureTime, "2I", "ExposureTime"),  # RATIONAL
    (Base.FNumber, "2I", "FNumber"),  # RATIONAL
    (Base.ISOSpeedRatings, "H", "ISOSpeedRatings"),
    (Base.DateTimeOriginal, "s", "DateTimeOriginal"),
    (Base.DateTimeDigitized, "s", "DateTimeDigitized"),
    (Base.FocalLength, "2I", "FocalLength"),  # RATIONAL
    (Base.FocalLengthIn35mmFilm, "H", "FocalLengthIn35mmFilm"),
    (Base.LensMake, "s", "LensMake"),
    (Base.LensModel, "s", "LensModel"),
    (Base.LensSpecification, "2I", "LensSpecification"),  # RATIONAL[4]
    (Base.ExposureProgram, "H", "ExposureProgram"),
    (Base.MeteringMode, "H", "MeteringMode"),
    (Base.Flash, "H", "Flash"),
    (Base.WhiteBalance, "H", "WhiteBalance"),
    (Base.ExposureBiasValue, "2i", "ExposureBiasValue"),  # SRATIONAL
]


def extract_exif(pil_exif) -> dict:
    """
    pil_exif: a PIL.Image.Exif, e.g. from Image.open(path).getexif()
        (call .load() on the image first so it's actually populated).
    Returns: {'ifd0': {tag_id: value}, 'exif_sub': {tag_id: value}} --
    only tags from our curated allowlists above that are ACTUALLY
    PRESENT in the source. No fabrication, no defaults substituted for
    absent fields. No GPS -- see module docstring.
    """
    ifd0_ids = {t[0] for t in _IFD0_FIELDS}
    sub_ids = {t[0] for t in _EXIF_SUB_FIELDS}

    ifd0 = {k: v for k, v in dict(pil_exif).items() if k in ifd0_ids}

    try:
        sub = dict(pil_exif.get_ifd(IFD.Exif))
    except Exception:
        sub = {}
    sub = {k: v for k, v in sub.items() if k in sub_ids}

    return {"ifd0": ifd0, "exif_sub": sub}


def _rational_to_flat(value, signed: bool) -> list[int] | None:
    """PIL gives rationals as PIL.TiffImagePlugin.IFDRational objects
    (.numerator/.denominator) when read from a real file, or as plain
    (num, den) tuples if constructed by hand (e.g. in tests). Flatten to
    [num, den] for tifffile.

    Deliberately checks for IFDRational specifically rather than generic
    hasattr(value, 'numerator') -- plain Python ints ALSO have a
    .numerator/.denominator (int is a numbers.Rational), so that duck-typed
    check wrongly matched on bare ints too, corrupting any (num, den)
    tuple passed as plain ints (e.g. FNumber=(28, 10) was being split
    into two separate rationals, 28/1 and 10/1, instead of one, 28/10 --
    caught by test_dng_write_carries_real_make_model_overriding_placeholder).
    """
    try:
        if isinstance(value, IFDRational):
            num, den = value.numerator, value.denominator
        else:
            num, den = value
        return [int(num), int(den)]
    except (TypeError, ValueError, AttributeError):
        return None


def _value_to_tifffile(dtype: str, value):
    """Convert one PIL EXIF value into the form tifffile's extratags
    wants for the given dtype code. Returns None if the value couldn't
    be converted (field is then skipped, not fabricated-around)."""
    if dtype == "s":
        try:
            return str(value)
        except Exception:
            return None
    if dtype in ("2I", "2i"):
        # Could be a single rational or (for LensSpecification) a tuple
        # of several rationals. Checked via IFDRational specifically --
        # see _rational_to_flat's docstring for why generic
        # hasattr(..., 'numerator') is wrong here (plain ints have that
        # attribute too).
        if isinstance(value, (tuple, list)) and value and isinstance(value[0], IFDRational):
            flat: list[int] = []
            for v in value:
                r = _rational_to_flat(v, signed=(dtype == "2i"))
                if r is None:
                    return None
                flat.extend(r)
            return flat
        r = _rational_to_flat(value, signed=(dtype == "2i"))
        return r
    if dtype == "H":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if dtype == "B":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def build_dng_extratags(exif_fields: dict) -> list:
    """
    exif_fields: the dict returned by extract_exif().
    Returns: a list of tifffile extratag tuples ready to extend
        dng_writer.py's own extratags list with. Fields that are absent
        from exif_fields, or that fail to convert cleanly, are simply
        omitted -- never substituted with a guessed/default value.
    """
    tags = []

    for tag_id, dtype, _name in _IFD0_FIELDS + _EXIF_SUB_FIELDS:
        bucket = "ifd0" if (tag_id, dtype, _name) in _IFD0_FIELDS else "exif_sub"
        src = exif_fields.get(bucket, {})
        if tag_id not in src:
            continue
        converted = _value_to_tifffile(dtype, src[tag_id])
        if converted is None:
            continue
        count = 1 if dtype in ("s", "H", "B") else len(converted) // 2
        tags.append((tag_id, dtype, count, converted, False))

    return tags
