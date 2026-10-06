#!/usr/bin/env python3
"""
OpenRAW logo generator. Every shape is constructed from geometry, so the
logo is fully original and reproducible -- tweak the CONFIG values and
re-run to regenerate all variants.

Aperture construction: N lines, each extending one side of a regular
N-gon (the lens opening) outward to the rim. Those lines, drawn as gaps,
split the disk into N pinwheel blades.

Lettering: IBM Plex Sans (SIL Open Font License, see fonts/OFL-IBMPlexSans.txt), shaped
with HarfBuzz for real kerning and converted to outlines, so the SVGs
render identically without the font installed.

Output is FLATTENED geometry: blades, opening and the knockout around the
wordmark are computed as real boolean shape operations (shapely) and
written as plain filled paths -- no SVG masks or clip paths. A first
mask-based version rendered wrong in cairosvg (masks silently ignored),
and favicon generators, design-tool imports and print workflows are just
as uneven, so the logo must not depend on renderer features.

Requires: fonttools, uharfbuzz, shapely (and cairosvg for PNG export).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.basePen import BasePen
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from shapely import affinity

HERE = Path(__file__).resolve().parent

# ---- geometry (aperture units: rim radius = 100) ----
BLADES = 6
RING_OUTER, RING_INNER = 100.0, 91.0   # outer rim ring
DISK_R = 85.0                           # blade disk radius (gap between ring and blades)
OPENING_R = 33.0                        # lens-opening polygon circumradius
OPENING_ROT = math.radians(-8)          # opening rotation (gives the pinwheel its lean)
GAP = 4.6                               # gap between blades

# ---- palette ----
INK_LIGHT_BG, INK_DARK_BG = "#141414", "#F2F2F2"
BAYER = {"R": "#E5484D", "G": "#2FA36B", "B": "#3E63DD"}


# ---------------------------------------------------------------- text
_font_cache: dict = {}

# Wordmark typeface. Any variable TTF works; every axis except weight is
# pinned (optical size, if present, to its display end).
FONT = HERE / "fonts" / "IBMPlexSans[wdth,wght].ttf"


def _instance(wght: float, font: Path | None = None):
    font = Path(font or FONT)
    key = (str(font), wght)
    if key not in _font_cache:
        vf = TTFont(font)
        loc = {}
        for ax in vf["fvar"].axes:
            if ax.axisTag == "wght":
                loc["wght"] = max(ax.minValue, min(ax.maxValue, wght))  # clamp to the font's range
            elif ax.axisTag == "opsz":
                loc["opsz"] = ax.maxValue
            else:
                loc[ax.axisTag] = ax.defaultValue
        inst = instantiateVariableFont(vf, loc)
        path = HERE / f".inst_{font.stem}_{int(wght)}.ttf"
        inst.save(path)
        _font_cache[key] = (TTFont(path), hb.Font(hb.Face(hb.Blob(path.read_bytes()))), path)
    return _font_cache[key]


class _FlatPen(BasePen):
    """Flattens glyph outlines (lines + quadratic/cubic curves) to polygons."""
    STEPS = 24

    def __init__(self, glyphset):
        super().__init__(glyphset)
        self.contours, self.cur = [], []

    def _moveTo(self, p):
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self.cur[-1]
        for k in range(1, self.STEPS + 1):
            t = k / self.STEPS; u = 1 - t
            self.cur.append((u**3*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t**3*p3[0],
                             u**3*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t**3*p3[1]))

    def _qCurveToOne(self, p1, p2):
        p0 = self.cur[-1]
        for k in range(1, self.STEPS + 1):
            t = k / self.STEPS; u = 1 - t
            self.cur.append((u*u*p0[0] + 2*u*t*p1[0] + t*t*p2[0], u*u*p0[1] + 2*u*t*p1[1] + t*t*p2[1]))

    def _closePath(self):
        if len(self.cur) >= 3:
            self.contours.append(self.cur)
        self.cur = []

    _endPath = _closePath


def text_geometry(runs, size: float, x: float, y: float, font=None):
    """runs: [(text, weight), ...] on one baseline at y, starting at x.
    Returns ([shapely geometry per run], total advance width)."""
    geoms, pen_x = [], x
    for text, wght in runs:
        tt, hbfont, _ = _instance(wght, font)
        s = size / tt["head"].unitsPerEm
        buf = hb.Buffer(); buf.add_str(text); buf.guess_segment_properties()
        hb.shape(hbfont, buf, {"kern": True, "liga": True})
        gs, order = tt.getGlyphSet(), tt.getGlyphOrder()
        parts = []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            pen = _FlatPen(gs)
            gs[order[info.codepoint]].draw(pen)
            glyph = None
            for c in pen.contours:  # XOR of contours = outer shapes minus counters
                poly = Polygon(c).buffer(0)
                glyph = poly if glyph is None else glyph.symmetric_difference(poly)
            if glyph is not None:
                # font units (y up) -> logo units (y down)
                ox, oy = pen_x + pos.x_offset * s, y - pos.y_offset * s
                parts.append(affinity.affine_transform(glyph, [s, 0, 0, -s, ox, oy]))
            pen_x += pos.x_advance * s
        geoms.append(unary_union(parts))
    return geoms, pen_x - x


# ------------------------------------------------------------ aperture
RES = 256  # circle segments


def _polygon(n, r, rot, cx=0.0, cy=0.0):
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n), cy + r * math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]


def aperture_geometry(core: str):
    """Returns {'ink': geometry, 'R'/'G'/'B': geometry (bayer core only)}."""
    ring = Point(0, 0).buffer(RING_OUTER, RES).difference(Point(0, 0).buffer(RING_INNER, RES))
    opening_pts = _polygon(BLADES, OPENING_R, OPENING_ROT)
    opening = Polygon(opening_pts)
    gaps = []
    for i in range(BLADES):
        (x0, y0), (x1, y1) = opening_pts[i], opening_pts[(i + 1) % BLADES]
        L = math.hypot(x1 - x0, y1 - y0)
        far = 3 * DISK_R
        gaps.append(LineString([(x0, y0), (x0 + (x1 - x0) / L * far, y0 + (y1 - y0) / L * far)]).buffer(GAP / 2, cap_style="flat"))
    blades = Point(0, 0).buffer(DISK_R, RES).difference(opening).difference(unary_union(gaps))
    out = {"ink": unary_union([ring, blades])}
    if core == "bayer":
        inner = Polygon(_polygon(BLADES, OPENING_R - GAP * 0.9, OPENING_ROT))
        h, g = OPENING_R, GAP * 0.45
        quads = {"R": box(-h, -h, -g / 2, -g / 2), "G": unary_union([box(g / 2, -h, h, -g / 2), box(-h, g / 2, -g / 2, h)]),
                 "B": box(g / 2, g / 2, h, h)}
        for c, q in quads.items():
            out[c] = q.intersection(inner)
    return out


# ------------------------------------------------------------- output
def _path_d(geom) -> str:
    polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    out = []
    for p in polys:
        if p.is_empty or p.geom_type != "Polygon":
            continue
        for ring in [p.exterior, *p.interiors]:
            pts = list(ring.coords)[:-1]
            out.append("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + "Z")
    return "".join(out)


def _paths(layers):
    return "".join(f'\n  <path fill="{color}" fill-rule="evenodd" d="{_path_d(g)}"/>' for g, color in layers if not g.is_empty)


def svg(layers, pad=6.0, title="OpenRAW"):
    minx, miny, maxx, maxy = unary_union([g for g, _ in layers if not g.is_empty]).bounds
    x, y, w, h = minx - pad, miny - pad, maxx - minx + 2 * pad, maxy - miny + 2 * pad
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x:.2f} {y:.2f} {w:.2f} {h:.2f}" '
            f'role="img" aria-label="{title}">\n  <title>{title}</title>{_paths(layers)}\n</svg>\n')


# ---------------------------------------------------------- compositions
def lockup_layers(ink, core, runs, size=36.0, halo=5.4, baseline=63.0, font=None):
    """Aperture with the wordmark across its lower-right rim, about half the
    word outside the circle; the aperture is knocked out around the letters
    (a halo) so the word reads cleanly over the blades."""
    _, w = text_geometry(runs, size, 0, 0, font)
    mid = baseline - size * 0.35                      # optical middle of the caps
    rim_x = math.sqrt(max(RING_OUTER ** 2 - mid ** 2, 0))
    words, _ = text_geometry(runs, size, rim_x - w / 2, baseline, font)
    halo_shape = unary_union(words).buffer(halo, join_style="round")
    ap = aperture_geometry(core)
    layers = [(ap["ink"].difference(halo_shape), ink)]
    layers += [(ap[c].difference(halo_shape), BAYER[c]) for c in ("R", "G", "B") if c in ap]
    layers += [(g, ink) for g in words]
    return layers


def mark_layers(ink, core):
    ap = aperture_geometry(core)
    return [(ap["ink"], ink)] + [(ap[c], BAYER[c]) for c in ("R", "G", "B") if c in ap]


# ------------------------------------------------------------------ main
# The OpenRAW logo: aperture with a Bayer RGGB sensor tile in the opening,
# wordmark "Open" regular / "RAW" bold across the lower-right rim.
WORDMARK = [("Open", 400), ("RAW", 700)]


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for theme, ink in (("light", INK_LIGHT_BG), ("dark", INK_DARK_BG)):
        outputs = {
            f"openraw-logo-{theme}.svg": (svg(lockup_layers(ink, "bayer", WORDMARK)), ),
            f"openraw-mark-{theme}.svg": (svg(mark_layers(ink, "bayer"), pad=4), ),
            # single-colour mark: for contexts where colour isn't available
            f"openraw-mark-mono-{theme}.svg": (svg(mark_layers(ink, "plain"), pad=4), ),
        }
        for name, (content,) in outputs.items():
            p = out / name
            p.write_text(content)
            files.append(p)
    for f in HERE.glob(".inst_*.ttf"):
        f.unlink()
    return files


if __name__ == "__main__":
    for f in build(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "assets" / "logo"):
        print(f)
