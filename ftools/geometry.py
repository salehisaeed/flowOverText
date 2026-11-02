"""
Outline geometry for flowOverText.

Turns a piece of text (any installed font) or an SVG logo into a set of
closed 2D polygons, normalised so that the overall height is 1.  Solid
outlines are counter-clockwise, holes (e.g. the inside of 'o') clockwise.

The polygons are written as
  - an extruded "ribbon" STL (side walls only) for snappyHexMesh,
  - a JSON file that the renderer uses to draw crisp vector outlines.
"""

import json
import math
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib.path import Path


# --------------------------------------------------------------------------
# Bezier flattening
# --------------------------------------------------------------------------

def _bezier(ctrl, ds):
    """Sample a Bezier curve (2, 3 or 4 control points), excluding the start."""
    ctrl = np.asarray(ctrl, float)
    length = np.sum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1))
    n = max(2, int(math.ceil(length/ds)))
    t = np.linspace(0, 1, n + 1)[1:, None]
    if len(ctrl) == 2:
        return (1 - t)*ctrl[0] + t*ctrl[1]
    if len(ctrl) == 3:
        return (1 - t)**2*ctrl[0] + 2*(1 - t)*t*ctrl[1] + t**2*ctrl[2]
    return ((1 - t)**3*ctrl[0] + 3*(1 - t)**2*t*ctrl[1]
            + 3*(1 - t)*t**2*ctrl[2] + t**3*ctrl[3])


def _clean(poly, tol):
    """Drop repeated and collinear points, and the closing duplicate."""
    poly = np.asarray(poly, float)
    keep = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - keep[-1]) > tol:
            keep.append(p)
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= tol:
        keep.pop()
    poly = np.array(keep)
    changed = True
    while changed and len(poly) > 3:
        changed = False
        prev = np.roll(poly, 1, axis=0)
        nxt = np.roll(poly, -1, axis=0)
        cross = np.abs(np.cross(poly - prev, nxt - poly))
        scale = np.linalg.norm(nxt - prev, axis=1)
        straight = cross < 1e-9*np.maximum(scale, 1e-30)**2
        if straight.any():
            poly = poly[~straight]
            changed = True
    return poly


def _signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5*np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

def resolve_font(spec):
    """Font file from a path or a fontconfig pattern such as 'Inter:bold'."""
    import os
    if os.path.isfile(spec):
        return spec
    if shutil.which("fc-match"):
        out = subprocess.run(["fc-match", "-f", "%{file}", spec],
                             capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    from matplotlib.font_manager import FontProperties, findfont
    family, _, style = spec.partition(":")
    return findfont(FontProperties(family=family, weight=style or "normal"))


def text_polygons(text, font_file, ds):
    """Polygons of a (single-line) text, height not yet normalised."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    # Large size so that hinting/rounding effects are negligible
    size = 100.0
    tp = TextPath((0, 0), text, size=size, prop=FontProperties(fname=font_file))
    polys, cur, start = [], [], None
    ds_glyph = ds*size
    verts, codes = tp.vertices, tp.codes
    i = 0
    while i < len(codes):
        c = codes[i]
        if c == Path.MOVETO:
            if len(cur) > 2:
                polys.append(np.array(cur))
            cur = [verts[i]]
            start = verts[i]
            i += 1
        elif c == Path.LINETO:
            cur.append(verts[i])
            i += 1
        elif c == Path.CURVE3:
            cur.extend(_bezier([cur[-1], verts[i], verts[i + 1]], ds_glyph))
            i += 2
        elif c == Path.CURVE4:
            cur.extend(_bezier([cur[-1], verts[i], verts[i + 1], verts[i + 2]],
                               ds_glyph))
            i += 3
        elif c == Path.CLOSEPOLY:
            if start is not None:
                cur.append(start)
            if len(cur) > 2:
                polys.append(np.array(cur))
            cur, start = [], None
            i += 1
        else:
            i += 1
    if len(cur) > 2:
        polys.append(np.array(cur))
    return [p/size for p in polys]


# --------------------------------------------------------------------------
# SVG (paths and basic shapes; fills only, strokes are ignored)
# --------------------------------------------------------------------------

_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_CMD = re.compile(r"[MmZzLlHhVvCcSsQqTtAa]")


def _parse_transform(s):
    m = np.eye(3)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(a) for a in _NUM.findall(args)]
        t = np.eye(3)
        if name == "matrix":
            t[:2, :] = np.array(v).reshape(3, 2).T
        elif name == "translate":
            t[0, 2] = v[0]
            t[1, 2] = v[1] if len(v) > 1 else 0
        elif name == "scale":
            t[0, 0] = v[0]
            t[1, 1] = v[1] if len(v) > 1 else v[0]
        elif name == "rotate":
            a = math.radians(v[0])
            r = np.array([[math.cos(a), -math.sin(a), 0],
                          [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            if len(v) == 3:
                c1, c2 = np.eye(3), np.eye(3)
                c1[:2, 2] = v[1:]
                c2[:2, 2] = [-v[1], -v[2]]
                r = c1 @ r @ c2
            t = r
        elif name == "skewX":
            t[0, 1] = math.tan(math.radians(v[0]))
        elif name == "skewY":
            t[1, 0] = math.tan(math.radians(v[0]))
        m = m @ t
    return m


class _PathTokens:
    """Cursor over an SVG path 'd' string (handles packed arc flags)."""

    def __init__(self, d):
        self.d, self.i = d, 0

    def _skip(self):
        while self.i < len(self.d) and self.d[self.i] in " \t\r\n,":
            self.i += 1

    def command(self):
        self._skip()
        if self.i < len(self.d):
            m = _CMD.match(self.d, self.i)
            if m:
                self.i = m.end()
                return m.group()
        return None

    def has_number(self):
        self._skip()
        return self.i < len(self.d) and _NUM.match(self.d, self.i) is not None

    def number(self):
        self._skip()
        m = _NUM.match(self.d, self.i)
        if not m:
            raise ValueError(f"SVG path: number expected at '{self.d[self.i:self.i+20]}'")
        self.i = m.end()
        return float(m.group())

    def flag(self):
        self._skip()
        c = self.d[self.i]
        self.i += 1
        return c == "1"


def _arc(p0, rx, ry, phi, large, sweep, p1, ds):
    """SVG elliptical arc to polyline (endpoint parametrisation, SVG spec F.6)."""
    if rx == 0 or ry == 0:
        return np.array([p1])
    rx, ry = abs(rx), abs(ry)
    phi = math.radians(phi)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0])/2, (p0[1] - p1[1])/2
    x1, y1 = cp*dx + sp*dy, -sp*dx + cp*dy
    lam = x1**2/rx**2 + y1**2/ry**2
    if lam > 1:
        rx, ry = rx*math.sqrt(lam), ry*math.sqrt(lam)
    num = rx**2*ry**2 - rx**2*y1**2 - ry**2*x1**2
    den = rx**2*y1**2 + ry**2*x1**2
    co = math.sqrt(max(0, num/den)) if den else 0
    if large == sweep:
        co = -co
    cx1, cy1 = co*rx*y1/ry, -co*ry*x1/rx
    cx = cp*cx1 - sp*cy1 + (p0[0] + p1[0])/2
    cy = sp*cx1 + cp*cy1 + (p0[1] + p1[1])/2

    def ang(u, v):
        return math.atan2(u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1])

    t1 = ang((1, 0), ((x1 - cx1)/rx, (y1 - cy1)/ry))
    dt = ang(((x1 - cx1)/rx, (y1 - cy1)/ry), ((-x1 - cx1)/rx, (-y1 - cy1)/ry))
    if not sweep and dt > 0:
        dt -= 2*math.pi
    elif sweep and dt < 0:
        dt += 2*math.pi
    n = max(4, int(math.ceil(abs(dt)*max(rx, ry)/ds)))
    t = t1 + dt*np.linspace(0, 1, n + 1)[1:]
    return np.column_stack([cx + rx*np.cos(t)*cp - ry*np.sin(t)*sp,
                            cy + rx*np.cos(t)*sp + ry*np.sin(t)*cp])


def _path_d(d, ds):
    tok = _PathTokens(d)
    polys, cur = [], []
    p = np.zeros(2)
    start = np.zeros(2)
    last_ctrl, last_cmd = None, ""
    cmd = tok.command()
    while cmd:
        rel = cmd.islower()
        C = cmd.upper()
        first = True
        while True:
            if C == "Z":
                if len(cur) > 2:
                    polys.append(np.array(cur))
                cur, p = [], start.copy()
                last_ctrl = None
                break
            if not first and not tok.has_number():
                break
            off = p if rel else np.zeros(2)
            if C == "M":
                q = np.array([tok.number(), tok.number()]) + off
                if first:
                    if len(cur) > 2:
                        polys.append(np.array(cur))
                    cur, start = [q], q.copy()
                else:
                    cur.append(q)
                p = q
                last_ctrl = None
            elif C == "L":
                p = np.array([tok.number(), tok.number()]) + off
                cur.append(p)
                last_ctrl = None
            elif C == "H":
                x = tok.number() + (p[0] if rel else 0)
                p = np.array([x, p[1]])
                cur.append(p)
                last_ctrl = None
            elif C == "V":
                y = tok.number() + (p[1] if rel else 0)
                p = np.array([p[0], y])
                cur.append(p)
                last_ctrl = None
            elif C in "CS":
                if C == "C":
                    c1 = np.array([tok.number(), tok.number()]) + off
                else:
                    c1 = 2*p - last_ctrl if last_cmd in "CS" and last_ctrl is not None else p
                c2 = np.array([tok.number(), tok.number()]) + off
                q = np.array([tok.number(), tok.number()]) + off
                cur.extend(_bezier([p, c1, c2, q], ds))
                p, last_ctrl = q, c2
            elif C in "QT":
                if C == "Q":
                    c1 = np.array([tok.number(), tok.number()]) + off
                else:
                    c1 = 2*p - last_ctrl if last_cmd in "QT" and last_ctrl is not None else p
                q = np.array([tok.number(), tok.number()]) + off
                cur.extend(_bezier([p, c1, q], ds))
                p, last_ctrl = q, c1
            elif C == "A":
                rx, ry, phi = tok.number(), tok.number(), tok.number()
                large, sweep = tok.flag(), tok.flag()
                q = np.array([tok.number(), tok.number()]) + off
                cur.extend(_arc(p, rx, ry, phi, large, sweep, q, ds))
                p = q
                last_ctrl = None
            last_cmd = C
            first = False
            if C == "M":
                C = "L"     # implicit lineto after moveto
        last_cmd = C
        cmd = tok.command()
    if len(cur) > 2:
        polys.append(np.array(cur))
    return polys


def _ellipse(cx, cy, rx, ry, ds):
    n = max(16, int(math.ceil(2*math.pi*max(rx, ry)/ds)))
    t = np.linspace(0, 2*math.pi, n, endpoint=False)
    return [np.column_stack([cx + rx*np.cos(t), cy + ry*np.sin(t)])]


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _style(el):
    st = dict(el.attrib)
    for item in el.attrib.get("style", "").split(";"):
        if ":" in item:
            k, v = item.split(":", 1)
            st[k.strip()] = v.strip()
    return st


def svg_polygons(svg_file, ds_rel):
    """Filled shapes of an SVG file in SVG user units (y flipped upwards)."""
    root = ET.parse(svg_file).getroot()

    # Sampling distance in SVG units, relative to the drawing size
    vb = _NUM.findall(root.attrib.get("viewBox", ""))
    if len(vb) == 4:
        extent = float(vb[3])
    else:
        extent = float(_NUM.findall(root.attrib.get("height", "100"))[0])
    ds = ds_rel*extent

    polys = []
    skip = {"defs", "clipPath", "mask", "symbol", "marker", "pattern",
            "title", "desc", "metadata", "style", "text"}

    def walk(el, m, fill_none):
        tag = _local(el.tag)
        if tag in skip:
            return
        st = _style(el)
        if st.get("display") == "none" or st.get("visibility") == "hidden":
            return
        m = m @ _parse_transform(el.attrib.get("transform"))
        fill = st.get("fill")
        if fill is not None:
            fill_none = fill.strip() == "none"
        shapes = []
        if tag == "path":
            shapes = _path_d(el.attrib.get("d", ""), ds)
        elif tag in ("polygon", "polyline"):
            v = [float(a) for a in _NUM.findall(el.attrib.get("points", ""))]
            if len(v) >= 6:
                shapes = [np.array(v).reshape(-1, 2)]
        elif tag == "rect":
            x, y = float(st.get("x", 0)), float(st.get("y", 0))
            w, h = float(st.get("width", 0)), float(st.get("height", 0))
            if w > 0 and h > 0:
                shapes = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])]
        elif tag == "circle":
            r = float(st.get("r", 0))
            shapes = _ellipse(float(st.get("cx", 0)), float(st.get("cy", 0)), r, r, ds)
        elif tag == "ellipse":
            shapes = _ellipse(float(st.get("cx", 0)), float(st.get("cy", 0)),
                              float(st.get("rx", 0)), float(st.get("ry", 0)), ds)
        if shapes and not fill_none:
            for s in shapes:
                h = np.column_stack([s, np.ones(len(s))]) @ m.T
                polys.append(h[:, :2]*[1, -1])
        for child in el:
            walk(child, m, fill_none)

    walk(root, np.eye(3), False)
    if not polys:
        raise ValueError(f"No filled shapes found in {svg_file}"
                         " (strokes and <text> are not supported: convert"
                         " them to paths first, e.g. Inkscape 'Object to Path').")
    return polys


# --------------------------------------------------------------------------
# Normalisation and orientation
# --------------------------------------------------------------------------

def finalise(polys, ds):
    """Scale to unit height, place at x=0/y-centred, orient solids/holes.

    Returns (polygons, warnings).
    """
    allp = np.vstack(polys)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    scale = 1.0/(hi[1] - lo[1])
    shift = np.array([lo[0], 0.5*(lo[1] + hi[1])])
    polys = [_clean((p - shift)*scale, 1e-3*ds) for p in polys]
    polys = [p for p in polys if len(p) >= 3 and abs(_signed_area(p)) > 1e-8]

    warnings = []
    paths = [Path(p) for p in polys]
    depth = np.zeros(len(polys), int)
    for i, p in enumerate(polys):
        for j, path in enumerate(paths):
            if i == j:
                continue
            inside = path.contains_points(p)
            if inside.all():
                depth[i] += 1
            elif inside.any():
                warnings.append(
                    f"outlines {i} and {j} overlap; merge overlapping shapes"
                    " (e.g. Inkscape Path > Union) for a clean surface")
    out = []
    for p, d in zip(polys, depth):
        solid = d % 2 == 0
        if (_signed_area(p) > 0) != solid:
            p = p[::-1]
        out.append(p)
    return out, sorted(set(warnings))


def make_polygons(text=None, svg=None, font="DejaVu Sans:bold", ds=0.004):
    """Front-end: polygons (unit height) + info dict."""
    if svg:
        polys = svg_polygons(svg, ds)
        info = {"source": "svg", "svg": str(svg)}
    else:
        font_file = resolve_font(font)
        polys = text_polygons(text, font_file, ds)
        info = {"source": "text", "text": text, "font": font,
                "fontFile": font_file}
    polys, warnings = finalise(polys, ds)
    allp = np.vstack(polys)
    info["bounds"] = [*allp.min(axis=0).tolist(), *allp.max(axis=0).tolist()]
    info["nPolygons"] = len(polys)
    return polys, info, warnings


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_ribbon_stl(polys, path, z0, z1, name="text"):
    """Side walls of the extruded outlines, normals pointing into the fluid."""
    with open(path, "w") as f:
        f.write(f"solid {name}\n")
        for p in polys:
            q = np.roll(p, -1, axis=0)
            d = q - p
            n = np.column_stack([d[:, 1], -d[:, 0]])
            n /= np.linalg.norm(n, axis=1)[:, None]
            for a, b, nn in zip(p, q, n):
                A0, B0 = (a[0], a[1], z0), (b[0], b[1], z0)
                A1, B1 = (a[0], a[1], z1), (b[0], b[1], z1)
                for tri in ((A0, B0, B1), (A0, B1, A1)):
                    f.write(f"  facet normal {nn[0]:.9g} {nn[1]:.9g} 0\n"
                            "    outer loop\n")
                    for v in tri:
                        f.write(f"      vertex {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
                    f.write("    endloop\n  endfacet\n")
        f.write(f"endsolid {name}\n")


def write_json(polys, info, path):
    data = dict(info)
    data["polygons"] = [np.round(p, 7).tolist() for p in polys]
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    data["polygons"] = [np.array(p) for p in data["polygons"]]
    return data
