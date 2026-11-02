#!/usr/bin/env python3
"""
Animation of a flowOverText case in the style of the TMMV64 course
cylinder-wake pictures (Linköping University): filled bands that fade into
the paper colour near zero (RdBu reversed; many more bands than the course
for a smooth gradation), display smoothing, and contours drawn as smooth
curves; optional contour lines (negative ones optionally dashed).  The
letters are drawn like the cylinder there: grey with a darker outline.

Steps per frame
  1. the field (cell centres + wall faces, read straight from processor*/)
     is interpolated linearly onto a uniform grid over the view window,
     NaN inside the letters
  2. NaN-aware Gaussian smoothing (display only)
  3. contourpy bands and lines, simplified (Ramer-Douglas-Peucker) and drawn
     as Catmull-Rom curves (cubic Beziers)

Output in <case>/animation/
  <case>.mp4            the movie (H.264)
  svg/frame_NNNNN.svg   vector frames (render.svg: none | last | all | every N)
  frames/               PNG frames used for the movie (removed by Allrun.cleanup)

Settings come from the render section of <case>/flowOverText.yaml;
command-line options override them.
"""

import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from contourpy import FillType, LineType, contour_generator  # noqa: E402
from scipy.ndimage import distance_transform_edt, gaussian_filter  # noqa: E402
from scipy.spatial import Delaunay, cKDTree  # noqa: E402

from ftools import foamio  # noqa: E402

WALL = "text"

# --------------------------------------------------------------------------
# Style (as in the TMMV64 course cylinder-wake pictures)
# --------------------------------------------------------------------------

LIM = 3.0                                   # reference: bands up to |w| = 3
LINES = [-3, -2, -1, -0.6, 0.6, 1, 2, 3]    # contour lines for lim = 3


def band_levels(n):
    """n band edges for lim = 3, evenly spaced (the course uses 6:
    0.25, 0.6, 1, 1.5, 2, 3); more bands give a smoother gradation."""
    return list(np.linspace(0.15, LIM, n))
BG = "#f7f6f2"                              # paper
LINE = "#3a3c40"
BODY, BODY_EDGE = "#8a8d93", "#55585d"
FRAME = "#d6d4cc"



def _hex(rgb):
    return "#%02x%02x%02x" % tuple(round(255*v) for v in rgb)


def level_colours(levels, cmap="RdBu_r", bg=BG):
    """{+1: [...], -1: [...]}: opaque colours of the bands |f| > levels[i],
    faded into the background near zero, as the course.  A diverging map
    (RdBu_r) is used from its centre outwards; a sequential map over its
    whole range for +1."""
    from matplotlib.colors import to_rgb
    diverging = cmap in ("RdBu_r", "RdBu", "coolwarm", "bwr", "seismic", "PuOr_r", "PRGn")
    cmap = plt.get_cmap(cmap)
    bgc = to_rgb(bg)
    ups = levels[1:] + [LIM + 1]
    out = {}
    for sign in (1, -1):
        cols = []
        for lo, hi in zip(levels, ups):
            mid = min((lo + hi)/2, LIM)
            r, g, b, _ = cmap(0.5 + 0.5*sign*mid/LIM if diverging else mid/LIM)
            a = float(min(max((mid - 0.05)/0.35, 0), 1)**0.8)
            cols.append(_hex([a*c + (1 - a)*k for c, k in zip((r, g, b), bgc)]))
        out[sign] = cols
    return out


# --------------------------------------------------------------------------
# Smooth vector contours (as make_cylinder_wake.py)
# --------------------------------------------------------------------------

def simplify(P, tol):
    """Ramer-Douglas-Peucker: keep only the points needed to stay within tol."""
    if len(P) < 3:
        return P
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        d = b - a
        L = np.hypot(*d)
        seg = P[i + 1:j]
        dist = (np.abs(d[0]*(seg[:, 1] - a[1]) - d[1]*(seg[:, 0] - a[0]))/L
                if L > 0 else np.hypot(*(seg - a).T))
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return P[keep]


def smooth_path(pts, closed, tol):
    """Vertices and codes of a Catmull-Rom curve through the points (cubic Beziers)."""
    P = np.asarray(pts, float)
    if closed and len(P) > 3:
        h = len(P)//2                       # simplify a closed ring as two open halves
        P = np.vstack([simplify(P[:h + 1], tol)[:-1], simplify(P[h:], tol)])
    else:
        P = simplify(P, tol)
    if closed and len(P) > 1 and np.allclose(P[0], P[-1]):
        P = P[:-1]
    n = len(P)
    if n < 3:
        v = list(P) + ([P[0]] if closed else [])
        c = [Path.MOVETO] + [Path.LINETO]*(len(v) - 1)
        return v, c
    verts, codes = [P[0]], [Path.MOVETO]
    last = n if closed else n - 1
    for i in range(last):
        p0 = P[(i - 1) % n] if (closed or i > 0) else P[0]
        p1, p2 = P[i], P[(i + 1) % n]
        p3 = P[(i + 2) % n] if (closed or i + 2 < n) else P[-1]
        verts += [p1 + (p2 - p0)/6, p2 - (p3 - p1)/6, p2]
        codes += [Path.CURVE4]*3
    if closed:
        verts.append(P[0])
        codes.append(Path.CLOSEPOLY)
    return verts, codes


def compound(parts):
    verts, codes = [], []
    for v, c in parts:
        verts += list(v)
        codes += c
    return Path(np.array(verts), codes) if verts else None


def filled_paths(points, codes, tol):
    starts = list(np.flatnonzero(codes == 1)) + [len(codes)]
    return [smooth_path(points[a:b], True, tol) for a, b in zip(starts[:-1], starts[1:])]


def smooth_for_display(w, sigma):
    """NaN-aware Gaussian smoothing (sigma in grid points), for pictures only."""
    m = ~np.isnan(w)
    a = gaussian_filter(np.where(m, w, 0.0), sigma)
    b = gaussian_filter(m.astype(float), sigma)
    out = a/np.maximum(b, 1e-9)
    out[~m] = np.nan
    return out


def fill_nearest(w):
    """NaN (inside the letters) replaced by the nearest valid value, so the
    contour bands keep their wall value right up to the outline instead of
    dropping towards zero across the last grid cell."""
    m = np.isnan(w)
    if not m.any():
        return w
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    return w[tuple(idx)]


# --------------------------------------------------------------------------
# Data: OpenFOAM points -> uniform grid
# --------------------------------------------------------------------------

def load_case(case):
    import json
    with open(os.path.join(case, "constant", "geometry.json")) as f:
        meta = json.load(f)
    meta["polygons"] = [np.array(p) for p in meta["polygons"]]
    return meta


def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5*np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)


class Sampler:
    """Linear interpolation of the case's cell (and wall face) values onto a
    uniform grid over the view window; NaN inside the letters."""

    def __init__(self, case, view, polys, h):
        self.roots = foamio.processor_dirs(case)
        x0, x1, y0, y1 = view
        m = 3*h
        self.sel, pts = [], []
        for root in self.roots:
            cfile = os.path.join(root, "0", "C")
            if not os.path.isfile(cfile):
                sys.exit(f"{cfile} not found: run 'postProcess -func writeCellCentres -time 0'"
                         " (Allrun.pre does this)")
            C, bf = foamio.read_field(cfile, patches=[WALL])
            ci = ((C[:, 0] > x0 - m) & (C[:, 0] < x1 + m)
                  & (C[:, 1] > y0 - m) & (C[:, 1] < y1 + m))
            pc = [C[ci, :2]]
            wi = near = None
            if WALL in bf and bf[WALL][0] == "list" and len(bf[WALL][1]):
                Wc = bf[WALL][1]
                wi = ((Wc[:, 0] > x0 - m) & (Wc[:, 0] < x1 + m)
                      & (Wc[:, 1] > y0 - m) & (Wc[:, 1] < y1 + m))
                pc.append(Wc[wi, :2])
                # wall-adjacent cell, for fields without wall values (zeroGradient)
                near = cKDTree(C[:, :2]).query(Wc[wi, :2])[1]
            self.sel.append((ci, wi, near))
            pts.append(np.vstack(pc))
        pts = np.vstack(pts)

        nx = int(round((x1 - x0)/h)) + 1
        ny = int(round((y1 - y0)/h)) + 1
        self.x = np.linspace(x0, x1, nx)
        self.y = np.linspace(y0, y1, ny)
        XX, YY = np.meshgrid(self.x, self.y)
        G = np.column_stack([XX.ravel(), YY.ravel()])

        # Barycentric weights of every grid point (jitter: the structured
        # background has cocircular points)
        rng = np.random.default_rng(0)
        tri = Delaunay(pts + 1e-9*rng.standard_normal(pts.shape))
        s = tri.find_simplex(G)
        T = tri.transform[s]
        b = np.einsum("ijk,ik->ij", T[:, :2], G - T[:, 2])
        self.w = np.column_stack([b, 1 - b.sum(axis=1)])
        self.idx = tri.simplices[s]

        inside = s < 0
        for p in polys:
            if signed_area(p) > 0:
                inside |= Path(p).contains_points(G)
        self.nan = inside
        self.shape = (ny, nx)
        print(f"Sampling {len(pts)} points onto a {nx} x {ny} grid (h = H/{1/h:.0f})")

    def point_values(self, field, tname):
        out = []
        for root, (ci, wi, near) in zip(self.roots, self.sel):
            f, bf = foamio.read_field(os.path.join(root, tname, field), patches=[WALL])
            f = np.broadcast_to(f, ci.shape) if f.size == 1 else f
            out.append(f[ci])
            if wi is not None:
                if WALL in bf:
                    v = bf[WALL][1]
                    out.append((np.broadcast_to(v, wi.shape) if v.size == 1 else v)[wi])
                else:
                    out.append(f[near])
        return np.concatenate(out)

    def grid(self, field, tname):
        v = self.point_values(field, tname)
        g = np.sum(v[self.idx]*self.w, axis=1)
        g[self.nan] = np.nan
        return g.reshape(self.shape)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def make_axes(view, opts):
    x0, x1, y0, y1 = view
    total = x1 - x0
    Wpx = opts.width
    Hpx = int(round(Wpx*(y1 - y0)/total/2))*2
    dpi = 200
    fig = plt.figure(figsize=(Wpx/dpi, Hpx/dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x0 + total)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)
    return fig, ax


def draw(ax, sampler, z, meta, opts):
    x0, x1, y0, y1 = meta["view"]
    signed = meta["mode"] == "forced"
    s = opts.lim/LIM
    levels = band_levels(opts.bands)
    cols = level_colours(levels, opts.cmap)
    lw = 0.6*opts.width/1920                    # line width in points
    tol = opts.tol

    z = fill_nearest(smooth_for_display(z, opts.sigma))
    X, Y = sampler.x[::opts.step], sampler.y[::opts.step]
    z = z[::opts.step, ::opts.step]
    fill_gen = contour_generator(X, Y, z, fill_type=FillType.OuterCode)
    line_gen = contour_generator(X, Y, z, line_type=LineType.Separate)

    clip = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax.transData)
    for sign in ((1, -1) if signed else (1,)):
        for i, lev in enumerate(levels):        # nested regions, weakest first
            lo, hi = (lev*s, 1e30) if sign > 0 else (-1e30, -lev*s)
            pts, codes = fill_gen.filled(lo, hi)
            parts = []
            for p_, c_ in zip(pts, codes):
                parts += filled_paths(p_, c_, tol)
            path = compound(parts)
            if path is not None:
                patch = ax.add_patch(PathPatch(path, facecolor=cols[sign][i], edgecolor="none",
                                               zorder=2 + 0.01*i))
                patch.set_clip_path(clip)
    if opts.lines:
        for lev in LINES:
            if not signed and lev < 0:
                continue
            for seg in line_gen.lines(lev*s):
                if len(seg) > 2:
                    v, c = smooth_path(seg, np.allclose(seg[0], seg[-1]), tol)
                    patch = ax.add_patch(PathPatch(
                        Path(np.array(v), c), facecolor="none", edgecolor=LINE, alpha=0.75,
                        linewidth=lw,
                        linestyle=(0, (2.2, 1.8)) if (lev < 0 and opts.dashed) else "solid",
                        zorder=3))
                    patch.set_clip_path(clip)

    # Letters, drawn like the cylinder of the course pictures
    verts, codes = [], []
    for p in meta["polygons"]:
        verts += p.tolist() + [p[0].tolist()]
        codes += [Path.MOVETO] + [Path.LINETO]*(len(p) - 1) + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(np.array(verts), codes), facecolor=BODY, edgecolor=BODY_EDGE,
                           linewidth=0.8*lw, zorder=4))
    # Counters (inside of o, e, a, ...) hold no fluid: paper colour
    for p in meta["polygons"]:
        if signed_area(p) < 0:
            ax.add_patch(PathPatch(Path(p), facecolor=BG, edgecolor="none", zorder=3.9))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=FRAME,
                           linewidth=1.2*lw, zorder=5))


# --------------------------------------------------------------------------

_G = {}


def _render_one(job):
    i, tname = job
    sampler, meta, opts = _G["sampler"], _G["meta"], _G["opts"]
    z = sampler.grid(meta["render"]["field"], tname)
    if meta["mode"] == "convection":
        r = meta["render"]
        z = (z - r["Tcold"])/(r["Thot"] - r["Tcold"])
    fig, ax = make_axes(meta["view"], opts)
    draw(ax, sampler, z, meta, opts)
    fig.savefig(os.path.join(opts.out, "frames", f"frame_{i:05d}.png"), dpi=200, facecolor=BG)
    if opts.svg_set is not None and i in opts.svg_set:
        fig.savefig(os.path.join(opts.out, "svg", f"frame_{i:05d}.svg"), facecolor=BG)
    plt.close(fig)
    return i


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", help="flowOverText case directory")
    ap.add_argument("--width", type=int, help="frame width in pixels")
    ap.add_argument("--fps", type=int)
    ap.add_argument("--lim", type=float,
                    help="value of the strongest band (forced: vorticity in U/H, default 6;"
                         " convection: theta, default 0.8)")
    ap.add_argument("--sigma", type=float, help="display smoothing in grid points (default 2)")
    ap.add_argument("--h", type=float, help="sampling grid spacing in H (default 0.02)")
    ap.add_argument("--lines", action=argparse.BooleanOptionalAction, default=None,
                    help="contour lines over the bands")
    ap.add_argument("--cmap", help="colour map (default: forced RdBu_r, convection inferno_r)")
    ap.add_argument("--bands", type=int, help="number of colour bands per sign (default 24)")
    ap.add_argument("--dashed", action=argparse.BooleanOptionalAction, default=None,
                    help="dashed lines for negative values (default off)")
    ap.add_argument("--svg", help="SVG frames to write: none, last, all, or every N (e.g. 25)")
    ap.add_argument("--view", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                    help="override the view window (units of H)")
    ap.add_argument("--start", type=float, default=0, help="first time to render")
    ap.add_argument("--end", type=float, default=np.inf, help="last time to render")
    ap.add_argument("--no-movie", action="store_true", help="frames only")
    ap.add_argument("-j", "--jobs", type=int, default=min(16, os.cpu_count() or 1))
    opts = ap.parse_args()

    case = os.path.abspath(opts.case)
    meta = load_case(case)
    forced = meta["mode"] == "forced"
    defaults = {"width": 1920, "fps": 30, "lim": 6.0 if forced else 0.8, "sigma": 2.0,
                "h": 0.02, "lines": forced, "dashed": False, "bands": 24, "svg": "last", "view": None,
                "cmap": "RdBu_r" if forced else "inferno_r"}
    cfgfile = os.path.join(case, "flowOverText.yaml")
    if os.path.isfile(cfgfile):
        with open(cfgfile) as f:
            r = (yaml.safe_load(f) or {}).get("render") or {}
        defaults.update({k: v for k, v in r.items() if k in defaults and v is not None})
    for k, v in defaults.items():
        if getattr(opts, k) is None:
            setattr(opts, k, v)
    opts.svg = str(opts.svg)
    opts.step = 2                     # contours on every 2nd grid point, as the course
    opts.tol = 0.25*opts.h            # simplification tolerance
    if opts.view:
        meta["view"] = list(opts.view)

    field = meta["render"]["field"]
    root0 = foamio.processor_dirs(case)[0]
    times = [(t, n) for t, n in foamio.time_dirs(root0, field) if opts.start <= t <= opts.end]
    if not times:
        sys.exit(f"No time directories with '{field}' in {root0}")

    opts.out = os.path.join(case, "animation")
    shutil.rmtree(os.path.join(opts.out, "frames"), ignore_errors=True)
    os.makedirs(os.path.join(opts.out, "frames"))
    n = len(times)
    if opts.svg == "none":
        opts.svg_set = None
    else:
        shutil.rmtree(os.path.join(opts.out, "svg"), ignore_errors=True)
        os.makedirs(os.path.join(opts.out, "svg"))
        opts.svg_set = ({n - 1} if opts.svg == "last" else set(range(n)) if opts.svg == "all"
                        else set(range(0, n, int(opts.svg))) | {n - 1})

    sampler = Sampler(case, meta["view"], meta["polygons"], opts.h)
    _G.update(sampler=sampler, meta=meta, opts=opts)

    jobs = [(i, name) for i, (t, name) in enumerate(times)]
    print(f"Rendering {n} frames of '{field}' with {opts.jobs} processes ...")
    with mp.get_context("fork").Pool(opts.jobs) as pool:
        for k, _ in enumerate(pool.imap_unordered(_render_one, jobs, chunksize=2)):
            if (k + 1) % 50 == 0 or k + 1 == n:
                print(f"  {k + 1}/{n}", flush=True)

    if not opts.no_movie:
        if shutil.which("ffmpeg") is None:
            print("ffmpeg not found; frames are in", os.path.join(opts.out, "frames"))
            return
        mp4 = os.path.join(opts.out, os.path.basename(case.rstrip("/")) + ".mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(opts.fps),
                        "-i", os.path.join(opts.out, "frames", "frame_%05d.png"),
                        "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",   # H.264: even size
                        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4], check=True)
        print("Movie:", mp4)


if __name__ == "__main__":
    main()
