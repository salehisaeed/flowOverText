#!/usr/bin/env python3
"""
Pictures of the 2D mesh of a flowOverText case, for judging it.

    ./plotMesh.py cases/Saeed_forced

Draws the cells (the faces of the 'back' patch) of constant/polyMesh and
writes <case>/mesh/ with PNG and SVG pictures of
  domain   the whole domain
  view     the animation window
  letter   the first letter and its surroundings
  wall     a close-up of the wall cells at the first letter
  corner   a close-up of the top right corner of the last letter
"""

import json
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

BG, LINE, BODY = "#f7f6f2", "#3a3c40", "#8a8d93"


def _lists(path, dtype, ncomp):
    """All top-level 'N ( ... )' lists of an OpenFOAM mesh file (ascii or binary)."""
    data = open(path, "rb").read()
    hdr = data[:data.find(b"}") + 1]
    binary = re.search(rb"format\s+binary", hdr) is not None
    i = data.find(b"}") + 1
    out = []
    num = re.compile(rb"\s*(?://[^\n]*\s*)*(\d+)\s*\(")
    while True:
        m = num.search(data, i)
        if not m:
            break
        n = int(m.group(1))
        j = m.end()
        if binary:
            size = np.dtype(dtype).itemsize*n*ncomp
            arr = np.frombuffer(data, dtype, n*ncomp, j)
            i = j + size + 1
        else:
            k = data.find(b")\n)", j) + 2 if ncomp > 1 else data.find(b")", j)
            txt = data[j:k].replace(b"(", b" ").replace(b")", b" ")
            arr = np.array(txt.split(), dtype)
            i = k + 1
        out.append(arr.reshape(n, ncomp) if ncomp > 1 else arr)
    return out


def read_back_faces(case):
    pm = os.path.join(case, "constant", "polyMesh")
    points = _lists(os.path.join(pm, "points"), "<f8", 3)[0]
    offsets, labels = _lists(os.path.join(pm, "faces"), "<i4", 1)[:2]
    btxt = open(os.path.join(pm, "boundary")).read()
    m = re.search(r"\bback\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);", btxt, re.S)
    n, start = int(m.group(1)), int(m.group(2))
    polys = [points[labels[offsets[f]:offsets[f + 1]], :2] for f in range(start, start + n)]
    return polys


def letters(case):
    with open(os.path.join(case, "constant", "geometry.json")) as f:
        g = json.load(f)
    verts, codes = [], []
    for p in g["polygons"]:
        verts += p + [p[0]]
        codes += [Path.MOVETO] + [Path.LINETO]*(len(p) - 1) + [Path.CLOSEPOLY]
    return g, Path(np.array(verts), codes)


def picture(polys, outline, window, out, width_px=1920, lw=0.25):
    x0, x1, y0, y1 = window
    W = x1 - x0
    H = y1 - y0
    dpi = 200
    fig = plt.figure(figsize=(width_px/dpi, width_px*H/W/dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    m = 0.02*max(W, H)
    sel = [p for p in polys if p[:, 0].max() > x0 - m and p[:, 0].min() < x1 + m
           and p[:, 1].max() > y0 - m and p[:, 1].min() < y1 + m]
    ax.add_collection(PolyCollection(sel, facecolors="none", edgecolors=LINE, linewidths=lw))
    ax.add_patch(PathPatch(outline, facecolor=BODY, edgecolor="none", alpha=0.35, zorder=0))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out + ".png", dpi=dpi, facecolor=BG)
    fig.savefig(out + ".svg", facecolor=BG)
    plt.close(fig)
    return len(sel)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    case = os.path.abspath(sys.argv[1])
    polys = read_back_faces(case)
    g, outline = letters(case)
    out = os.path.join(case, "mesh")
    os.makedirs(out, exist_ok=True)

    allp = np.vstack(polys)
    dom = [allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max()]
    first = np.array(g["polygons"][0])
    fx0, fy0 = first.min(axis=0)
    fx1, fy1 = first.max(axis=0)
    X1, Y1 = np.vstack([np.array(p) for p in g["polygons"]]).max(axis=0)
    windows = {
        "domain": dom,
        "view": g["view"],
        "letter": [fx0 - 0.25, fx1 + 0.25, -0.75, 0.75],
        "wall": [fx0 - 0.08, fx0 + 0.17, fy0 + 0.05, fy0 + 0.2],
        "corner": [X1 - 0.15, X1 + 0.1, Y1 - 0.12, Y1 + 0.03],
    }
    print(f"{len(polys)} cells")
    for name, w in windows.items():
        lw = 0.15 if name in ("domain", "view") else 0.35
        n = picture(polys, outline, w, os.path.join(out, name), lw=lw)
        print(f"  {name:7s} {n:7d} cells drawn -> mesh/{name}.png, .svg")


if __name__ == "__main__":
    main()
