#!/usr/bin/env python3
"""
flowOverText: build (and optionally run) a 2D OpenFOAM case of flow over a
word or an SVG logo.

Usage
  ./flowOverText.py run.yaml                       # create the case
  ./flowOverText.py run.yaml --run                 # ... and mesh/solve/render
  ./flowOverText.py run.yaml --set mode=convection --set convection.Ra=1e7
  ./flowOverText.py run.yaml --preview             # only draw the outlines

All settings live in a YAML file (see examples/*.yaml); anything not given
there takes the defaults in DEFAULTS below.  The resolved configuration is
stored in the case as flowOverText.yaml (render.py reads it from there).

Modes
  forced      uniform flow from the left (pimpleFoam), Re = U H / nu
  convection  hot letters in cold air at rest (buoyantPimpleFoam, Boussinesq),
              Ra = g beta dT H^3 / (nu alpha)
H is the total height of the text/logo, which is scaled to 1.
"""

import argparse
import copy
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ftools import geometry  # noqa: E402

TEMPLATES = os.path.join(HERE, "templates")

DEFAULTS = {
    "case": None,                       # default: cases/<name>_<mode>
    "geometry": {
        "text": None,
        "svg": None,
        "font": "Inter Display:black",  # font file or fontconfig pattern
    },
    "mode": "forced",
    "forced": {
        "Re": 1000,
        "endTime": 30,                  # in H/U
        "frameDt": 0.04,                # in H/U
    },
    "convection": {
        "Ra": 1e7,
        "Pr": 0.71,
        "deltaT": 10,                   # K, hot letters minus cold air
        "endTime": 40,                  # in free-fall times t_ff
        "framesPerTff": 20,
    },
    "mesh": {
        "cells": 100000,                # target number of 2D cells
        "layers": 8,                    # prism layers on the letters
        "firstLayer": 0.2,              # first layer height / wall cell size
        "expansion": 1.2,               # layer growth ratio
    },
    "run": {
        "np": 16,
        "cleanup": True,                # remove the bulky data after rendering
    },
    "render": {                         # style of the TMMV64 wake pictures
        "width": 1920,                  # pixels
        "fps": 30,
        "lim": None,                    # strongest band: forced |omega| H/U (6), convection theta (0.8)
        "sigma": 2.0,                   # display smoothing, grid points
        "h": 0.02,                      # sampling grid spacing, units of H
        "bands": 24,                    # colour bands per sign
        "cmap": None,                   # forced RdBu_r, convection inferno_r
        "lines": None,                  # contour lines (default: forced on, convection off)
        "dashed": False,                # dashed lines for negative values
        "svg": "last",                  # none | last | all | every N
        "view": None,                   # [x0, x1, y0, y1] in units of H
    },
}



# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _coerce(v):
    """YAML 1.1 reads '1e8' as a string; turn numeric strings into numbers."""
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    return v


def load_config(path, sets):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for s in sets:
        key, _, val = s.partition("=")
        d = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = yaml.safe_load(val)
    cfg = _coerce(cfg)
    g = cfg["geometry"]
    if bool(g["text"]) == bool(g["svg"]):
        sys.exit("Give exactly one of geometry.text or geometry.svg")
    if cfg["mode"] not in ("forced", "convection"):
        sys.exit("mode must be 'forced' or 'convection'")
    if g["svg"] and path:
        g["svg"] = os.path.abspath(os.path.join(os.path.dirname(path), g["svg"])) \
            if not os.path.isfile(g["svg"]) else os.path.abspath(g["svg"])
    return cfg


# --------------------------------------------------------------------------
# Physics-based mesh sizing (all lengths in units of H)
# --------------------------------------------------------------------------

def sizing(cfg):
    """Physics-based cell sizes: wall, near the letters, bulk of the view window.

    These are the (DNS-like) ideal sizes; plan_mesh() scales all of them by
    one factor to meet the cell budget.
    """
    if cfg["mode"] == "forced":
        Re = cfg["forced"]["Re"]
        delta = 1/math.sqrt(Re)             # boundary/shear layer thickness scale
        s = {"delta": delta,
             "wall": delta/8,               # ~8 cells across the wall layer
             "near": delta/2,               # shear layers leaving the letters
             "bulk": delta,                 # wake vortices
             "note": f"delta = H/sqrt(Re) = H/{1/delta:.0f}"}
    else:
        c = cfg["convection"]
        Ra, Pr = c["Ra"], c["Pr"]
        Nu = 0.54*Ra**0.25                  # heated surface, laminar correlation
        deltaT = 1/(2*Nu)                   # thermal boundary layer thickness
        nBL = max(3.0, 0.35*Ra**0.15)       # points in the BL (Shishkina et al. 2010)
        eta = (Pr**2/(Ra*Nu))**0.25         # Kolmogorov/Batchelor length
        s = {"delta": deltaT,
             "wall": deltaT/nBL,
             "near": deltaT,                # plume stems
             "bulk": min(math.pi*eta, deltaT*1.5),   # Groetzbach: h <= pi eta
             "note": f"Nu ~ {Nu:.0f}, delta_T = H/{1/deltaT:.0f},"
                     f" {nBL:.1f} cells in delta_T, pi*eta = H/{1/(math.pi*eta):.0f}"}
    s["far"] = 1.0                          # far-field cap (grading 15 %/cell)
    return s


def graded_axis(a, c0, c1, b, dcore, dfar, rmax=1.15):
    """Multi-section blockMesh grading: coarse -> uniform core -> coarse.

    Returns (nCells, grading list string, first cell size at a).
    """
    def side(L):
        if L <= dcore:
            return 1, 1.0, L
        n = 1
        while True:
            if n*dcore >= L:
                return n, 1.0, L/n
            lo, hi = 1.0, 2.0                       # solve dcore (r^n-1)/(r-1) = L
            for _ in range(80):
                r = 0.5*(lo + hi)
                S = dcore*(r**n - 1)/(r - 1)
                lo, hi = (r, hi) if S < L else (lo, r)
            if r <= rmax and dcore*r**(n - 1) <= dfar:
                return n, r**(n - 1), dcore*r**(n - 1)
            n += 1

    nl, gl, first = side(c0 - a)
    nc = max(1, math.ceil((c1 - c0)/dcore))
    nr, gr, _ = side(b - c1)
    n = nl + nc + nr
    L = b - a
    grading = (f"(({(c0 - a)/L:.6g} {nl/n:.6g} {1/gl:.6g})"
               f" ({(c1 - c0)/L:.6g} {nc/n:.6g} 1)"
               f" ({(b - c1)/L:.6g} {nr/n:.6g} {gr:.6g}))")
    return n, grading, first


def level(dbase, target):
    return max(0, math.ceil(math.log2(dbase/target) - 0.1))


# --------------------------------------------------------------------------
# Case parameters per mode
# --------------------------------------------------------------------------

def plan_mesh(s, f, domain, core, near, perimeter, nLayers=0):
    """Mesh parameters for the physics sizes s scaled by f, plus a cell estimate."""
    dcore = s["bulk"]*f
    xa, xb, ya, yb = domain
    nx, xGrading, fx = graded_axis(xa, max(xa, core[0]), min(xb, core[1]), xb, dcore, s["far"])
    ny, yGrading, fy = graded_axis(ya, max(ya, core[2]), min(yb, core[3]), yb, dcore, s["far"])
    lNear = level(dcore, s["near"]*f)
    lSurf = max(lNear, level(dcore, s["wall"]*f))
    dNear = dcore/2**lNear
    dSurf = dcore/2**lSurf

    # Estimate: background + near box + surface bands (each refinement band
    # is ~4 cells wide; the finest one is set by the distance refinement)
    n = nx*ny
    n += (near[1] - near[0])*(near[3] - near[2])*(1/dNear**2 - 1/dcore**2)
    for lv in range(lNear + 1, lSurf + 1):
        d = dcore/2**lv
        width = 6*d if lv == lSurf else 4*d
        n += perimeter*width*0.75/d**2      # 3/4: each new cell replaces 1/4 of a parent
    n += nLayers*perimeter/dSurf            # prism layers
    return {"f": f, "dcore": dcore, "dNear": dNear, "dSurf": dSurf,
            "nx": nx, "ny": ny, "xGrading": xGrading, "yGrading": yGrading,
            "fx": fx, "fy": fy, "lNear": lNear, "lSurf": lSurf, "estimate": int(n)}


def fit_budget(s, target, *args):
    """Smallest scale factor f >= 1 whose estimated cell count fits the target."""
    lo, hi = 1.0, 64.0
    if plan_mesh(s, lo, *args)["estimate"] <= target:
        return plan_mesh(s, lo, *args)
    for _ in range(40):
        mid = (lo*hi)**0.5
        if plan_mesh(s, mid, *args)["estimate"] > target:
            lo = mid
        else:
            hi = mid
    return plan_mesh(s, hi, *args)


def views(cfg, W):
    """Default animation window [x0, x1, y0, y1] (units of H)."""
    if cfg["mode"] == "forced":
        x0, x1 = -1.0, W + 7.0
        h = (x1 - x0)/2.25                  # proportions of the course wake pictures
        return [x0, x1, -h/2, h/2]
    x0, x1 = -0.75, W + 0.75
    width = x1 - x0
    height = min(width, max(5.5, 0.6*width))
    y0 = -0.5 - 0.17*height
    return [x0, x1, y0, y0 + height]


def physics(cfg, W, view):
    if cfg["mode"] == "forced":
        c = cfg["forced"]
        U = 1.0
        p = {
            "patchXMin": "inlet", "patchXMinType": "patch",
            "patchXMax": "outlet", "patchXMaxType": "patch",
            "patchYMin": "bottom", "patchYMinType": "patch",
            "patchYMax": "top", "patchYMaxType": "patch",
            "Re": c["Re"], "nu": U/c["Re"], "Uinf": f"({U:g} 0 0)",
            "Uref": U, "Href": 1.0,
            "endTime": c["endTime"], "frameDt": c["frameDt"],
            "maxCo": 0.8, "deltaT0": 1e-4,
        }
        domain = (-5.0, W + 15.0, -6.0, 6.0)
        core = (view[0] - 0.5, view[1] + 2.0, -2.0, 2.0)   # wake band
        near = (-0.75, W + 2.5, -1.0, 1.0)
        render = {"field": "vortZ", "time": "t U/H", "timeScale": 1.0}
        return p, domain, core, near, render

    c = cfg["convection"]
    g, dT, Tcold, Pr = 9.81, c["deltaT"], 300.0, c["Pr"]
    beta = 1.0/Tcold
    nu = math.sqrt(g*beta*dT*Pr/c["Ra"])        # H = 1 m
    tff = math.sqrt(1.0/(g*beta*dT))            # free-fall time H/U_ff
    p = {
        "patchXMin": "left", "patchXMinType": "wall",
        "patchXMax": "right", "patchXMaxType": "wall",
        "patchYMin": "bottom", "patchYMinType": "patch",
        "patchYMax": "top", "patchYMaxType": "patch",
        "Ra": c["Ra"], "Pr": Pr, "gravity": f"(0 {-g:g} 0)",
        "Tcold": Tcold, "Thot": Tcold + dT, "beta": beta,
        "rho0": 1.0, "Cp": 1005.0, "mu": nu, "nu": nu, "pRef": 1e5,
        "freeFallTime": tff,
        "endTime": round(c["endTime"]*tff, 4),
        "frameDt": round(tff/c["framesPerTff"], 6),
        "maxCo": 0.8, "maxDeltaT": round(tff/50, 6), "deltaT0": 1e-3*tff,
    }
    domain = (-2.5, W + 2.5, -2.5, view[3] + 5.0)
    core = (view[0] - 0.3, view[1] + 0.3, view[2] - 0.3, view[3] + 0.6)
    near = (-0.3, W + 0.3, -0.8, 1.2)
    render = {"field": "T", "time": "t / t_ff", "timeScale": 1/tff,
              "Tcold": Tcold, "Thot": Tcold + dT}
    return p, domain, core, near, render


# --------------------------------------------------------------------------

def fmt(v):
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def vec(x, y, z):
    return f"({fmt(x)} {fmt(y)} {fmt(z)})"


def write_case_parameters(path, p, header):
    with open(path, "w") as f:
        f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n"
                "  flowOverText case parameters (generated from flowOverText.yaml)\n"
                "\\*---------------------------------------------------------------------------*/\n"
                "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
                "    class       dictionary;\n    location    \"system\";\n"
                "    object      caseParameters;\n}\n"
                "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
        for line in header:
            f.write(f"// {line}\n")
        f.write("\n")
        width = max(len(k) for k in p) + 2
        for k, v in p.items():
            f.write(f"{k:<{width}}{fmt(v)};\n")
        f.write("\n// ************************************************************************* //\n")


def write_script(path, lines):
    with open(path, "w") as f:
        f.write("#!/bin/sh\n"
                'cd "${0%/*}" || exit                                # Run from this directory\n'
                "#" + "-"*78 + "\n\n" + "\n".join(lines) + "\n\n" + "#" + "-"*78 + "\n")
    os.chmod(path, 0o755)


def preview(polys, info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    v, c = [], []
    for p in polys:
        v += p.tolist() + [p[0].tolist()]
        c += [Path.MOVETO] + [Path.LINETO]*(len(p) - 1) + [Path.CLOSEPOLY]
    x0, y0, x1, y1 = info["bounds"]
    fig = plt.figure(figsize=(2*(x1 - x0) + 0.4, 2.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.add_patch(PathPatch(Path(v, c), facecolor="#1a1c22", edgecolor="none"))
    ax.set_xlim(x0 - 0.1, x1 + 0.1)
    ax.set_ylim(y0 - 0.1, y1 + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=150)
    print("Preview:", path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="YAML run configuration")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a setting, e.g. --set forced.Re=2000 (repeatable)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing case")
    ap.add_argument("--run", action="store_true", help="run ./Allrun after creating the case")
    ap.add_argument("--preview", action="store_true",
                    help="only draw the outlines to <name>_preview.png and stop")
    a = ap.parse_args()
    if not a.config and not a.set:
        ap.error("give a YAML file and/or --set geometry.text=...")

    cfg = load_config(a.config, a.set)
    gcfg, mode = cfg["geometry"], cfg["mode"]

    # ---- geometry ---------------------------------------------------------
    s = sizing(cfg)
    polys, info, warnings = geometry.make_polygons(
        text=gcfg["text"], svg=gcfg["svg"], font=gcfg["font"], ds=0.75*s["wall"])
    for w in warnings:
        print("Warning:", w, file=sys.stderr)
    x0, y0, x1, y1 = info["bounds"]
    W = x1 - x0

    name = gcfg["text"] if gcfg["text"] else os.path.splitext(os.path.basename(gcfg["svg"]))[0]
    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "text"
    if a.preview:
        preview(polys, info, f"{safe}_preview.png")
        return

    case = os.path.abspath(cfg["case"] or os.path.join(HERE, "cases", f"{safe}_{mode}"))
    if os.path.exists(case):
        if not a.force:
            sys.exit(f"Case {case} exists (use --force to overwrite)")
        shutil.rmtree(case)
    shutil.copytree(os.path.join(TEMPLATES, "common"), case)
    shutil.copytree(os.path.join(TEMPLATES, mode), case, dirs_exist_ok=True)

    # ---- mesh -------------------------------------------------------------
    view = cfg["render"]["view"] or views(cfg, W)
    p, domain, core, near, render = physics(cfg, W, view)

    perimeter = sum(np.sum(np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)) for q in polys)
    mc = cfg["mesh"]
    m = fit_budget(s, mc["cells"], domain, core, near, perimeter, mc["layers"])
    dcore, lNear, lSurf, dSurf = m["dcore"], m["lNear"], m["lSurf"], m["dSurf"]
    nx, ny, xGrading, yGrading, fx, fy = (m[k] for k in ("nx", "ny", "xGrading", "yGrading", "fx", "fy"))
    xa, xb, ya, yb = domain
    za, zb = 0.0, dcore
    d1 = 6*dSurf
    d2 = d1 + 6*2*dSurf
    scale = (f"{m['f']:.1f}x coarser than the physics estimate" if m["f"] > 1.01
             else "at the physics estimate")

    geo = {
        "xMin": xa, "xMax": xb, "yMin": ya, "yMax": yb, "zMin": za, "zMax": zb,
        "zThickness": zb - za, "nx": nx, "ny": ny,
        "xGrading": xGrading, "yGrading": yGrading,
        "nearMin": vec(near[0], near[2], za - dcore), "nearMax": vec(near[1], near[3], zb + dcore),
        "levelSurface": lSurf, "levelNear": lNear,
        "surfaceDistanceLevels": f"(({fmt(d1)} {lSurf}) ({fmt(d2)} {max(lSurf - 1, lNear)}))",
        "locationInMesh": vec(xa + 0.4137*fx, ya + 0.4291*fy, za + 0.4713*(zb - za)),
        "nLayers": mc["layers"],
        "firstLayerThickness": mc["firstLayer"],
        "layerExpansion": mc["expansion"],
        "nProcs": cfg["run"]["np"],
        "Aref": zb - za,
    }
    p = {**geo, **p}
    p["fieldWriteInterval"] = round(p["endTime"]/4, 6)

    header = [f"mode        {mode}",
              f"source      {gcfg['text'] or gcfg['svg']}",
              f"font        {info.get('fontFile', '-')}",
              f"sizing      {s['note']}",
              f"mesh        ~{cfg['mesh']['cells']} cells, {scale}: bulk H/{1/dcore:.0f},"
              f" near H/{2**lNear/dcore:.0f}, wall H/{1/dSurf:.0f}",
              f"layers      {mc['layers']}, first H/{1/(mc['firstLayer']*dSurf):.0f},"
              f" expansion {mc['expansion']}",
              "All lengths are in units of the text height H = 1"
              + (" m" if mode == "convection" else "")]
    write_case_parameters(os.path.join(case, "system", "caseParameters"), p, header)

    # ---- surfaces and metadata -------------------------------------------
    tri = os.path.join(case, "constant", "triSurface")
    os.makedirs(tri, exist_ok=True)
    geometry.write_ribbon_stl(polys, os.path.join(tri, "text.stl"),
                              za - (zb - za), zb + (zb - za))

    info.update({"mode": mode, "view": view, "render": render,
                 "Re": p.get("Re"), "Ra": p.get("Ra")})
    geometry.write_json(polys, info, os.path.join(case, "constant", "geometry.json"))

    with open(os.path.join(case, "flowOverText.yaml"), "w") as f:
        f.write("# Resolved flowOverText configuration of this case\n")
        yaml.safe_dump(cfg, f, sort_keys=False)

    write_script(os.path.join(case, "Allrun.render"),
                 [f'python3 "{os.path.join(HERE, "render.py")}" . "$@"'])
    allrun = ["# Mesh, solve, render" + (", clean up" if cfg["run"]["cleanup"] else ""),
              "", "./Allrun.pre || exit 1", "", "./Allrun.run || exit 1", "",
              "./Allrun.render || exit 1"]
    if cfg["run"]["cleanup"]:
        allrun += ["", "./Allrun.cleanup"]
    write_script(os.path.join(case, "Allrun"), allrun)

    print(f"Case       : {case}")
    print(f"Mode       : {mode}  (" + (f"Re = {p['Re']:g}" if mode == "forced"
                                       else f"Ra = {p['Ra']:g}, Pr = {p['Pr']:g}") + ")")
    if gcfg["text"]:
        print(f"Font       : {info['fontFile']}")
    print(f"Sizing     : {s['note']}")
    print(f"Mesh       : ~{m['estimate']} cells (target {cfg['mesh']['cells']}), {scale}")
    print(f"             bulk H/{1/dcore:.0f}, near H/{2**lNear/dcore:.0f} (level {lNear}),"
          f" wall H/{1/dSurf:.0f} (level {lSurf}); background {nx} x {ny}")
    print(f"             {mc['layers']} prism layers, first layer H/{1/(mc['firstLayer']*dSurf):.0f},"
          f" expansion {mc['expansion']}")
    print(f"Run        : {case}/Allrun")

    if a.run:
        sys.exit(subprocess.call([os.path.join(case, "Allrun")]))


if __name__ == "__main__":
    main()
