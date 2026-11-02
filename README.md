# flowOverText

2D flow over a word or a logo, entirely in OpenFOAM (v2606), with a vector
graphics animation at the end. Two kinds of flow: a uniform stream from the
left that sheds vortices off the letters, and natural convection from hot
letters in cold air at rest.

[results/](results/) has example movies, each with its last frame as SVG.

```
text / SVG ──► outlines ──► ribbon STL ──► blockMesh (graded) ──► snappyHexMesh ──► extrudeMesh (2D)
                                                                                        │
                            mp4 + SVG frames ◄── render.py ◄── pimpleFoam / buoyantPimpleFoam
```

## Quick start

```sh
source ~/OpenFOAM/OpenFOAM-v2606/etc/bashrc

./flowOverText.py examples/saeed.yaml --run                          # forced flow
./flowOverText.py examples/saeed.yaml --set mode=convection --run    # hot letters
./flowOverText.py examples/LiU.yaml --run                            # SVG logo
./flowOverText.py examples/saeed.yaml --preview                      # just look at the outlines
```

The case is created in `cases/<name>_<mode>/` (or `case:` in the YAML), and the
movie ends up in its `animation/` folder.

## Configuration

Each run is described by one YAML file, and
[examples/saeed.yaml](examples/saeed.yaml) lists every setting with comments.
Anything you leave out falls back to `DEFAULTS` in
[flowOverText.py](flowOverText.py). `--set key=value` overrides a single
entry from the command line (for example `--set forced.Re=2000` or
`--set mesh.cells=40000`), which is handy for variations.

The case keeps a copy of the resolved configuration as `flowOverText.yaml`.
`render.py` reads its `render:` section from there.

| Section | Key | Default | |
|---|---|---|---|
| geometry | `text` / `svg` | | one line of text, or an SVG file |
| | `font` | `Inter Display:black` | a fontconfig pattern or a font file (`fc-list : family style`) |
| (top level) | `mode` | `forced` | `forced` or `convection` |
| forced | `Re`, `endTime`, `frameDt` | 1000, 30, 0.04 | times in H/U |
| convection | `Ra`, `Pr`, `deltaT`, `endTime`, `framesPerTff` | 1e7, 0.71, 10 K, 40, 20 | times in free-fall times |
| mesh | `cells` | 100000 | target number of 2D cells (the budget) |
| | `layers`, `firstLayer`, `expansion` | 8, 0.2, 1.2 | prism layers on the letters; first layer height relative to the wall cell |
| run | `np`, `cleanup` | 16, true | |
| render | `width`, `fps`, `lim`, `cmap`, `bands`, `lines`, `dashed`, `sigma`, `h`, `svg`, `view` | 1920, 30, per mode, per mode, 24, forced only, false, 2, 0.02, last, per mode | see [Rendering](#rendering) |

## Mesh sizing

The cell sizes come from Re or Ra. All lengths are in units of the text
height H:

| | forced | convection |
|---|---|---|
| physical scale | δ = H/√Re | δ_T = H/(2 Nu), with Nu ≈ 0.54 Ra^¼ |
| wall cell | δ/8 | δ_T / max(3, 0.35 Ra^0.15) (Shishkina et al. 2010) |
| near the letters | δ/2 | δ_T |
| bulk (the whole animation window) | δ | min(π η_K, 1.5 δ_T) with η_K = (Pr²/(Ra Nu))^¼ (Grötzbach) |

These are DNS-like ideal sizes. The budget `mesh.cells` (default 100k)
decides the actual mesh: all three sizes are scaled by one common factor
until the estimated cell count matches the budget, so their ratios stay as
the physics says. The generator prints how far from the ideal the result is
(e.g. "1.7x coarser than the physics estimate"). The estimate is within a few
percent of the final snappyHexMesh count.

The bulk size is set directly by a multi-graded `blockMesh`: it is uniform
over the animation window (for the forced case, only the wake band
|y| < 2 H) and grows by at most 15 % per cell towards a far-field size of 1 H.
snappyHexMesh then only refines near the letters (a `near` box plus distance
bands from the surface) and adds body-fitted prism layers on them: 8 layers
by default, the first one 0.2 times the wall cell, growing by 1.2. Almost the
whole budget goes into the animation window, because the graded far field is
cheap; the layer cells are counted in the budget.

With 100k cells ("Saeed" in Inter Display Black):

* Re = 1000: bulk H/20, near-letter H/39, wall H/156, first layer H/780
  (1.6x coarser than ideal).
* Ra = 1e7: bulk H/23, near-letter H/47, wall H/187, first layer H/934
  (2.1x coarser).
* Ra = 1e8 would be considerably coarser, so raise `mesh.cells` or accept
  smeared plume stems.

`./plotMesh.py <case>` draws the 2D mesh into `<case>/mesh/` (PNG and SVG):
the whole domain, the animation window, the first letter, a near-wall
close-up and a sharp corner.

The values used are written at the top of the case's `system/caseParameters`.

## The two modes

**forced** (`pimpleFoam`, laminar): uniform flow U = 1 from the left, slip
top and bottom, an `inletOutlet` outlet. The domain runs from 5 H upstream to
15 H downstream, and ±6 H vertically. The animation shows vorticity.

**convection** (`buoyantPimpleFoam`, laminar, Boussinesq, `dpdt no`): air at
rest at T_c = 300 K and letters at T_c + ΔT, with gravity in −y. ν follows
from Ra with H = 1 m. The side walls are adiabatic no-slip walls; the bottom
and top are open (`totalPressure` on p_rgh, `pressureInletOutletVelocity`,
`inletOutlet` on T). The animation shows θ = (T − T_c)/ΔT. The view window is
square-ish and puts the text in the lower third. The side walls are
2.5 H from the text, and the open top is 5 H above the window.

## Case scripts

| Script | Does |
|---|---|
| `Allrun.pre` | blockMesh, surfaceFeatureExtract, snappyHexMesh (parallel), extrudeMesh → 2D, checkMesh, decomposePar, cell centres |
| `Allrun.run` | the solver (parallel) |
| `Allrun.render` | `render.py .` (extra options are passed on) |
| `Allrun.cleanup` | removes the field data, the mesh and the PNG frames once the mp4 exists; keeps the movie, the SVG frames, `postProcessing` and the setup (logs are removed too) |
| `Allrun` | pre, run, render, plus cleanup if `run.cleanup: true` |

A 100k-cell case is roughly 0.5–1 GB before cleanup.
**Re-rendering needs the field data**, so set `run.cleanup: false` while you
experiment with colours or framing, and run `./Allrun.cleanup` once you're
happy.

## Rendering

[render.py](render.py) draws the animation with matplotlib, with no ParaView
and no reconstructPar. The style follows the cylinder-wake figures of the
TMMV64 course at Linköping University: filled bands that fade into a paper
colour near zero, smooth vector contours, and the body in grey.

1. The field (cell centres plus wall faces, read from `processor*/`) is
   interpolated linearly onto a uniform grid (spacing `h`, default 0.02 H)
   over the view window. The inside of the letters is filled with the
   nearest wall value, so the bands run right up to the outline.
2. Gaussian display smoothing (`sigma`, default 2 grid points).
3. `bands` filled bands per sign (default 24), evenly spaced up to `lim` and
   faded into the paper colour near zero. Optional contour lines (`lines`, on
   by default for forced flow; `dashed` for negative values). Contours are
   computed on every second grid point, simplified (Ramer–Douglas–Peucker)
   and drawn as Catmull–Rom curves.
4. The letters are drawn grey with a darker outline, the counters (inside of
   o, e, a, ...) in the paper colour, and the frame as a thin line.

Forced flow shows ω_z in U/H with RdBu_r (default `lim` 6), in a 2.25:1
window. Convection shows θ = (T − T_c)/ΔT with inferno_r (default `lim` 0.8),
in a square window with the text in the lower third.

```sh
./render.py cases/Saeed_forced                       # mp4 + last frame as SVG
./render.py cases/Saeed_forced --svg all --no-lines
./render.py cases/Saeed_convection --lim 1 --cmap magma_r --width 2560
./render.py cases/Saeed_forced --view -1 14 -3 3
```

The output is `animation/<case>.mp4` (H.264), `animation/svg/` (vector
frames: the last by default, or `svg: all` / `svg: 25`), and
`animation/frames/*.png`.

## SVG input

The parser handles filled `path`, `rect`, `circle`, `ellipse`, `polygon` and
`polyline` elements, with nested `transform`s. Strokes, `<text>`, clip paths
and masks are ignored, so convert those to paths first (in Inkscape: *Object
to Path*, *Stroke to Path*). Shapes that overlap should be merged (*Path ▸
Union*); the script warns when it finds overlaps.

## Notes

* Variable fonts (e.g. `UbuntuSans[wdth,wght].ttf`) only give their default
  instance; pick a static file such as `Inter Display:black`.
  [examples/font_samples.png](examples/font_samples.png) shows a few of the
  installed fonts.
* Closed counters (the inside of o, e, a, …) contain no fluid, because snappy's
  `locationInMesh` removes them. The renderer fills them with the
  undisturbed-fluid colour (forced) or the hot colour (convection).
* The `vortZ` field from `exprField` has zeroGradient walls, so the renderer
  takes the wall vorticity from the wall-adjacent cell.
