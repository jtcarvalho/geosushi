# Geo<span style="color:#ff006e;">SUSHI</span>

## Geoscience **S**cientific **U**tilities for **S**cripting, **H**andling and **I**nference

My day-to-day toolbox for metocean models — **WW3, SWAN, ROMS and WRF** — in one
place, so scripts don't get lost. Every tool is both a Python function and a
terminal command.

## Install

```bash
git clone https://github.com/jtcarvalho/geosushi.git
cd geosushi
pip install -e .          # or: poetry install
geosushi --help
```

`-e` (editable) means changes you make to the code are picked up immediately.

## Layout (one folder per model)

```
src/geosushi/
├── cli.py            ← the `geosushi` terminal command
├── common/           ← shared by all models
│   ├── config.py         YAML helpers
│   ├── bathymetry.py     GEBCO/ETOPO/pygmt loading + interpolation
│   ├── spectra.py        directional spectrum E(f,θ) polar plot
│   └── tools.py          plots, Taylor diagram, time/space selection
├── obs/              ← observations: PNBOIA buoys, OSCAR currents
├── ww3/              ← WAVEWATCH III
│   ├── grid.py           grid generation (rect, curvilinear, polar stereo)
│   ├── mask.py           open-boundary points (mask value 2)
│   ├── obstruction.py    sub-grid obstructions (.obs)
│   ├── boundary.py       boundary point list for nesting (.list)
│   ├── meta.py           read/write .meta (ww3_grid.inp block)
│   ├── io.py             read WW3 outputs (fields, points)
│   ├── validation.py     model x buoy metrics
│   └── confs/            YAML templates
├── swan/             ← (to be filled)
├── roms/             ← (to be filled)
└── wrf/              ← (to be filled)
```

## Cheat sheet

| Task | Terminal | Python |
|---|---|---|
| List all commands | `geosushi list` | |
| Copy a config template | `geosushi template ww3-grid` | |
| WW3 grid (.dep .mask .obs .meta) | `geosushi ww3 grid grid.yaml` | `ww3.make_grid_from_config("grid.yaml")` |
| Open-boundary mask (.mask2) | `geosushi ww3 mask out/rio.mask` | `ww3.add_boundary("out/rio.mask")` |
| Sub-grid obstructions | `geosushi ww3 obstr out/rio.meta` | `ww3.compute_obstructions("out/rio.meta")` |
| BC point list (.list) | `geosushi ww3 bc out/rio.mask2 --id bnd` | `ww3.boundary_points("out/rio.mask2")` |
| **All four steps** | `geosushi ww3 prep grid.yaml` | |
| Directional spectrum plot | | `common.spectra.plot_directional_spectrum(E, f, d)` |

`geosushi ww3 mask` accepts `--sides south east ...` when not every edge is open.

### Example: new WW3 grid in 3 commands

```bash
mkdir rio && cd rio
geosushi template ww3-grid          # creates grid.yaml — edit bounds, dx, GEBCO path
geosushi ww3 prep grid.yaml         # grid + mask2 + obs + list + figures
```

From Python (notebooks):

```python
from geosushi import ww3
files = ww3.make_grid(lon_min=-46, lon_max=-38, lat_min=-26, lat_max=-18,
                      dx=0.05, dy=0.05, bathy_file="gebco.nc",
                      output_dir="rio_grid", prefix="rio")
```

A ROMS grid can be reused as a curvilinear WW3 grid (`grid.type: curv`,
`grid.curv_file: roms_grd.nc` — `lon_rho/lat_rho` are detected automatically).

## Adding a new tool

1. Put the function in the model folder (e.g. `src/geosushi/roms/forcing.py`).
2. Export it in that folder's `__init__.py`.
3. Optional: add a sub-command in `cli.py` (copy one of the `ww3` blocks).
4. Add a small test in `tests/` and run `pytest`.

## History

The WW3 pre-processing tools came from the
[myWW3tools](https://github.com/jtcarvalho/myWW3tools) repository (a Python
replacement for NOAA-EMC gridgen); see `docs/ww3_preprocessing.md`.
