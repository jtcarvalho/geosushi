"""
WW3 grid generation (replaces NOAA-EMC gridgen / myWW3tools genWW3grid).

Creates the ASCII files read by ``ww3_grid``:

  {prefix}.dep   bathymetry (mm, scale 0.001 -> m)
  {prefix}.mask  land/sea mask (0 = land, 1 = ocean)
  {prefix}.obs   sub-grid obstacles (zeros; see geosushi.ww3.obstruction)
  {prefix}.meta  grid block for ww3_grid.inp
  {prefix}.lon / .lat   coordinates (curvilinear grids only)

Python
------
>>> from geosushi.ww3 import make_grid
>>> make_grid(lon_min=-46, lon_max=-38, lat_min=-26, lat_max=-18,
...           dx=0.05, dy=0.05, bathy_file="gebco.nc",
...           output_dir="rio_grid", prefix="rio")

Terminal
--------
    geosushi ww3 grid config.yaml
"""

import os

import numpy as np
import xarray as xr

from ..common.bathymetry import (load_bathymetry, interpolate_to_regular,
                                 interpolate_to_curvilinear)
from ..common.config import load_config, cfg_get
from .meta import write_meta, write_meta_curv


def make_mask(elevation, zdep=-0.10):
    """
    Build ocean/land mask from elevation (metres, negative = ocean).
      0 = land  (elevation >= zdep)
      1 = ocean (elevation <  zdep)

    Boundary-condition marking (value 2) is intentionally left to the
    geosushi.ww3.mask.add_boundary(), matching the convention in the examples.
    """
    mask = np.where(elevation < zdep, 1, 0).astype(int)
    return mask


# ============================================================================
# File writers
# ============================================================================

LAND_VALUE_MM = 999999000   # sentinel for land points in dep file

def write_dep(filepath, elevation_m, nx, ny):
    """
    Write WW3 bathymetry file.
    Values are stored as integers in millimetres (scale factor 0.001 → m).
    Layout: IDLA=1, first row = southernmost latitude (bottom-to-top order).
    Land sentinel: 999999000 mm (= 999999 m).
    """
    data_mm = np.round(elevation_m * 1000).astype(np.int64)
    # Replace unrealistic values (fill from interpolation) with land sentinel
    data_mm[data_mm > 900000000] = LAND_VALUE_MM

    with open(filepath, 'w') as f:
        for j in range(ny):
            line = ''.join(f'{data_mm[j, i]:12d}' for i in range(nx))
            f.write(line + '\n')


def write_mask(filepath, mask, nx, ny):
    """
    Write WW3 mask file.
    0 = land, 1 = ocean.  Layout: IDLA=1 (south to north).
    """
    with open(filepath, 'w') as f:
        for j in range(ny):
            line = ''.join(f'{mask[j, i]:3d}' for i in range(nx))
            f.write(line + '\n')


def write_obs(filepath, nx, ny):
    """
    Write WW3 sub-grid obstacles file (all zeros).
    Shape: (2*NY, NX) — rows 0..NY-1 for E-W faces, NY..2NY-1 for N-S faces.
    Values 0-100 represent % blockage; all zeros = no obstacles.
    """
    with open(filepath, 'w') as f:
        for j in range(2 * ny):
            line = ''.join(f'{0:3d}' for _ in range(nx))
            f.write(line + '\n')


def write_curv_coord(filepath, data2d, nx, ny):
    """
    Write a WW3 curvilinear coordinate file (.lon or .lat).
    Format: IDLA=1, one row per latitude index (south to north),
    free-format floating-point (12.6f per value).
    """
    with open(filepath, 'w') as f:
        for j in range(ny):
            line = ''.join(f'{data2d[j, i]:12.6f}' for i in range(nx))
            f.write(line + '\n')


def build_curv_from_file(curv_file, lon_var='lon', lat_var='lat'):
    """
    Load a curvilinear grid from a NetCDF file containing 2-D lon/lat arrays.

    curv_file : NetCDF with 1-D or 2-D lon/lat (e.g. a ROMS grid file)
    lon_var, lat_var : variable names (common alternatives are tried)

    Returns lon2d, lat2d  both of shape (ny, nx)
    """
    if curv_file is None:
        raise ValueError(
            "grid.curv_file must be set when grid.type = 'curv' "
            "and grid.projection is not specified.")
    if not os.path.exists(curv_file):
        raise FileNotFoundError(f"Curvilinear grid file not found: {curv_file}")


    print(f"  Loading curvilinear grid from: {curv_file}")
    ds = xr.open_dataset(curv_file)

    if lon_var not in ds and lat_var not in ds:
        # Try common alternatives
        for lv in ['longitude', 'LONGITUDE', 'lon_rho', 'nav_lon', 'TLON', 'XLONG', 'x', 'X']:
            if lv in ds:
                lon_var = lv
                break
        for lv in ['latitude', 'LATITUDE', 'lat_rho', 'nav_lat', 'TLAT', 'XLAT', 'y', 'Y']:
            if lv in ds:
                lat_var = lv
                break

    lon2d = ds[lon_var].values.astype(float)
    lat2d = ds[lat_var].values.astype(float)

    if lon2d.ndim == 1:
        # 1-D arrays → make 2-D meshgrid
        lon2d, lat2d = np.meshgrid(lon2d, lat2d)

    ny, nx = lon2d.shape
    print(f"  Curvilinear grid size: {nx} x {ny}")
    print(f"  lon range: [{lon2d.min():.3f}, {lon2d.max():.3f}]°")
    print(f"  lat range: [{lat2d.min():.3f}, {lat2d.max():.3f}]°")
    return lon2d, lat2d


def build_curv_stereo(center_lon=0.0, center_lat=-90.0, nx=100, ny=100, dx_km=25.0):
    """
    Build a polar stereographic curvilinear grid.

    Config keys used:
        grid.center_lon  : central longitude (degrees, default 0)
        grid.center_lat  : central latitude  (degrees, default -90 = south pole)
        grid.nx          : number of columns
        grid.ny          : number of rows
        grid.dx_km       : grid spacing in km (uniform, default 25)

    Returns lon2d, lat2d  both of shape (ny, nx)
    """
    clon = float(center_lon)
    clat = float(center_lat)
    nx   = int(nx)
    ny   = int(ny)
    dkm  = float(dx_km)

    print(f"  Building polar stereographic grid:")
    print(f"    centre = ({clon}°, {clat}°)  |  "
          f"{nx} x {ny} cells  |  spacing = {dkm} km")

    # Grid coordinates in km, centred at origin
    x_km = (np.arange(nx) - (nx - 1) / 2.0) * dkm
    y_km = (np.arange(ny) - (ny - 1) / 2.0) * dkm
    X, Y = np.meshgrid(x_km, y_km)

    # Inverse stereographic projection on a sphere, k0 = 1 at the centre
    # Reference: Snyder (1987) Map Projections - A Working Manual, eqs. 20-14/20-15
    R_earth = 6371.0  # km
    phi1 = np.radians(clat)
    lam0 = np.radians(clon)

    rho = np.hypot(X, Y)
    rho = np.where(rho == 0, 1e-12, rho)   # avoid 0/0 at the centre
    c = 2.0 * np.arctan2(rho, 2.0 * R_earth)

    lat_r = np.arcsin(np.cos(c) * np.sin(phi1)
                      + Y * np.sin(c) * np.cos(phi1) / rho)
    lon_r = lam0 + np.arctan2(X * np.sin(c),
                              rho * np.cos(phi1) * np.cos(c)
                              - Y * np.sin(phi1) * np.sin(c))

    lat2d = np.degrees(lat_r)
    lon2d = np.degrees(lon_r)
    # Wrap longitude to -180..180
    lon2d = ((lon2d + 180) % 360) - 180

    print(f"  lon range: [{lon2d.min():.3f}, {lon2d.max():.3f}]°")
    print(f"  lat range: [{lat2d.min():.3f}, {lat2d.max():.3f}]°")
    return lon2d, lat2d


# ============================================================================
# High-level API
# ============================================================================

def _write_common(output_dir, prefix, elevation, mask, nx, ny):
    files = {
        "dep": os.path.join(output_dir, f"{prefix}.dep"),
        "mask": os.path.join(output_dir, f"{prefix}.mask"),
        "obs": os.path.join(output_dir, f"{prefix}.obs"),
    }
    write_dep(files["dep"], elevation, nx, ny)
    write_mask(files["mask"], mask, nx, ny)
    write_obs(files["obs"], nx, ny)
    return files


def make_grid(lon_min=None, lon_max=None, lat_min=None, lat_max=None,
              dx=None, dy=None, bathy_file=None, output_dir="ww3_grid",
              prefix="grid", zdep=-0.10, zmin=2.50, bathy_source="file",
              elev_var=None, pygmt_resolution="01m",
              grid_type="rect", lon2d=None, lat2d=None, verbose=True):
    """
    Generate WW3 grid files.

    Rectilinear (default): give lon/lat bounds and dx, dy (degrees).
    Curvilinear: ``grid_type='curv'`` and 2-D ``lon2d``, ``lat2d`` arrays
    (see build_curv_from_file / build_curv_stereo).

    Returns
    -------
    dict with the paths written ('dep', 'mask', 'obs', 'meta'[, 'lon', 'lat'])
    plus 'nx', 'ny', 'elevation' (m) and 'mask_array'.
    """
    os.makedirs(output_dir, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    if grid_type == "curv":
        if lon2d is None or lat2d is None:
            raise ValueError("curvilinear grids need lon2d and lat2d")
        ny, nx = lon2d.shape
        bounds = (float(np.nanmin(lon2d)), float(np.nanmax(lon2d)),
                  float(np.nanmin(lat2d)), float(np.nanmax(lat2d)))
        log(f"-> CURV grid {nx} x {ny}")
        bathy = load_bathymetry(bathy_file, *bounds, source=bathy_source,
                                variable=elev_var,
                                pygmt_resolution=pygmt_resolution)
        elevation = interpolate_to_curvilinear(bathy, lon2d, lat2d)
    else:
        if None in (lon_min, lon_max, lat_min, lat_max, dx, dy):
            raise ValueError("lon_min, lon_max, lat_min, lat_max, dx, dy are required")
        # floor (never exceed the requested max); the epsilon avoids losing a
        # row to float error when the span is an exact multiple of dx
        nx = int(np.floor((lon_max - lon_min) / dx + 1e-9)) + 1
        ny = int(np.floor((lat_max - lat_min) / dy + 1e-9)) + 1
        lon1d = lon_min + np.arange(nx) * dx
        lat1d = lat_min + np.arange(ny) * dy
        log(f"-> RECT grid {nx} x {ny}  dx={dx} dy={dy}")
        bathy = load_bathymetry(bathy_file, lon_min, lon_max, lat_min, lat_max,
                                source=bathy_source, variable=elev_var,
                                pygmt_resolution=pygmt_resolution)
        elevation = interpolate_to_regular(bathy, lon1d, lat1d)

    mask = make_mask(elevation, zdep=zdep)
    n_ocean = int((mask == 1).sum())
    log(f"   ocean cells: {n_ocean} / {nx * ny} ({100 * n_ocean / (nx * ny):.1f}%)")
    if n_ocean == 0:
        log("   WARNING: all cells are land - check bounds and bathymetry")

    files = _write_common(output_dir, prefix, elevation, mask, nx, ny)
    files["meta"] = os.path.join(output_dir, f"{prefix}.meta")
    if grid_type == "curv":
        files["lon"] = os.path.join(output_dir, f"{prefix}.lon")
        files["lat"] = os.path.join(output_dir, f"{prefix}.lat")
        write_curv_coord(files["lon"], lon2d, nx, ny)
        write_curv_coord(files["lat"], lat2d, nx, ny)
        write_meta_curv(files["meta"], nx, ny, prefix, zdep=zdep, zmin=zmin)
    else:
        write_meta(files["meta"], nx, ny, dx, dy, lon1d[0], lat1d[0], prefix,
                   zdep=zdep, zmin=zmin)

    for k in ("dep", "mask", "obs", "meta", "lon", "lat"):
        if k in files:
            log(f"   wrote {files[k]}")
    files.update(nx=nx, ny=ny, elevation=elevation, mask_array=mask)
    return files


def make_grid_from_config(config):
    """
    Same as make_grid, driven by a YAML file (path or already-loaded dict).
    See ``geosushi/ww3/confs/grid.yaml`` for all keys.
    """
    cfg = load_config(config) if isinstance(config, str) else config
    base = os.path.dirname(os.path.abspath(config)) if isinstance(config, str) else os.getcwd()

    def path(p):  # paths in the YAML are relative to the YAML itself
        return p if p is None or os.path.isabs(p) else os.path.join(base, p)

    g = lambda k, d=None: cfg_get(cfg, k, d)
    grid_type = str(g("grid.type", "rect")).lower()
    kw = dict(
        bathy_file=path(g("bathymetry.file")),
        bathy_source=g("bathymetry.source", "file"),
        elev_var=g("bathymetry.variable"),
        pygmt_resolution=g("bathymetry.pygmt_resolution", "01m"),
        output_dir=path(g("output.directory", "ww3_grid")),
        prefix=g("output.prefix", "grid"),
        zdep=float(g("grid.zdep", -0.10)),
        zmin=float(g("grid.zmin", 2.50)),
    )
    if grid_type == "curv":
        if str(g("grid.projection", "")).lower() == "stereo":
            lon2d, lat2d = build_curv_stereo(
                g("grid.center_lon", 0.0), g("grid.center_lat", -90.0),
                g("grid.nx", 100), g("grid.ny", 100), g("grid.dx_km", 25.0))
        else:
            lon2d, lat2d = build_curv_from_file(
                path(g("grid.curv_file")),
                g("grid.curv_lon_var", g("grid.lon_var", "lon")),
                g("grid.curv_lat_var", g("grid.lat_var", "lat")))
        return make_grid(grid_type="curv", lon2d=lon2d, lat2d=lat2d, **kw)

    return make_grid(lon_min=float(g("grid.lon_min")), lon_max=float(g("grid.lon_max")),
                     lat_min=float(g("grid.lat_min")), lat_max=float(g("grid.lat_max")),
                     dx=float(g("grid.dx")), dy=float(g("grid.dy")), **kw)


# Old geosushi name (waves/gera_reg_ww3grid.py)
gera_grade_regular = make_grid
