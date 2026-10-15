"""
Bathymetry loading and interpolation (GEBCO, ETOPO, SRTM, pygmt, ...).

Shared by the grid generators of WW3, SWAN and ROMS.

Example
-------
>>> from geosushi.common.bathymetry import load_bathymetry, interpolate_to_regular
>>> da = load_bathymetry("gebco.nc", lon_min=-46, lon_max=-38, lat_min=-26, lat_max=-18)
>>> elev = interpolate_to_regular(da, lon1d, lat1d)
"""

import os

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, griddata


def _normalize_coords(da):
    """
    Rename dimension/coordinate names so that the DataArray always
    has dims named 'lat' and 'lon' (1-D, monotonically increasing).
    """
    rename = {}
    for dim in da.dims:
        d = dim.lower()
        if d in ('latitude', 'y'):
            rename[dim] = 'lat'
        elif d in ('longitude', 'x'):
            rename[dim] = 'lon'
    if rename:
        da = da.rename(rename)
    # Also rename coordinates if they differ from dims
    coord_rename = {}
    for coord in da.coords:
        c = coord.lower()
        if c in ('latitude', 'y') and coord not in da.dims:
            coord_rename[coord] = 'lat'
        elif c in ('longitude', 'x') and coord not in da.dims:
            coord_rename[coord] = 'lon'
    if coord_rename:
        da = da.rename(coord_rename)
    return da


def _sort_coords(da):
    """Ensure lat and lon are monotonically increasing."""
    if 'lat' in da.dims and da['lat'].values[0] > da['lat'].values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    if 'lon' in da.dims and da['lon'].values[0] > da['lon'].values[-1]:
        da = da.isel(lon=slice(None, None, -1))
    return da


def load_bathy_from_file(filepath, elev_var=None, lon_min=None, lon_max=None,
                         lat_min=None, lat_max=None, pad=1.0):
    """
    Load bathymetry from a local NetCDF file (GEBCO, ETOPO, etc.).
    Subsets to the requested region (+ pad degrees) for efficiency.

    Returns a DataArray with dims ('lat', 'lon') and elevation in metres
    (negative = ocean, positive = land).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Bathymetry file not found: {filepath}")

    print(f"  Opening bathymetry file: {filepath}")
    ds = xr.open_dataset(filepath)

    # Auto-detect elevation variable
    if elev_var is None:
        for candidate in ['elevation', 'z', 'altitude', 'depth',
                          'topo', 'height', 'Band1']:
            if candidate in ds.data_vars:
                elev_var = candidate
                break
    if elev_var is None:
        raise ValueError(
            f"Cannot detect elevation variable. Available: {list(ds.data_vars)}\n"
            "Pass variable=... (or bathymetry.variable in the YAML config)."
        )

    da = ds[elev_var].squeeze()   # drop any size-1 dims (e.g., time)
    da = _normalize_coords(da)
    da = _sort_coords(da)

    # Subset to region
    if lat_min is not None:
        lat_lo = max(float(da['lat'].min()), lat_min - pad)
        lat_hi = min(float(da['lat'].max()), lat_max + pad)
        da = da.sel(lat=slice(lat_lo, lat_hi))

    if lon_min is not None:
        # Handle 0-360 vs -180-180 mismatch
        src_lon_min = float(da['lon'].min())
        src_lon_max = float(da['lon'].max())

        req_lon_min = lon_min - pad
        req_lon_max = lon_max + pad

        if src_lon_min >= 0 and req_lon_min < 0:
            # Source is 0-360, requested is negative → convert request
            req_lon_min += 360
            req_lon_max += 360
        elif src_lon_min < 0 and req_lon_min > 180:
            # Source is -180-180, requested >180 → convert request
            req_lon_min -= 360
            req_lon_max -= 360

        req_lon_min = max(src_lon_min, req_lon_min)
        req_lon_max = min(src_lon_max, req_lon_max)
        da = da.sel(lon=slice(req_lon_min, req_lon_max))

    print(f"  Variable '{elev_var}' | Region subset: "
          f"lat [{float(da.lat.min()):.2f}, {float(da.lat.max()):.2f}], "
          f"lon [{float(da.lon.min()):.2f}, {float(da.lon.max()):.2f}]")
    return da


def load_bathy_pygmt(lon_min, lon_max, lat_min, lat_max, resolution='01m'):
    """
    Load bathymetry via pygmt (downloads GEBCO/SRTM15+ tiles as needed).
    resolution: GMT grid resolution string, e.g. '01m' (1 arc-min), '30s', '15s'.
    """
    try:
        import pygmt
    except ImportError:
        raise ImportError(
            "pygmt is not installed. Install with: pip install pygmt\n"
            "Also requires GMT (https://www.generic-mapping-tools.org/)"
        )
    pad = 0.5
    region = [lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad]
    print(f"  Downloading bathymetry via pygmt (resolution={resolution}, region={region})")
    grid = pygmt.datasets.load_earth_relief(resolution=resolution, region=region)
    # pygmt returns an xarray DataArray with dims 'lat'/'lon'
    da = _normalize_coords(grid)
    da = _sort_coords(da)
    return da


def load_bathymetry(bathy_file=None, lon_min=None, lon_max=None,
                    lat_min=None, lat_max=None, source="file",
                    variable=None, pygmt_resolution="01m"):
    """
    Load bathymetry for a region.

    Parameters
    ----------
    bathy_file : str, optional
        Local NetCDF (GEBCO, ETOPO, ...). If None, falls back to pygmt.
    lon_min, lon_max, lat_min, lat_max : float
        Region of interest (degrees; -180/180 or 0/360).
    source : {'file', 'pygmt'}
    variable : str, optional
        Elevation variable name (auto-detected if None).
    pygmt_resolution : str
        Resolution used when downloading via pygmt ('01m', '30s', '15s').

    Returns
    -------
    xarray.DataArray with dims ('lat', 'lon'), elevation in metres
    (negative = ocean).
    """
    if source == "pygmt" or bathy_file is None:
        return load_bathy_pygmt(lon_min, lon_max, lat_min, lat_max,
                                pygmt_resolution)
    if source != "file":
        raise ValueError(f"Unknown bathymetry source '{source}' (use 'file' or 'pygmt')")
    return load_bathy_from_file(bathy_file, variable,
                                lon_min, lon_max, lat_min, lat_max)


def interpolate_to_regular(bathy_da, lon_ww3, lat_ww3):
    """
    Bi-linear interpolation of source bathymetry onto a WW3 REGULAR grid.

    lon_ww3, lat_ww3 : 1-D arrays (rectilinear case).
    Points outside the source domain are filled with 999999 (land sentinel).
    Returns a 2-D array of shape (ny, nx) in metres.
    """
    src_lat = bathy_da['lat'].values.astype(float)
    src_lon = bathy_da['lon'].values.astype(float)
    src_values = bathy_da.values.astype(float)

    # Handle source lon convention vs request lon convention
    src_lon_0360 = src_lon[0] >= 0
    ww3_lon_neg  = lon_ww3[0] < 0

    interp_lon_ww3 = lon_ww3.copy()
    if src_lon_0360 and ww3_lon_neg:
        interp_lon_ww3 = lon_ww3 + 360.0
    elif not src_lon_0360 and not ww3_lon_neg and lon_ww3[0] > 180:
        interp_lon_ww3 = lon_ww3 - 360.0

    interp_lon_ww3 = np.clip(interp_lon_ww3, src_lon.min(), src_lon.max())
    interp_lat_ww3 = np.clip(lat_ww3,        src_lat.min(), src_lat.max())

    interpolator = RegularGridInterpolator(
        (src_lat, src_lon), src_values,
        method='linear', bounds_error=False, fill_value=999999.0
    )

    lon_grid, lat_grid = np.meshgrid(interp_lon_ww3, interp_lat_ww3)
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    elev = interpolator(pts).reshape(len(lat_ww3), len(lon_ww3))
    return elev


def interpolate_to_curvilinear(bathy_da, lon2d, lat2d):
    """
    Interpolate source bathymetry onto a WW3 CURVILINEAR grid.

    lon2d, lat2d : 2-D arrays of shape (ny, nx) — target grid node positions.
    Uses nearest-neighbour via scipy.griddata for robustness with irregular
    target geometries (polar stereographic, rotated, etc.).
    Points outside the source domain are filled with 999999 (land sentinel).
    Returns a 2-D array of shape (ny, nx) in metres.
    """
    src_lat = bathy_da['lat'].values.astype(float)
    src_lon = bathy_da['lon'].values.astype(float)
    src_values = bathy_da.values.astype(float)

    # Build flat source point cloud
    src_lon2d, src_lat2d = np.meshgrid(src_lon, src_lat)
    src_pts = np.column_stack([src_lon2d.ravel(), src_lat2d.ravel()])
    src_vals = src_values.ravel()

    # Harmonise lon conventions
    query_lon = lon2d.copy()
    if src_lon.min() >= 0 and query_lon.min() < 0:
        query_lon = query_lon + 360.0
    elif src_lon.min() < 0 and query_lon.max() > 180:
        query_lon = query_lon - 360.0

    dst_pts = np.column_stack([query_lon.ravel(), lat2d.ravel()])

    print("  Using linear griddata interpolation for curvilinear target …",
          end=' ', flush=True)
    elev = griddata(src_pts, src_vals, dst_pts, method='linear',
                    fill_value=999999.0)
    print("done")
    return elev.reshape(lon2d.shape)


# Backwards-compatible names (myWW3tools)
interpolate_to_ww3_grid = interpolate_to_regular
interpolate_to_curv_grid = interpolate_to_curvilinear
