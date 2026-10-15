"""
WW3 boundary conditions: list of boundary points for a nested run.

Reads ``{area}.mask2`` (points with value 2) and ``{area}.meta`` and writes
``{area}.list`` in WW3 point format, one line per point::

     -45.950   -25.950 'bnd_00001'

This list is used to extract spectra from the parent grid (ww3_outp / ww3_bounc).

Python
------
>>> from geosushi.ww3 import boundary_points
>>> boundary_points("rio_grid/rio.mask2", label="bnd")

Terminal
--------
    geosushi ww3 bc rio_grid/rio.mask2 [--id bnd]
"""

import os

import numpy as np

from .meta import parse_meta


def get_boundary_points(mask2_file, meta_file=None):
    """Return (lons, lats) of every mask point equal to 2 (bottom-to-top, IDLA=1)."""
    if meta_file is None:
        meta_file = os.path.splitext(mask2_file)[0] + ".meta"
    geo = parse_meta(meta_file)
    nx, ny = geo["nx"], geo["ny"]
    lon = geo["lon1"] + np.arange(nx) * geo["dx"]
    lat = geo["lat1"] + np.arange(ny) * geo["dy"]

    mask = np.loadtxt(mask2_file, dtype=int)
    if mask.shape != (ny, nx):
        raise ValueError(f"mask shape {mask.shape} does not match meta NX={nx}, NY={ny}")
    rows, cols = np.where(mask == 2)
    return lon[cols], lat[rows], (lon, lat)


def write_point_list(path, lons, lats, label="bnd"):
    with open(path, "w") as f:
        for k, (x, y) in enumerate(zip(lons, lats), start=1):
            f.write(f"  {x:10.3f}{y:10.3f} '{label}_{k:05d}'\n")
    return path


def plot_boundary_points(lons, lats, lon_axis, lat_axis, title="", output_file=None,
                         pad=1.0):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_axis[0] - pad, lon_axis[-1] + pad,
                   lat_axis[0] - pad, lat_axis[-1] + pad], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="bisque")
    ax.add_feature(cfeature.OCEAN, facecolor="lightcyan")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="k")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--")
    ax.plot(lons, lats, "r.", markersize=3, transform=ccrs.PlateCarree(),
            label=f"Boundary points ({len(lons)})")
    x0, x1, y0, y1 = lon_axis[0], lon_axis[-1], lat_axis[0], lat_axis[-1]
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "b-", linewidth=1.2,
            transform=ccrs.PlateCarree(), label="Grid domain")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.legend(loc="upper right")
    ax.set_title(title)
    plt.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=150)
        plt.close(fig)
    return fig


def boundary_points(mask2_file, label="bnd", meta_file=None, output_file=None,
                    plot=True):
    """
    Write ``{area}.list`` with the boundary points of a .mask2 file
    (and ``{area}_bc_points.png`` if plot=True). Returns the list path.
    """
    lons, lats, (lon_axis, lat_axis) = get_boundary_points(mask2_file, meta_file)
    base_dir = os.path.dirname(mask2_file)
    area = os.path.basename(mask2_file).replace(".mask2", "")
    if len(lons) == 0:
        print("   no boundary points (value 2) found - run `geosushi ww3 mask` first")
        return None
    output_file = output_file or os.path.join(base_dir, f"{area}.list")
    write_point_list(output_file, lons, lats, label)
    print(f"   {len(lons)} boundary points -> {output_file}")
    if plot:
        fig_path = os.path.join(base_dir, f"{area}_bc_points.png")
        try:
            plot_boundary_points(lons, lats, lon_axis, lat_axis,
                                 f"{area} - WW3 boundary points", fig_path)
            print(f"   figure: {fig_path}")
        except Exception as e:  # e.g. cartopy cannot download coastlines offline
            print(f"   WARNING: figure not created ({e.__class__.__name__}: {e})")
    return output_file
