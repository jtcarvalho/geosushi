"""Tests for the WW3 pre-processing chain (synthetic bathymetry, no downloads)."""

import numpy as np
import pytest
import xarray as xr

from geosushi.cli import main as cli
from geosushi.ww3 import (add_boundary, boundary_points, compute_obstructions,
                          make_grid, mark_boundary, parse_meta)
from geosushi.ww3.grid import build_curv_stereo


@pytest.fixture
def bathy(tmp_path):
    """1x1 degree box: land west of -45.5, ocean (-100 m) elsewhere, one island."""
    lon = np.arange(-47, -43.99, 0.01)
    lat = np.arange(-25, -21.99, 0.01)
    elev = np.full((lat.size, lon.size), -100.0)
    elev[:, lon < -45.5] = 50.0
    island = (np.abs(lon[None, :] + 44.8) < 0.06) & (np.abs(lat[:, None] + 23.5) < 0.06)
    elev[island] = 10.0
    path = tmp_path / "bathy.nc"
    xr.Dataset({"elevation": (("lat", "lon"), elev)},
               coords={"lat": lat, "lon": lon}).to_netcdf(path)
    return path


def _grid(bathy, tmp_path):
    return make_grid(-46, -44, -24, -23, 0.1, 0.1, bathy_file=str(bathy),
                     output_dir=str(tmp_path / "out"), prefix="t", verbose=False)


def test_make_grid(bathy, tmp_path):
    f = _grid(bathy, tmp_path)
    assert (f["nx"], f["ny"]) == (21, 11)
    mask = np.loadtxt(f["mask"], dtype=int)
    assert mask.shape == (11, 21)
    assert mask[:, 0].sum() == 0          # land side
    assert mask[:, -1].all()              # ocean side
    meta = parse_meta(f["meta"])
    assert meta["nx"] == 21 and meta["ny"] == 11
    assert meta["dx"] == pytest.approx(0.1) and meta["lon1"] == pytest.approx(-46)


def test_mark_boundary_skips_land():
    m = np.ones((5, 6), dtype=int)
    m[:, :2] = 0
    out = mark_boundary(m)
    assert (out[1, 2:5] == 2).all() and (out[3, 2:5] == 2).all()
    assert (out[1:4, 4] == 2).all()
    assert (out[:, 1] == 0).all()          # land never becomes boundary
    assert (mark_boundary(m, sides=["east"])[1:4, 4] == 2).all()
    assert (mark_boundary(m, sides=["east"])[1, 2] == 1)


def test_full_chain(bathy, tmp_path):
    f = _grid(bathy, tmp_path)
    mask2 = add_boundary(f["mask"], plot=False)
    obs = compute_obstructions(f["meta"], plot=False)
    lst = boundary_points(mask2, plot=False)
    o = np.loadtxt(obs, dtype=int)
    assert o.shape == (22, 21) and o.max() <= 100 and o.min() >= 0
    lines = open(lst).read().splitlines()
    assert len(lines) == int((np.loadtxt(mask2) == 2).sum()) > 0
    assert lines[0].strip().endswith("'bnd_00001'")


def test_cli_prep(bathy, tmp_path):
    cfg = tmp_path / "grid.yaml"
    cfg.write_text(f"""
grid: {{lon_min: -46, lon_max: -44, lat_min: -24, lat_max: -23, dx: 0.1, dy: 0.1}}
bathymetry: {{source: file, file: {bathy.name}}}
output: {{directory: out, prefix: cli}}
""")
    cli(["ww3", "prep", str(cfg), "--no-plot"])
    for ext in ("dep", "mask", "mask2", "obs", "meta", "list"):
        assert (tmp_path / "out" / f"cli.{ext}").exists()


def test_stereo_centre_is_pole():
    lon, lat = build_curv_stereo(0, -90, 5, 5, 1000)
    assert lat[2, 2] == pytest.approx(-90)
    assert lat.max() < -60
    np.testing.assert_allclose(lat, lat[::-1, ::-1])   # symmetric around the pole


def test_plot_directional_spectrum():
    import matplotlib
    matplotlib.use("Agg")
    from geosushi.common.spectra import plot_directional_spectrum

    freq = np.geomspace(0.04, 0.5, 30)
    dirs = np.arange(0, 360, 15.0)
    E = np.exp(-((freq[:, None] - 0.1) / 0.02) ** 2) * np.cos(np.radians(dirs - 135) / 2) ** 8
    fig, ax = plot_directional_spectrum(E, freq, dirs, hs=2.0, tp=10.0, dp=135.0)
    assert fig is not None and ax is not None
