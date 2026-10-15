"""
WW3 Sub-grid Obstruction Calculator

Computes realistic sub-grid obstruction grids (sx, sy) for WW3 from
a land/sea mask (and optionally a bathymetry file for fractional blocking).

Background
----------
WW3 uses two 2D arrays sx and sy (values 0–100 %) to represent the fraction
of each cell face that is blocked by land.  A value of 0 means fully open,
100 means fully closed.  These control the propagation of wave energy across
narrow passages (straits, islands, etc.).

The MATLAB gridgen computes these by intersecting coastal polygon segments
with cell-face lines.  This script uses a simpler but physically consistent
approach:

  Method A — mask-based (default, fast)
  ──────────────────────────────────────
  For each pair of adjacent ocean cells, the shared face obstruction is
  estimated from the fraction of the *surrounding* cells that are land.
  Specifically, for the E-W face between cell (j,k) and (j+1,k):

      sx(k,j) = fraction of the 4 cells in the 2×2 block centred on
                the shared face that are classified as land.

  This captures partial blockage at the sub-grid scale and matches the
  "shadow zone" concept in the MATLAB algorithm.  If both neighbouring
  cells are ocean and all surrounding cells are ocean, sx=0.
  If one of the two sharing cells is land, sx=1 (fully blocked — this
  face would normally not be written, but WW3 handles it correctly).

  Method B — bathymetry-based (requires .dep file, more accurate)
  ───────────────────────────────────────────────────────────────
  Along each shared cell face, the fraction of sub-grid depth values
  that are below the water level (zdep) is used as the openness factor,
  so:
      sx(k,j) = 1 − (fraction of sub-grid points along face that are ocean)

  This requires the high-resolution source bathymetry file and is best
  for grids that were generated at a coarser resolution than the underlying
  bathymetry data.

In both cases:
  - Obstruction is set to 0 next to dry cells (to prevent spurious swell
    attenuation near the coast), matching the MATLAB gridgen behaviour.
  - Output: rows 0..NY-1 = sx (E-W face), rows NY..2NY-1 = sy (N-S face).
  - Values are rounded to integers in 0–100.

Python
------
>>> from geosushi.ww3 import compute_obstructions
>>> compute_obstructions("rio_grid/rio.meta", method="mask")

Terminal
--------
    geosushi ww3 obstr rio_grid/rio.meta [--method bathy]
"""

import os

import numpy as np

from .meta import parse_meta


def read_mask(path, nx, ny):
    data = np.loadtxt(path, dtype=int)
    if data.shape != (ny, nx):
        raise ValueError(
            f"mask shape {data.shape} ≠ ({ny},{nx})")
    return data


def read_dep(path, nx, ny):
    data = np.loadtxt(path, dtype=np.int64)
    if data.shape != (ny, nx):
        raise ValueError(
            f"dep shape {data.shape} ≠ ({ny},{nx})")
    return data


# ============================================================================
# Method A — mask-based obstruction
# ============================================================================

def compute_obstr_from_mask(mask):
    """
    Estimate sx, sy (0.0–1.0) from the binary land/sea mask.

    For each shared E-W face between columns j and j+1 at row k:
      Consider the 2×2 block of cells: (k-1,j),(k,j),(k-1,j+1),(k,j+1)
      sx(k,j) = fraction of those cells that are land (mask==0).

    Similarly for N-S faces (sy).

    After computing the raw fractions, we apply two corrections:
      1. If either sharing cell is already fully land → sx/sy = 1 (irrelevant
         but kept consistent).
      2. If either sharing cell has a dry neighbour on the other side →
         sx/sy = 0  (prevent spurious swell attenuation near coast).
    """
    ny, nx = mask.shape
    ocean = (mask != 0).astype(float)   # 1=ocean, 0=land
    sx = np.zeros((ny, nx), dtype=float)
    sy = np.zeros((ny, nx), dtype=float)

    # -- sx: E-W face between (k, j) and (k, j+1) -------------------------
    # For interior faces j = 0 .. nx-2:
    for j in range(nx - 1):
        jj = j + 1
        # 2×2 block rows: k-1, k  — clamp at 0 and ny-1
        k_lo = np.maximum(np.arange(ny) - 1, 0)
        k_hi = np.minimum(np.arange(ny),     ny - 1)
        # land fractions from the 4 neighbours of the shared face
        land_frac = (
              (1 - ocean[k_lo, j])
            + (1 - ocean[k_hi, j])
            + (1 - ocean[k_lo, jj])
            + (1 - ocean[k_hi, jj])
        ) / 4.0
        sx[:, j] = land_frac

    # Zero-out next to dry cells (no spurious attenuation) — x direction
    dry = (mask == 0)
    sx[:, :-1][dry[:, 1:]] = 0.0   # right neighbour is land
    sx[:, 1:][dry[:, :-1]] = 0.0   # left  neighbour is land
    # Only relevant for ocean cells — land cells get 0 anyway
    sx[mask == 0] = 0.0

    # -- sy: N-S face between (k, j) and (k+1, j) -------------------------
    for k in range(ny - 1):
        kk = k + 1
        j_lo = np.maximum(np.arange(nx) - 1, 0)
        j_hi = np.minimum(np.arange(nx),     nx - 1)
        land_frac = (
              (1 - ocean[k,  j_lo])
            + (1 - ocean[k,  j_hi])
            + (1 - ocean[kk, j_lo])
            + (1 - ocean[kk, j_hi])
        ) / 4.0
        sy[k, :] = land_frac

    # Zero-out next to dry cells — y direction
    sy[:-1, :][dry[1:, :]] = 0.0
    sy[1:,  :][dry[:-1,:]] = 0.0
    sy[mask == 0] = 0.0

    return sx, sy


# ============================================================================
# Method B — bathymetry-based obstruction (higher accuracy)
# ============================================================================

def compute_obstr_from_bathy(dep_mm, mask, zdep_mm=-100):
    """
    Estimate sx, sy using sub-grid depth information.

    dep_mm : (ny, nx) integer array in millimetres (from .dep file).
    zdep_mm: depth threshold in mm (default -100 mm = -0.10 m).

    For each interior E-W face between (k,j) and (k,j+1):
      Take the two depth values and the two depth values at (k,j-1) and
      (k,j+2) as a 4-point cross-section along the face direction.
      The fraction of those that are LAND (depth >= zdep_mm) is sx(k,j).

    This is a simplified approximation; for true sub-grid accuracy a
    high-resolution bathymetry file should be passed instead of .dep values.
    """
    ny, nx = dep_mm.shape
    sx = np.zeros((ny, nx), dtype=float)
    sy = np.zeros((ny, nx), dtype=float)

    # -- sx faces --
    for j in range(nx - 1):
        jj = j + 1
        j0 = max(j - 1, 0)
        j2 = min(jj + 1, nx - 1)
        land_frac = (
              (dep_mm[:, j]  >= zdep_mm).astype(float)
            + (dep_mm[:, jj] >= zdep_mm).astype(float)
            + (dep_mm[:, j0] >= zdep_mm).astype(float)
            + (dep_mm[:, j2] >= zdep_mm).astype(float)
        ) / 4.0
        sx[:, j] = land_frac

    # -- sy faces --
    for k in range(ny - 1):
        kk = k + 1
        k0 = max(k - 1, 0)
        k2 = min(kk + 1, ny - 1)
        land_frac = (
              (dep_mm[k,  :] >= zdep_mm).astype(float)
            + (dep_mm[kk, :] >= zdep_mm).astype(float)
            + (dep_mm[k0, :] >= zdep_mm).astype(float)
            + (dep_mm[k2, :] >= zdep_mm).astype(float)
        ) / 4.0
        sy[k, :] = land_frac

    # Zero-out next to dry cells
    dry = (mask == 0)
    sx[:, :-1][dry[:, 1:]] = 0.0
    sx[:, 1:][dry[:, :-1]] = 0.0
    sx[mask == 0] = 0.0
    sy[:-1, :][dry[1:, :]] = 0.0
    sy[1:,  :][dry[:-1,:]] = 0.0
    sy[mask == 0] = 0.0

    return sx, sy


# ============================================================================
# File writer
# ============================================================================

def write_obs_values(path, sx, sy, nx, ny):
    """
    Write .obs file: rows 0..NY-1 = sx (E-W), rows NY..2NY-1 = sy (N-S).
    Values are clipped to [0, 100] and written as 3-character integers.
    """
    obs = np.vstack([sx, sy])
    obs_int = np.clip(np.round(obs * 100).astype(int), 0, 100)
    with open(path, 'w') as f:
        for j in range(2 * ny):
            f.write(''.join(f'{obs_int[j, i]:3d}' for i in range(nx)) + '\n')
    return obs_int


# ============================================================================
# Main
# ============================================================================


def plot_obstructions(obs_int, ny, title="", output_file=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=11)
    cmap = plt.cm.hot_r.copy()
    cmap.set_bad("white")
    norm = mcolors.Normalize(vmin=1, vmax=100)
    for ax, data, name in ((axes[0], obs_int[:ny], "sx  (E-W face blockage %)"),
                           (axes[1], obs_int[ny:], "sy  (N-S face blockage %)")):
        im = ax.pcolormesh(np.ma.masked_equal(data.astype(float), 0),
                           cmap=cmap, norm=norm)
        ax.set_title(name)
        ax.set_xlabel("i (lon index)")
        ax.set_ylabel("j (lat index)")
        plt.colorbar(im, ax=ax, label="blockage (%)")
    plt.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=150)
        plt.close(fig)
    return fig


def compute_obstructions(meta_file, method="mask", zdep=-0.10,
                         output_dir=None, output_prefix=None, plot=True):
    """
    Compute sub-grid obstructions for a WW3 grid and overwrite ``{prefix}.obs``.

    Reads ``{prefix}.mask`` (and ``{prefix}.dep`` for method='bathy') from the
    same directory as ``meta_file``.

    method : 'mask' (fast, default) or 'bathy' (uses the .dep depths)

    Returns the path of the .obs file.
    """
    geo = parse_meta(meta_file)
    nx, ny = geo["nx"], geo["ny"]
    src_dir = os.path.dirname(os.path.abspath(meta_file))
    prefix = os.path.basename(meta_file)[:-len(".meta")]
    output_dir = output_dir or src_dir
    output_prefix = output_prefix or prefix

    mask = read_mask(os.path.join(src_dir, f"{prefix}.mask"), nx, ny)
    method = method.lower()
    if method == "bathy":
        dep_path = os.path.join(src_dir, f"{prefix}.dep")
        dep_mm = read_dep(dep_path, nx, ny)
        sx, sy = compute_obstr_from_bathy(dep_mm, mask, int(zdep * 1000))
    elif method == "mask":
        sx, sy = compute_obstr_from_mask(mask)
    else:
        raise ValueError("method must be 'mask' or 'bathy'")

    os.makedirs(output_dir, exist_ok=True)
    obs_path = os.path.join(output_dir, f"{output_prefix}.obs")
    obs_int = write_obs_values(obs_path, sx, sy, nx, ny)
    n_blk = int((obs_int > 0).sum())
    print(f"   {method} method: {n_blk} / {obs_int.size} blocked faces -> {obs_path}")
    if plot:
        fig_path = os.path.join(output_dir, f"{output_prefix}_obstr.png")
        plot_obstructions(obs_int, ny,
                          f"Sub-grid obstructions - {output_prefix} ({method} method)",
                          fig_path)
        print(f"   figure: {fig_path}")
    return obs_path
