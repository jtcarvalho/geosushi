"""
WW3 mask utilities: mark open-boundary points (value 2).

WW3 mask convention: 0 = land, 1 = sea, 2 = boundary (input) point.

Python
------
>>> from geosushi.ww3 import add_boundary
>>> add_boundary("rio_grid/rio.mask")        # writes rio.mask2 + rio_mask.png

Terminal
--------
    geosushi ww3 mask rio_grid/rio.mask
"""

import os

import numpy as np


def mark_boundary(mask, sides=("south", "north", "west", "east")):
    """
    Return a copy of ``mask`` with sea points on the open boundary set to 2.

    Same rule as myWW3tools/prepMaskWW3.py: the boundary is the second
    row/column from each edge (the outermost line is always land in WW3);
    land points are skipped.

    sides : which edges are open; any of 'south', 'north', 'west', 'east'.
    """
    m = np.array(mask, dtype=int, copy=True)
    ny, nx = m.shape
    sel = np.zeros_like(m, dtype=bool)
    if "south" in sides:
        sel[1, 1:nx - 1] = True
    if "north" in sides:
        sel[ny - 2, 1:nx - 1] = True
    if "west" in sides:
        sel[1:ny - 1, 1] = True
    if "east" in sides:
        sel[1:ny - 1, nx - 2] = True
    m[sel & (m == 1)] = 2
    return m


def plot_mask(mask, title="", output_file=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    pm = ax.pcolormesh(mask, cmap="RdYlGn")
    plt.colorbar(pm, ax=ax, label="Mask (0=land, 1=sea, 2=boundary)")
    ax.set_title(title)
    ax.set_xlabel("i (lon index)")
    ax.set_ylabel("j (lat index)")
    plt.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=150)
        plt.close(fig)
    return fig


def add_boundary(mask_file, output_file=None, sides=("south", "north", "west", "east"),
                 plot=True):
    """
    Read a .mask file, mark boundary points and write ``<mask_file>2``
    (e.g. rio.mask -> rio.mask2) plus ``<prefix>_mask.png``.

    Returns the path of the new mask file.
    """
    mask = np.loadtxt(mask_file, dtype=int)
    new = mark_boundary(mask, sides=sides)
    output_file = output_file or mask_file + "2"
    np.savetxt(output_file, new, fmt="%d")
    n_bc = int((new == 2).sum())
    print(f"   {n_bc} boundary points -> {output_file}")
    if plot:
        fig_path = os.path.splitext(output_file)[0] + "_mask.png"
        plot_mask(new, os.path.basename(mask_file), fig_path)
        print(f"   figure: {fig_path}")
    return output_file
