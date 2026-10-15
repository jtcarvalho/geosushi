"""
WAVEWATCH III tools.

Typical pre-processing workflow (each step also exists as a terminal command)::

    make_grid_from_config("grid.yaml")              # geosushi ww3 grid grid.yaml
    add_boundary("out/rio.mask")                    # geosushi ww3 mask out/rio.mask
    compute_obstructions("out/rio.meta")            # geosushi ww3 obstr out/rio.meta
    boundary_points("out/rio.mask2")                # geosushi ww3 bc out/rio.mask2

Post-processing: geosushi.ww3.io (read outputs), geosushi.ww3.validation.
"""

from .grid import make_grid, make_grid_from_config, build_curv_from_file, build_curv_stereo
from .meta import parse_meta
from .mask import add_boundary, mark_boundary
from .obstruction import compute_obstructions
from .boundary import boundary_points, get_boundary_points
