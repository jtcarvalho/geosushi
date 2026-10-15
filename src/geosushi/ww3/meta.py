"""
WW3 grid ``.meta`` files (the grid block of ``ww3_grid.inp``): read and write.
"""

import os


def parse_meta(meta_path):
    """
    Read a RECT WW3 ``.meta`` file.

    Returns
    -------
    dict with nx, ny, dx, dy (degrees), lon1, lat1 (degrees of point (1,1)).
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path) as f:
        lines = [l.strip() for l in f
                 if l.strip() and not l.strip().startswith("$")]
    try:
        gtype = lines[0].split()[0].strip("'\"").upper()
        if gtype != "RECT":
            raise ValueError(f"only RECT grids are supported here (found {gtype})")
        nx, ny = map(int, lines[1].split()[:2])
        sx, sy, scale_xy = map(float, lines[2].split()[:3])
        x0, y0, scale_c = map(float, lines[3].split()[:3])
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not parse meta file '{meta_path}': {e}")
    return dict(nx=nx, ny=ny, dx=sx / scale_xy, dy=sy / scale_xy,
                lon1=x0 / scale_c, lat1=y0 / scale_c)


_META_HEADER = """\
$ Define grid -------------------------------------- $
$ Five records containing :
$  1 Type of grid, coordinate system and type of closure: GSTRG, FLAGLL,
$    CSTRG. Grid closure can only be applied in spherical coordinates.
$      GSTRG  : String indicating type of grid :
$               'RECT'  : rectilinear
$               'CURV'  : curvilinear
$      FLAGLL : Flag to indicate coordinate system :
$               T  : Spherical (lon/lat in degrees)
$               F  : Cartesian (meters)
$      CSTRG  : String indicating the type of grid index space closure :
$               'NONE'  : No closure is applied
$               'SMPL'  : Simple grid closure : Grid is periodic in the
$                         : i-index and wraps at i=NX+1. In other words,
$                         : (NX+1,J) => (1,J). A grid with simple closure
$                         : may be rectilinear or curvilinear.
$               'TRPL'  : Tripole grid closure : Grid is periodic in the
$                         : i-index and wraps at i=NX+1 and has closure at
$                         : j=NY+1. In other words, (NX+1,J<=NY) => (1,J)
$                         : and (I,NY+1) => (MOD(NX-I+1,NX)+1,NY). Tripole
$                         : grid closure requires that NX be even. A grid
$                         : with tripole closure must be curvilinear.
$  2 NX, NY. As the outer grid lines are always defined as land
$    points, the minimum size is 3x3.
$  3 Grid increments SX, SY (degr.or m) and scaling (division) factor.
$    If NX*SX = 360., latitudinal closure is applied.
$  4 Coordinates of (1,1) (degr.) and scaling (division) factor.
$  5 Limiting bottom depth (m) to discriminate between land and sea
$    points, minimum water depth (m) as allowed in model, unit number
$    of file with bottom depths, scale factor for bottom depths (mult.),
$    IDLA, IDFM, format for formatted read, FROM and filename.
$      IDLA : Layout indicator :
$                  1   : Read line-by-line bottom to top.
$                  2   : Like 1, single read statement.
$                  3   : Read line-by-line top to bottom.
$                  4   : Like 3, single read statement.
$      IDFM : format indicator :
$                  1   : Free format.
$                  2   : Fixed format with above format descriptor.
$                  3   : Unformatted.
$      FROM : file type parameter
$             'UNIT' : open file by unit number only.
$             'NAME' : open file by name and assign to unit.
$  If the Unit Numbers in above files is 10 then data is read from this file
$
"""


def write_meta(filepath, nx, ny, dx, dy, lon1, lat1, prefix,
               zdep=-0.10, zmin=2.50, closure='NONE'):
    """
    Write the WW3 grid meta file (can be embedded in ww3_grid.inp).

    Grid increments are stored in arc-minutes (SX = dx * 60) with
    SCALE=60, matching the convention used in the provided examples.

    Global grids (lon span ≈ 360°) automatically use 'SMPL' closure.
    """
    sx = dx * 60.0
    sy = dy * 60.0
    scale_xy = 60.0

    # Auto-detect simple closure for global grids
    lon_span = (nx - 1) * dx
    if abs(lon_span - 360.0) < dx:
        closure = 'SMPL'

    with open(filepath, 'w') as f:
        f.write(_META_HEADER)
        f.write(f"   'RECT'  T  '{closure}'\n")
        f.write(f"{nx:7d}  {ny:7d}\n")
        f.write(f"{sx:8.2f}   {sy:8.2f}   {scale_xy:.2f}\n")
        f.write(f"{lon1:.4f}         {lat1:.4f}         1.00\n")
        f.write(f"$ Bottom Bathymetry\n")
        f.write(f"{zdep:.2f}   {zmin:.2f}  40  0.001000  1  1 "
                f"'(....)'  NAME  './{prefix}.dep'\n")
        f.write(f"$ Sub-grid information\n")
        f.write(f"50  0.010000  1  1  '(....)'  NAME  './{prefix}.obs'\n")
        f.write(f"$ Mask Information\n")
        f.write(f"60  1  1  '(....)'  NAME  './{prefix}.mask'\n")


def write_meta_curv(filepath, nx, ny, prefix, zdep=-0.10, zmin=2.50,
                    coord_scale=1.0):
    """
    Write the WW3 grid meta file for a CURVILINEAR grid.

    The .lon and .lat files are read via unit numbers 20 and 30 respectively.
    coord_scale is the multiplier applied to the stored coordinate values
    (1.0 for degrees — WW3 reads them and multiplies by this factor).
    """
    with open(filepath, 'w') as f:
        f.write(_META_HEADER)
        f.write(f"   'CURV'  T  'NONE'\n")
        f.write(f"{nx:7d}  {ny:7d}\n")
        # X coordinate file
        f.write(f"20  {coord_scale:.6f}  0.00  1  1 "
                f"'(....)'  NAME  './{prefix}.lon'\n")
        # Y coordinate file
        f.write(f"30  {coord_scale:.6f}  0.00  1  1 "
                f"'(....)'  NAME  './{prefix}.lat'\n")
        f.write(f"$ Bottom Bathymetry\n")
        f.write(f"{zdep:.2f}   {zmin:.2f}  40  0.001000  1  1 "
                f"'(....)'  NAME  './{prefix}.dep'\n")
        f.write(f"$ Sub-grid information\n")
        f.write(f"50  0.010000  1  1  '(....)'  NAME  './{prefix}.obs'\n")
        f.write(f"$ Mask Information\n")
        f.write(f"60  1  1  '(....)'  NAME  './{prefix}.mask'\n")
