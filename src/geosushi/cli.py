"""
Terminal entry point:  ``geosushi <model> <task> ...``

    geosushi list                         show every available command
    geosushi template ww3-grid [dest]     copy an example YAML to edit
    geosushi ww3 grid grid.yaml           generate .dep/.mask/.obs/.meta
    geosushi ww3 mask out/rio.mask        mark open-boundary points -> rio.mask2
    geosushi ww3 obstr out/rio.meta       sub-grid obstructions -> rio.obs
    geosushi ww3 bc out/rio.mask2         boundary point list -> rio.list
    geosushi ww3 prep grid.yaml           grid + mask + obstr + bc in one go

New tools: add a function in geosushi/<model>/..., then a sub-command below.
"""

import argparse
import os
import shutil
import sys

from . import __version__

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> (path inside the package, default file name)
TEMPLATES = {
    "ww3-grid": ("ww3/confs/grid.yaml", "grid.yaml"),
    "ww3-plot": ("ww3/confs/plot_regular.yaml", "plot_regular.yaml"),
}


# ---------------------------------------------------------------- handlers ---

def _template(a):
    if a.name not in TEMPLATES:
        sys.exit(f"unknown template '{a.name}'. Options: {', '.join(TEMPLATES)}")
    src, default = TEMPLATES[a.name]
    dest = a.dest or default
    if os.path.isdir(dest):
        dest = os.path.join(dest, default)
    if os.path.exists(dest) and not a.force:
        sys.exit(f"{dest} already exists (use --force to overwrite)")
    shutil.copy(os.path.join(HERE, src), dest)
    print(f"created {dest} - edit it and run the matching command")


def _ww3_grid(a):
    from .ww3.grid import make_grid_from_config
    return make_grid_from_config(a.config)


def _ww3_mask(a):
    from .ww3.mask import add_boundary
    return add_boundary(a.mask_file, sides=a.sides, plot=not a.no_plot)


def _ww3_obstr(a):
    from .ww3.obstruction import compute_obstructions
    return compute_obstructions(a.meta_file, method=a.method, zdep=a.zdep,
                                plot=not a.no_plot)


def _ww3_bc(a):
    from .ww3.boundary import boundary_points
    return boundary_points(a.mask2_file, label=a.label, plot=not a.no_plot)


def _ww3_prep(a):
    from .ww3.grid import make_grid_from_config
    from .ww3.mask import add_boundary
    from .ww3.obstruction import compute_obstructions
    from .ww3.boundary import boundary_points

    print("[1/4] grid")
    files = make_grid_from_config(a.config)
    print("[2/4] boundary mask")
    mask2 = add_boundary(files["mask"], sides=a.sides, plot=not a.no_plot)
    if "lon" in files:
        print("[3/4] obstructions: skipped (curvilinear grid)")
        print("[4/4] boundary list: skipped (curvilinear grid)")
        return
    print("[3/4] obstructions")
    compute_obstructions(files["meta"], method=a.method, plot=not a.no_plot)
    print("[4/4] boundary list")
    boundary_points(mask2, label=a.label, plot=not a.no_plot)


# ------------------------------------------------------------------ parser ---

SIDES = ["south", "north", "west", "east"]


def build_parser():
    p = argparse.ArgumentParser(
        prog="geosushi",
        description="Day-to-day tools for ROMS, WRF, WW3 and SWAN.")
    p.add_argument("--version", action="version", version=f"geosushi {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="<command>")

    s = sub.add_parser("list", help="show every available command")
    s.set_defaults(func=lambda a: print(__doc__))

    s = sub.add_parser("template", help="copy an example YAML config to edit")
    s.add_argument("name", help=f"one of: {', '.join(TEMPLATES)}")
    s.add_argument("dest", nargs="?", help="destination file or folder")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=_template)

    # ---- WW3 ----
    ww3 = sub.add_parser("ww3", help="WAVEWATCH III tools").add_subparsers(
        dest="task", metavar="<task>")

    s = ww3.add_parser("grid", help="generate grid files from a YAML config")
    s.add_argument("config")
    s.set_defaults(func=_ww3_grid)

    s = ww3.add_parser("mask", help="mark open-boundary points (2) -> .mask2")
    s.add_argument("mask_file")
    s.add_argument("--sides", nargs="+", choices=SIDES, default=SIDES,
                   help="open boundaries (default: all four)")
    s.add_argument("--no-plot", action="store_true")
    s.set_defaults(func=_ww3_mask)

    s = ww3.add_parser("obstr", help="sub-grid obstructions -> .obs")
    s.add_argument("meta_file")
    s.add_argument("--method", choices=["mask", "bathy"], default="mask")
    s.add_argument("--zdep", type=float, default=-0.10)
    s.add_argument("--no-plot", action="store_true")
    s.set_defaults(func=_ww3_obstr)

    s = ww3.add_parser("bc", help="boundary point list (.list) from a .mask2")
    s.add_argument("mask2_file")
    s.add_argument("--id", dest="label", default="bnd", help="point label prefix")
    s.add_argument("--no-plot", action="store_true")
    s.set_defaults(func=_ww3_bc)

    s = ww3.add_parser("prep", help="grid + mask + obstr + bc in one go")
    s.add_argument("config")
    s.add_argument("--sides", nargs="+", choices=SIDES, default=SIDES)
    s.add_argument("--method", choices=["mask", "bathy"], default="mask")
    s.add_argument("--id", dest="label", default="bnd")
    s.add_argument("--no-plot", action="store_true")
    s.set_defaults(func=_ww3_prep)

    # ---- placeholders, so `geosushi --help` shows the plan ----
    for name, title in (("swan", "SWAN"), ("roms", "ROMS"), ("wrf", "WRF")):
        s = sub.add_parser(name, help=f"{title} tools (coming soon)")
        s.set_defaults(func=lambda a, t=title: print(
            f"No {t} commands yet - add them in geosushi/{a.cmd}/ and cli.py"))
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        # `geosushi` or `geosushi ww3` without a task
        parser.parse_args((argv or sys.argv[1:]) + ["--help"])
        return
    args.func(args)


if __name__ == "__main__":
    main()
