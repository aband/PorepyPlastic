"""Acute triangular meshes for two benchmarks: python -m examples.example_acute_mesh.

Builds strictly acute (TPSA-orthogonal) triangulations with
utils.meshing.planar_domain at several refinement levels for

    cook   Cook's membrane: tapered panel, clamped at x = 0, loaded at x = 48 mm.
    strip  Perforated strip, one quarter: 100 x 180 mm minus a radius-50 mm
           quarter hole, refined toward the hole.

Geometry follows Abbas, Ern and Pignet (2019), Sections 4.2 and 4.3, in metres.
Each mesh is checked and exported as a PNG; nothing is solved.

    python -m examples.example_acute_mesh                       # both cases, levels 0 1 2
    python -m examples.example_acute_mesh --case cook --levels 0 1 2 3
    python -m examples.example_acute_mesh --no_png              # diagnostics only
"""

from __future__ import annotations

# Standard library: command-line arguments
import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

# Third-party packages
import numpy as np

# Mesh builders and angle diagnostics
from utils import meshing
from utils.acute import triangle_angles
from utils.meshing import Arc, Line, Mesh

# Visualization module
from utils.visualization import export_mesh_png

CASES = ("cook", "strip")
OUTPUT = Path("results/example_acute_mesh")
MM = 1.0e-3   # millimetre in metres


def cook_boundary() -> list[Line]:
    """Cook's membrane outline, counterclockwise from the clamped bottom corner.

    Groups: "clamped" (x = 0), "load" (x = 48 mm, where the shear load acts),
    "bottom" and "top" (traction free). Point A of the benchmark, (48, 60) mm,
    is a mesh node.
    """
    corners = [(0.0, 0.0), (48.0 * MM, 44.0 * MM), (48.0 * MM, 60.0 * MM), (0.0, 44.0 * MM)]
    names = ("bottom", "load", "top", "clamped")
    return [Line(corners[k], corners[(k + 1) % 4], names[k]) for k in range(4)]


def strip_boundary(R: float = 50.0 * MM, L: float = 100.0 * MM, H: float = 180.0 * MM) -> list[Line | Arc]:
    """Quarter perforated strip, counterclockwise, hole centred at the origin.

    Groups: "bottom" (y = 0 symmetry), "left" (x = 0 symmetry), "top"
    (prescribed extension), "right" (traction free) and "hole". Points A = (0, R)
    and B = (R, 0) of the benchmark are mesh nodes.
    """
    return [
        Line((R, 0.0), (L, 0.0), "bottom"),
        Line((L, 0.0), (L, H), "right"),
        Line((L, H), (0.0, H), "top"),
        Line((0.0, H), (0.0, R), "left"),
        Arc((0.0, R), (R, 0.0), (0.0, 0.0), "hole"),
    ]


def strip_size(level: int, R: float = 50.0 * MM) -> Callable[[float, float], float]:
    """Cell size h(x, y): 5 mm / 2**level at the hole, growing linearly to four
    times that at distance 2 R from the hole."""
    near = 5.0 * MM / 2**level

    def size(x: float, y: float) -> float:
        distance = float(np.hypot(x, y)) - R
        return float(near * (1.0 + 3.0 * min(1.0, max(0.0, distance / (2.0 * R)))))

    return size


def build_mesh(case: str, level: int) -> Mesh:
    """Acute mesh of one case; each level halves the cell size."""
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("level must be a nonnegative integer.")
    if case == "cook":
        return meshing.planar_domain(
            cook_boundary(), size=4.0 * MM / 2**level, description=f"cook level {level}",
        )
    if case == "strip":
        return meshing.planar_domain(
            strip_boundary(), size=strip_size(level), description=f"strip level {level}",
        )
    raise ValueError(f"Unsupported case: {case!r}; choose from {CASES}.")


def angle_range(mesh: Mesh) -> tuple[float, float]:
    """Smallest and largest triangle angle in degrees."""
    grid = mesh.grid
    cells = grid.cell_nodes().tocsc().indices.reshape((3, grid.num_cells), order="F").astype(np.int64)
    angles = np.abs(triangle_angles(grid.nodes[:2], cells))
    return float(angles.min()), float(angles.max())


def run_mesh(case: str, level: int, *, export: bool = True) -> Mesh:
    """Build one mesh, check TPSA admissibility, print diagnostics, export a PNG."""
    mesh = build_mesh(case, level)
    smallest, largest = angle_range(mesh)
    # planar_domain raises if acute meshing fails; check again so the example
    # states its guarantee explicitly.
    if largest >= 90.0 or mesh.capped_cells or not mesh.tpsa_orthogonal:
        raise RuntimeError(f"{case} level {level}: mesh is not TPSA-admissible.")

    groups = ", ".join(f"{name} {faces.size}" for name, faces in mesh.groups.items())
    print(
        f"{case:5s} level {level}: {mesh.grid.num_cells:6d} cells, "
        f"angles [{smallest:5.1f}, {largest:5.1f}] deg, capped cells {mesh.capped_cells}, "
        f"max nonorthogonality {mesh.max_nonorthogonality:.1e}"
    )
    print(f"    boundary faces: {groups}")
    if export:
        path = export_mesh_png(
            mesh.grid,
            folder_name=OUTPUT,
            file_name=f"{case}_level{level}_{mesh.grid.num_cells}cells",
            title=f"{case}, level {level}: {mesh.grid.num_cells} cells",
        )
        print(f"    {path}")
    return mesh


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build acute meshes for the Cook and strip benchmarks.")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--no_png", action="store_true", help="skip the PNG export")
    args = parser.parse_args(argv)

    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        for level in args.levels:
            run_mesh(case, level, export=not args.no_png)


if __name__ == "__main__":
    main()
