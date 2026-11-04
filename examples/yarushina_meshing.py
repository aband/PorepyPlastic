"""Yarushina plate-with-hole meshes built on utils.meshing, for comparison with yarushina.py.

Run from the repository root:

    python -m examples.yarushina_meshing                 # levels 0, 1, 2: diagnostics + PNG
    python -m examples.yarushina_meshing --level 1 --compare

build_yarushina_mesh and assign_boundary mirror examples/yarushina.py, but use
utils.meshing.plate_with_hole and its named boundary groups ("outer", "hole",
and the four sides) instead of coordinate tests. --compare checks that the grid,
the boundary faces, the quality numbers and the boundary forces are identical
to the original, when examples/yarushina.py can be imported.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import porepy as pp

from core.layout import FloatArray
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import HardeningParameters, isotropic_hardening_K, kinematic_hardening_H
from utils import meshing
from utils.meshing import Mesh
from utils.visualization import export_mesh_png

OUTPUT = Path("results/yarushina_meshing")


def build_yarushina_mesh(*, radius: float = 1.0, half_width: float = 4.0, level: int = 0) -> Mesh:
    """Triangulate [-half_width, half_width]^2 minus a disk of the given radius."""
    return meshing.plate_with_hole(radius=radius, half_width=half_width, level=level)


def assign_boundary(
    mesh: Mesh,
    *,
    far_field_pressure: float,
    far_field_shear: float = 0.0,
    hole_pressure: float = 0.0,
) -> tuple[pp.BoundaryConditionVectorial, FloatArray]:
    """All-Neumann BCs and outward integrated tractions, shape (2, num_faces).

    Same output as yarushina.assign_boundary. Note: with tractions on the whole
    boundary nothing restrains rigid-body motion, so the Jacobian is singular;
    a solvable case needs at least one Dirichlet group.
    """
    if not np.all(np.isfinite([far_field_pressure, far_field_shear, hole_pressure])):
        raise ValueError("Boundary stresses must be finite.")
    grid = mesh.grid
    bc = pp.BoundaryConditionVectorial(grid, mesh.boundary, "neu")
    sigma_inf = np.diag([
        -far_field_pressure - far_field_shear,
        -far_field_pressure + far_field_shear,
    ])
    forces = np.zeros((2, grid.num_faces), dtype=np.float64)
    for group, stress in (("outer", sigma_inf), ("hole", -hole_pressure * np.eye(2))):
        faces = mesh.groups[group]
        # Outward unit normal times face length = PorePy's area-weighted outward normal.
        area_normals = mesh.outward_normals(faces) * grid.face_areas[faces]
        forces[:, faces] = stress @ area_normals
    return bc, forces


def make_material_points(
    num_cells: int,
    gamma: float = 0.3,      # Poisson ratio
    E: float = 210.0e9,      # Young's modulus, Pa
) -> list[MaterialPoint]:
    material = _materialProperty()
    material.young_modulus = E
    material.poisson_ratio = gamma
    parameters = HardeningParameters(
        sigma_y=250.0e6, sigma_u=250.0e6, H_bar=1.0e9, theta=0.4, delta=0.0,
    )
    return [
        MaterialPoint(
            material=material, parameters=parameters, model=vonMisesModel("J2"),
            K_law=isotropic_hardening_K, H_law=kinematic_hardening_H,
        )
        for _ in range(num_cells)
    ]


def run_mesh(level: int) -> Path:
    mesh = build_yarushina_mesh(level=level)
    print(f"Level {level}")
    print(f"Cells: {mesh.grid.num_cells}")
    print(f"Capped cells: {mesh.capped_cells}")
    print(f"Maximum nonorthogonality: {mesh.max_nonorthogonality:.3e}")
    print(f"TPSA-admissible: {mesh.tpsa_orthogonal}")
    print(f"Boundary groups: { {name: faces.size for name, faces in mesh.groups.items()} }")
    path: Path = export_mesh_png(
        mesh.grid,
        folder_name=str(OUTPUT / "meshes"),
        file_name=f"yarushina_triangular_level{level}",
        title=f"Yarushina mesh at level {level}",
    )
    print(path)
    return path


# ------------------------------------------------------------------- comparison
def compare(level: int, *, original: object | None = None) -> dict[str, bool] | None:
    """Compare with examples/yarushina.py; None if the original cannot be imported.

    `original` may be any module providing build_yarushina_mesh and
    assign_boundary with the original signatures (used by tests).
    """
    if original is None:
        try:
            original = importlib.import_module("examples.yarushina")
        except (ImportError, SyntaxError) as error:
            print(f"examples/yarushina.py cannot be imported ({type(error).__name__}: {error}); comparison skipped.")
            return None

    old = original.build_yarushina_mesh(level=level)  # type: ignore[attr-defined]
    new = build_yarushina_mesh(level=level)
    grid_old, grid_new = old.grid, new.grid
    checks = {
        name: bool(np.array_equal(getattr(grid_old, name), getattr(grid_new, name)))
        for name in ("nodes", "cell_centers", "face_centers", "face_normals", "cell_volumes")
    }
    checks["cell_faces"] = abs(grid_old.cell_faces - grid_new.cell_faces).sum() == 0
    checks["hole_faces"] = bool(np.array_equal(np.sort(old.hole_faces), new.groups["hole"]))
    checks["outer_faces"] = bool(np.array_equal(np.sort(old.outer_faces), new.groups["outer"]))
    checks["max_nonorthogonality"] = bool(np.isclose(old.max_nonorthogonality, new.max_nonorthogonality, rtol=1e-12))
    checks["capped_cells"] = old.capped_cells == new.capped_cells

    loads = {"far_field_pressure": 10e6, "far_field_shear": 3e6, "hole_pressure": 2e6}
    _, forces_old = original.assign_boundary(old, **loads)  # type: ignore[attr-defined]
    _, forces_new = assign_boundary(new, **loads)
    checks["boundary_forces"] = bool(np.allclose(forces_old, forces_new, rtol=1e-14, atol=0.0))

    failed = [name for name, ok in checks.items() if not ok]
    print(f"Level {level}: identical to yarushina.py: {not failed}" + (f" (differs: {failed})" if failed else ""))
    return checks


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, action="append", help="refinement level(s); default 0 1 2")
    parser.add_argument("--compare", action="store_true", help="compare with examples/yarushina.py")
    args = parser.parse_args(argv)
    for level in args.level or [0, 1, 2]:
        if args.compare:
            compare(level)
        else:
            run_mesh(level)


if __name__ == "__main__":
    main()
