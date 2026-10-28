"""Small spatial 3D J2 loading/unloading example, with full Dirichlet data.

Run from the repository root:
    python -B -m prototype.example_3d
    python -B -m prototype.example_3d --jacobian finite-difference --adaptive

The unit cube uses a uniform tensor grid. Every boundary face is prescribed
u = factor * G @ x, with G containing extension and shear in all three planes.
Material: E=210 GPa, nu=0.3, initial yield stress 250 MPa, linear combined
hardening H_bar=1 GPa and theta=0.4. Printed results are accepted load factors,
Newton iterations and volume-averaged accumulated plastic strain. No files are
exported; the existing shared plotting/example modules remain plane-strain APIs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from porepy.grids.grid import Grid
from porepy.grids.structured import CartGrid
from porepy.params.bc import BoundaryConditionVectorial

from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import HardeningParameters, isotropic_hardening_K, kinematic_hardening_H
from prototype.runner import prepare_retained_solver
from prototype.newton import BacktrackingOptions


@dataclass
class CubeProblem:
    grid: Grid
    bc: BoundaryConditionVectorial
    material_points: list[MaterialPoint]
    bc_values: NDArray[np.float64]


def make_problem(cells_per_axis: int = 2) -> CubeProblem:
    """Prepare the cube without assembling or solving a global matrix."""
    if isinstance(cells_per_axis, bool) or not isinstance(cells_per_axis, int) or cells_per_axis < 1:
        raise ValueError("cells_per_axis must be a positive integer.")
    grid = CartGrid(np.full(3, cells_per_axis), physdims=np.ones(3))
    grid.compute_geometry()
    faces = grid.get_all_boundary_faces()
    bc = BoundaryConditionVectorial(grid, faces, "dir")
    material = _materialProperty()
    material.young_modulus = 210e9
    material.poisson_ratio = .3
    parameters = HardeningParameters(250e6, 250e6, 1e9, .4, 0.)
    points = [MaterialPoint(material, parameters, vonMisesModel("J2"),
                           isotropic_hardening_K, kinematic_hardening_H)
              for _ in range(grid.num_cells)]
    gradient = .004 * np.array([[1., .3, -.2], [-.1, -.2, .4], [.15, -.25, .5]])
    boundary = np.zeros((3, grid.num_faces))
    boundary[:, faces] = gradient @ grid.face_centers[:, faces]
    return CubeProblem(grid, bc, points, boundary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=2, help="Cells per axis (default: 2).")
    parser.add_argument("--jacobian", choices=("analytical", "finite-difference"), default="analytical")
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()
    problem = make_problem(args.cells)
    setup = prepare_retained_solver(problem, args.jacobian, line_search=BacktrackingOptions())
    factors = [.25, .5, .75, 1., .75, .5]
    records = (setup.controller.iter_adaptive(factors) if args.adaptive
               else (setup.controller.advance(factor) for factor in factors))
    print(f"Spatial 3D, {problem.grid.num_cells} cells, {args.jacobian} Jacobian")
    for record in records:
        alpha = np.array([state.alpha for state in record.state.tpsa.material_states])
        mean = float(np.average(alpha, weights=problem.grid.cell_volumes))
        print(f"factor={record.load_factor:.4f}  Newton={record.newton.iterations:2d}  mean_alpha={mean:.6e}")


if __name__ == "__main__":
    main()
