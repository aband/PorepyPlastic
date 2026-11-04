"""Self convergence study: python -m examples.core_self_convergence.

Checks self convergence example on a full dirichlet problem.
Checks on different meshes.
"""

from __future__ import annotations

# Standard library: command-line argument
import argparse
from collections.abc import Sequence
from pathlib import Path

# Third-party packages
import numpy as np
import porepy as pp
from porepy.grids.grid import Grid
from dataclasses import field,dataclass

# Core solver
from core.layout import FloatArray
from core.loading import LoadStepController, LoadStepResult
from core.residual import ResidualAssembler
from core.state import SolutionState

# Material definition
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import(
    HardeningParameters,
    isotropic_hardening_K,
    kinematic_hardening_H,
)

# Adaptive stepping and Newton diagnostics
from utils.load_schedule import AdaptiveLoadOptions
from utils.newton_utils import BacktrackingOptions, print_newton_convergence
from utils.geometry import preprocess_simplex_grid

# Visualization module
from utils.visualization import (
    export_png,
    export_strain_png,
    export_vtk,
    export_mesh_png,
)

import matplotlib.pyplot as plt

def dirichlet_boundary(
        L : float,
        H : float,
        centers: FloatArray,
)-> FloatArray:
    """Map face centers to prescribed displacement.
    unit: m

    Dirichlet boundary: u_x = 0.004m * (x/L) * [1+0.2 sin(pi*y/H)]
                        u_y = 0.002m * (y/H) * [1-0.1 cos(pi*x/L)]

    """

    amplitudex, amplitudey = 0.004, 0.002 # m

    # scale x, y coordinates
    x, y = centers[:2]
    return np.vstack(
            (
                amplitudex * x/L * (1 + 0.2 * np.sin(np.pi * y/H)), 
                amplitudey * y/H * (1 - 0.1 * np.cos(np.pi * x/L))
            )
        )

@dataclass(eq=False)
class DirichletResult:
    """One completed core Dirichlet run at a given resolution."""
    assembler: ResidualAssembler
    controller: LoadStepController
    mesh_size: float
    outputs: dict[str, Path] = field(default_factory=dict)

    @property
    def grid(self) -> Grid:
        return self.assembler.operators.grid

    @property
    def state(self) -> SolutionState:
        """Independent snapshot of the final accepted state."""
        return self.controller.state

def build_simplex(
    nx: int,
    ny: int,
    L: float,
    H: float,
) -> Grid:
    """Create triangles from rectangular block, then verify alignment.
    """
    # Local coordinates on a 20 by 20 block
    vertices = np.array([
        [0, 0], [20, 0], [20, 20], [0, 20],      # Corners
        [10, 0], [20, 10], [10, 20], [0, 10],   # Edge midpoints
        [6, 6], [13, 6], [14, 14], [7, 14],     # Interior vertices
    ], dtype=np.int64).T

    triangles = []
    for corner in range(4):
        triangles.extend([
            [corner, 4 + corner, 8 + corner],
            [corner, 8 + corner, 4 + (corner - 1) % 4],
            [4 + corner, 8 + (corner + 1) % 4, 8 + corner],
        ])
    triangles.extend([[8, 9, 11], [9, 10, 11]])
    triangles = np.array(triangles, dtype=np.int64).T

    # Tile the pattern; merge shared nodes before physical scaling.
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    offsets = 20 * np.vstack((ix.ravel(), iy.ravel()))

    nodes = (vertices[:, :, None] + offsets[:, None, :]).reshape(
        (2, -1), order="F",
    )
    nodes, node_map = np.unique(nodes, axis=1, return_inverse=True)

    cells = (
        triangles[:, :, None] + 12 * np.arange(nx * ny)[None, None, :]
    ).reshape((3, -1), order="F")

    points = nodes * np.array([L / (20 * nx), H / (20 * ny)])[:, None]
    grid = pp.TriangleGrid(points, node_map[cells])

    grid.compute_geometry()
    preprocess_simplex_grid(grid)  # Existing default threshold: 0.95.

    # Check every side using its outward unit normal.
    incidence = grid.cell_faces.tocoo()
    normals = (
        grid.face_normals[:2, incidence.row]
        * incidence.data[None, :]
        / grid.face_areas[incidence.row][None, :]
    )
    offsets = (
        grid.face_centers[:2, incidence.row]
        - grid.cell_centers[:2, incidence.col]
    )
    distances = np.sum(offsets * normals, axis=0)
    relative_skew = np.linalg.norm(
        offsets - normals * distances[None, :], axis=0,
    ) / np.linalg.norm(offsets, axis=0)

    if (
        not np.all(np.isfinite(distances))
        or np.any(distances <= 0)
        or not np.all(np.isfinite(relative_skew))
        or np.max(relative_skew) > 1e-10
    ):
        raise ValueError(
            "Simplex mesh fails the TPSA geometry check. "
            "Choose resolution so L/nx is close to H/ny."
        )

    return grid


def build_mesh(
    mesh_kind: str,
    resolution: tuple[int, int],
    L: float,
    H: float,
    *,
    perturbation: float = 0.15,
    seed: int = 0,
) -> Grid:
    nx, ny = resolution
    if any(
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or n < 1
        for n in resolution
    ): 
        raise ValueError("Resolution must contain positive integers.")

    if not np.all(np.isfinite([L,H])) or min(L, H) <= 0:
        raise ValueError("Domain dimensions must be finite and positive.")

    if mesh_kind == "cartesian":
        grid = pp.CartGrid(
            np.array([nx, ny]),
            physdims= np.array([L,H])
        )
    elif mesh_kind == "graded":
        # Fixed smooth mapping
        def coordinates(n: int, length: float) -> FloatArray:
            s = np.linspace(0.0,1.0,n+1)
            beta = 0.6
            mapped = s - beta * np.sin(2.0 * np.pi * s) / (2.0 * np.pi)
            mapped[0], mapped[-1] = 0.0, 1.0
            return length * mapped
        grid = pp.TensorGrid(
            coordinates(nx, L),
            coordinates(ny, H),
        )
    elif mesh_kind == "perturbed":
        # Conservative bound to preserve convex quadrilateral cells.
        if not np.isfinite(perturbation) or not 0 <= perturbation < 0.25:
            raise ValueError("perturbation must lie in [0, 0.25).")

        grid = pp.CartGrid(
            np.array([nx, ny]),
            physdims=np.array([L, H]),
        )

        interior = ~grid.tags["domain_boundary_nodes"]
        spacing = np.array([L / nx, H / ny])[:, None]

        rng = np.random.default_rng(seed)
        shifts = rng.uniform(
            -1.0,
            1.0,
            size=(2, np.count_nonzero(interior)),
        )

        grid.nodes[:2, interior] += perturbation * spacing * shifts

    elif mesh_kind == "simplex":
        return build_simplex(nx, ny, L, H)
    else:
        raise ValueError(f"Unsupported mesh kind: {mesh_kind!r}")

    grid.compute_geometry()
    return grid

def make_material_points(num_cells: int) -> list[MaterialPoint]:
    material = _materialProperty()
    material.young_modulus = 210.0e9       # Pa
    material.poisson_ratio = 0.3

    parameters = HardeningParameters(
        sigma_y=250.0e6,                  # Pa
        sigma_u=250.0e6,                  # No saturation contribution
        H_bar=1.0e9,                      # Pa
        theta=0.4,                       # Mixed isotropic/kinematic hardening
        delta=0.0,
    )

    return [
        MaterialPoint(
            material=material,
            parameters=parameters,
            model=vonMisesModel("J2"),
            K_law=isotropic_hardening_K,
            H_law=kinematic_hardening_H,
        )
        for _ in range(num_cells)
    ]

def run(
    *,
    mesh_kind: str = "cartesian",
    resolution: tuple[int, int] = (8,8),
    L: float = 1.0,
    H: float = 1.0,
    load_factors: Sequence[float] = (0.25,0.5,0.75,1.0),
    output_dir: Path=Path("results/core_self_convergence/run_results"),
    verbose: bool = False,
)-> DirichletResult:
    """Solve the selected plasticity formulation and optionally export accepted steps.
    Uses the default material (E=210Gpa, nu=0.3, initial yield stress 250 Mpa and linear mixed hardening.) 
    """

    # Create the mesh/ material and evaluate the reference boundary condition
    grid = build_mesh(mesh_kind=mesh_kind, resolution=resolution, L=L, H=H)
    material_points = make_material_points(grid.num_cells)
    mesh_size = float(np.max(grid.cell_diameters()))
    
    path = export_mesh_png(
        grid,
        folder_name="results/core_self_convergence/meshes",
        file_name=f"core_{mesh_kind}_{grid.num_cells}cells",
        title=f"{mesh_kind}: {grid.num_cells} cells",
        info=None,  # Use "CF" for numbered cell and face centers.
    )

    # Prescribe both displacement components on every exterior face
    boundary_faces = grid.get_all_boundary_faces()
    bc = pp.BoundaryConditionVectorial(grid, boundary_faces, "dir")

    # Build the layout and cache the geometry and elastic operator
    assembler = ResidualAssembler(
        grid=grid,
        bc=bc,
        material_points=material_points,
    )
    layout = assembler.layout

    # Total prescribed displacement at load factor 1
    reference_dirichlet = dirichlet_boundary(
        L=L,
        H=H,
        centers = grid.face_centers[:, layout.dirichlet_faces],
    )

    # Initially undeformed, unstressed and unloaded
    initial_state = SolutionState.zeros(
        layout,
        num_sides = assembler.operators.num_sides,
        num_neumann_faces=assembler.neumann_faces.size,
    )

    controller = LoadStepController(
        assembler=assembler,
        committed=initial_state,
        reference_dirichlet=reference_dirichlet,
        initial_load_factor=0.0,
        max_iterations=20,
        line_search=BacktrackingOptions(),
        linear_solver=None,
    )

    for factor in load_factors:
        record = controller.advance(factor)

        if verbose:
            print(
                f"Load factor {record.load_factor:g}: "
                f"{record.newton.iterations} Newton iterations"
            )
            print_newton_convergence(record.newton, extra_block="f")

    return DirichletResult(
        assembler=assembler,
        controller=controller,
        mesh_size=mesh_size,
        outputs={"mesh":path},
    )

def run_mesh(
    *,
    mesh_kind: str = "simplex",
    resolution: tuple[int, int] = (8, 8),
    L: float = 1.0,
    H: float = 1.0,
    output_dir: Path = Path("results/core_self_convergence/meshes"),
    info: str | None = None,
) -> Path:
    """Build and export one mesh without creating materials or solving."""
    grid = build_mesh(mesh_kind, resolution, L, H)
    nx, ny = resolution

    return export_mesh_png(
        grid,
        folder_name=output_dir,
        file_name=f"core_{mesh_kind}_{nx}x{ny}_{grid.num_cells}cells",
        title=f"{mesh_kind}: {grid.num_cells} cells",
        info=info,
    )

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the 2D convergence example.")
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument(
        "--mesh",
        choices=("cartesian", "graded", "perturbed" , "simplex"),
        default="cartesian",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show_mesh", type=bool, default=False)

    args = parser.parse_args(argv)

    if args.show_mesh:
        for kind in ("cartesian", "graded", "perturbed", "simplex"):
            print(run_mesh(mesh_kind=kind))
    else:
        result = run(
            mesh_kind=args.mesh,
            resolution=(args.nx, args.ny),
            verbose=args.verbose,
        )

if __name__ == "__main__":
    main()