"""Self-convergence example built on utils.meshing, for comparison with core_self_convergence_2D.

Run from the repository root:

    python -m examples.core_self_convergence_meshing --mesh perturbed --nx 8 --ny 8
    python -m examples.core_self_convergence_meshing --compare     # all mesh kinds, both versions
    python -m examples.core_self_convergence_meshing --show_mesh

Physics (boundary displacement, material, load path, controller settings) is
imported from examples.core_self_convergence_2D, so the mesh construction is the
only difference: utils.meshing builders instead of the example's build_mesh, and
named boundary groups instead of "all boundary faces".
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import porepy as pp

from core.loading import LoadStepController
from core.residual import ResidualAssembler
from core.state import SolutionState
from examples.core_self_convergence_2D import (
    DirichletResult,
    build_mesh as original_build_mesh,
    dirichlet_boundary,
    make_material_points,
    run as original_run,
)
from utils import meshing
from utils.meshing import Mesh
from utils.newton_utils import BacktrackingOptions, print_newton_convergence
from utils.visualization import export_mesh_png

MESH_KINDS = ("cartesian", "graded", "perturbed", "simplex")
OUTPUT = Path("results/core_self_convergence_meshing")


def build_mesh(
    mesh_kind: str,
    resolution: tuple[int, int],
    L: float,
    H: float,
    *,
    perturbation: float = 0.15,
    seed: int = 0,
) -> Mesh:
    """Same meshes as core_self_convergence_2D.build_mesh, with boundary groups."""
    if mesh_kind == "simplex":
        return meshing.simplex_rectangle(resolution, (L, H))
    if mesh_kind in ("cartesian", "graded", "perturbed"):
        # The original graded mapping uses beta = 0.6.
        return meshing.structured(
            mesh_kind,  # type: ignore[arg-type]
            resolution, (L, H), grading=0.6, perturbation=perturbation, seed=seed,
        )
    raise ValueError(f"Unsupported mesh kind: {mesh_kind!r}")


def run(
    *,
    mesh_kind: str = "cartesian",
    resolution: tuple[int, int] = (8, 8),
    L: float = 1.0,
    H: float = 1.0,
    load_factors: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    export_mesh: bool = True,
    verbose: bool = False,
) -> DirichletResult:
    """Same problem as core_self_convergence_2D.run; mesh from utils.meshing."""
    mesh = build_mesh(mesh_kind, resolution, L, H)
    grid = mesh.grid
    outputs: dict[str, Path] = {}
    if export_mesh:
        outputs["mesh"] = export_mesh_png(
            grid,
            folder_name=str(OUTPUT / "meshes"),
            file_name=f"core_{mesh_kind}_{grid.num_cells}cells",
            title=f"{mesh_kind}: {grid.num_cells} cells",
            info=None,
        )

    # Every exterior face is Dirichlet: all four box sides, by name.
    dirichlet_faces = mesh.faces("xmin", "xmax", "ymin", "ymax")
    bc = pp.BoundaryConditionVectorial(grid, dirichlet_faces, "dir")
    assembler = ResidualAssembler(grid=grid, bc=bc, material_points=make_material_points(grid.num_cells))
    layout = assembler.layout

    controller = LoadStepController(
        assembler=assembler,
        committed=SolutionState.zeros(
            layout, num_sides=assembler.operators.num_sides,
            num_neumann_faces=assembler.neumann_faces.size,
        ),
        reference_dirichlet=dirichlet_boundary(L=L, H=H, centers=grid.face_centers[:, layout.dirichlet_faces]),
        initial_load_factor=0.0,
        max_iterations=20,
        line_search=BacktrackingOptions(),
        linear_solver=None,
    )
    for factor in load_factors:
        record = controller.advance(factor)
        if verbose:
            print(f"Load factor {record.load_factor:g}: {record.newton.iterations} Newton iterations")
            print_newton_convergence(record.newton, extra_block="f")

    return DirichletResult(
        assembler=assembler,
        controller=controller,
        mesh_size=float(np.max(grid.cell_diameters())),
        outputs=outputs,
    )


def run_mesh(
    *,
    mesh_kind: str = "simplex",
    resolution: tuple[int, int] = (8, 8),
    L: float = 1.0,
    H: float = 1.0,
    info: str | None = None,
) -> Path:
    """Build and export one mesh without solving; prints its quality summary."""
    mesh = build_mesh(mesh_kind, resolution, L, H)
    print(mesh.summary())
    nx, ny = resolution
    path: Path = export_mesh_png(
        mesh.grid,
        folder_name=str(OUTPUT / "meshes"),
        file_name=f"core_{mesh_kind}_{nx}x{ny}_{mesh.grid.num_cells}cells",
        title=f"{mesh_kind}: {mesh.grid.num_cells} cells",
        info=info,
    )
    return path


# ------------------------------------------------------------------- comparison
def compare_meshes(mesh_kind: str, resolution: tuple[int, int], L: float = 1.0, H: float = 1.0) -> dict[str, bool]:
    """Is the utils.meshing grid identical to the original example's grid?"""
    original = original_build_mesh(mesh_kind, resolution, L, H)
    new = build_mesh(mesh_kind, resolution, L, H).grid
    checks = {
        name: bool(np.array_equal(getattr(original, name), getattr(new, name)))
        for name in ("nodes", "cell_centers", "face_centers", "face_normals", "cell_volumes")
    }
    checks["cell_faces"] = abs(original.cell_faces - new.cell_faces).sum() == 0
    checks["boundary"] = bool(np.array_equal(
        np.sort(original.get_all_boundary_faces()), build_mesh(mesh_kind, resolution, L, H).faces(
            "xmin", "xmax", "ymin", "ymax",
        ),
    ))
    return checks


def compare_runs(mesh_kind: str, resolution: tuple[int, int]) -> dict[str, float]:
    """Run both versions; relative max differences of the final states and Newton counts."""
    old = original_run(mesh_kind=mesh_kind, resolution=resolution)
    new = run(mesh_kind=mesh_kind, resolution=resolution, export_mesh=False)
    a, b = old.state, new.state
    result = {
        name: float(np.abs(getattr(b, name) - getattr(a, name)).max()
                    / max(np.abs(getattr(a, name)).max(), 1e-300))
        for name in ("x", "face_displacement")
    }
    result["plastic_strain"] = float(
        np.abs(b.material.plastic_strain - a.material.plastic_strain).max()
        / max(np.abs(a.material.plastic_strain).max(), 1e-300)
    )
    result["same_newton_iterations"] = float(
        [s.newton.iterations for s in old.controller.steps] == [s.newton.iterations for s in new.controller.steps]
    )
    return result


def compare(resolution: tuple[int, int] = (8, 8)) -> None:
    for kind in MESH_KINDS:
        meshes = compare_meshes(kind, resolution)
        runs = compare_runs(kind, resolution)
        identical = all(meshes.values())
        print(
            f"{kind:9s} mesh identical: {identical}"
            + ("" if identical else f" {[k for k, v in meshes.items() if not v]}")
            + f"; final-state rel. diff x {runs['x']:.1e}, faces {runs['face_displacement']:.1e}, "
            f"plastic strain {runs['plastic_strain']:.1e}; same Newton iterations: {bool(runs['same_newton_iterations'])}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--mesh", choices=MESH_KINDS, default="cartesian")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show_mesh", action="store_true", help="export all mesh kinds, do not solve")
    parser.add_argument("--compare", action="store_true", help="compare with core_self_convergence_2D")
    args = parser.parse_args(argv)

    resolution = (args.nx, args.ny)
    if args.compare:
        compare(resolution)
    elif args.show_mesh:
        for kind in MESH_KINDS:
            print(run_mesh(mesh_kind=kind, resolution=resolution))
    else:
        result = run(mesh_kind=args.mesh, resolution=resolution, verbose=args.verbose)
        steps = result.controller.steps
        print(
            f"{args.mesh} {args.nx}x{args.ny}: {result.grid.num_cells} cells, h = {result.mesh_size:.3e}; "
            f"Newton iterations {[s.newton.iterations for s in steps]}"
        )


if __name__ == "__main__":
    main()
