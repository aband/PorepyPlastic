"""Nonuniform Dirichlet plasticity example: python -m examples.dirichlet_example.

Unit square in plane strain, zero body force, initially undeformed J2 material.
On EVERY boundary face prescribe u = factor * boundary_displacement(centers).
The callback specifies spatial data at factor 1; edit it to change the problem.
Load to factor 1, then unload to 0.98. No exact interior solution is assumed.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from examples._solver_setup import prepare_solver, problem_type
from utils.formulation import DEFAULT_FORMULATION, FORMULATION_CHOICES, FormulationName, select_formulation
from utils.load_schedule import AdaptiveLoadOptions
from utils.newton_utils import print_newton_convergence
from utils.problem import PlaneStrainProblem
from utils.state import as_tpsa_state
from utils.visualization import export_jacobian_png, export_plastic_history, export_png, export_strain_png

if TYPE_CHECKING:
    from examples._solver_setup import Controller, StepRecord


def boundary_displacement(centers: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map face centers (3, nb) in metres to prescribed displacement (2, nb).

    At factor 1, u_x = 0.004 m * (x/L) * [1 + 0.2 sin(pi*y/L)], u_y = 0,
    with L = 1 m. Both components are constrained on all four sides.
    """
    length, amplitude = 1.0, 0.004  # metres
    x, y = centers[:2] / length
    return np.vstack((amplitude * x * (1 + 0.2 * np.sin(np.pi * y)), np.zeros_like(y)))


@dataclass(eq=False)
class DirichletResult:
    """Prepared case, accepted solver history, and exported file paths."""

    case: PlaneStrainProblem
    controller: Controller
    outputs: dict[str, Path]
    formulation: FormulationName = DEFAULT_FORMULATION


def run_example(
    *,
    cells_per_axis: int = 10,
    formulation: str = DEFAULT_FORMULATION,
    jacobian: Literal["analytical", "finite-difference"] = "analytical",
    adaptive: bool = False,
    output_dir: str | Path | None = None,
    verbose: bool = False,
) -> DirichletResult:
    """Solve the selected plasticity formulation and optionally export accepted steps.

    Uses PlaneStrainProblem's default material (E=210 GPa, nu=0.3, initial yield
    stress 250 MPa, linear mixed hardening). Fixed mode uses 20 loading steps
    and one unloading step. Adaptive mode chooses substeps toward [1, 0.98].
    Newton enforces every residual block tolerance and failures propagate.
    Python calls write no files unless output_dir is provided; the CLI exports
    to results by default. Field plots show the final state after unloading;
    the Jacobian heatmap shows the converged peak loading increment. Formulation
    selection defaults to "retained" and is independent of jacobian.
    "original" solves [u,r,p];
    "coupled" solves [u,r,p,strain] with complementary weights and plastic face feedback.
    "retained" solves increments [du,dr,dp,d_free] with nested material updates.
    Filenames include both formulation and Jacobian type.
    """
    selected_formulation = select_formulation(formulation)
    if jacobian not in ("analytical", "finite-difference"):
        raise ValueError("jacobian must be 'analytical' or 'finite-difference'.")

    # 1. Create the mesh/material and evaluate the reference boundary values.
    case = problem_type(selected_formulation)(
        cells_per_axis=cells_per_axis, boundary_displacement=boundary_displacement,
    )
    case.prepare_simulation()  # The plastic workflow does not call case.solve().

    # 2. Select the solver and a tangent factory tied to each committed state.
    setup = prepare_solver(case, selected_formulation, jacobian)
    controller, initial = setup.controller, setup.initial

    # 3. Solve absolute load factors; accepted states contain reconstructed strain.
    accepted = (
        controller.iter_adaptive([1.0, 0.98], options=AdaptiveLoadOptions())
        if adaptive else (controller.advance(float(s)) for s in np.r_[np.linspace(0.05, 1, 20), 0.98])
    )
    if verbose:
        print(f"Formulation: {selected_formulation}")
        print(f"Nonuniform Dirichlet plane strain: {case.grid.num_cells} cells, {setup.num_dofs} unknowns; {jacobian} Jacobian")
        print(f"Load control: {'adaptive' if adaptive else 'fixed'}; targets 1 -> 0.98")
        print("step    factor  Newton  cells with alpha>0     alpha_max")
    for index, record in enumerate(accepted, 1):
        if verbose:
            alpha = np.array([state.alpha for state in as_tpsa_state(record.state).material_states])
            print(f"{index:4d} {record.load_factor:9.4f} {record.newton.iterations:7d} "
                  f"{np.count_nonzero(alpha > 0):19d} {np.max(alpha):13.3e}")
            print_newton_convergence(record.newton, extra_block="f" if selected_formulation == "retained" else "eps")
    records: Sequence[StepRecord] = controller.steps
    if verbose:
        rejected = sum(not attempt.accepted for attempt in controller.attempts)
        print(f"Converged: {len(records)} accepted steps, {rejected} rejected attempts")

    # 4. Label each export with its formulation and Jacobian type.
    outputs: dict[str, Path] = {}
    if output_dir is not None:
        prefix = f"dirichlet_example_{selected_formulation}_{jacobian}"
        outputs = export_plastic_history(case.grid, records, folder_name=output_dir, file_name=prefix)
        final = as_tpsa_state(records[-1].state)
        outputs["displacement"] = export_png(
            case.grid, final.x, folder_name=output_dir, file_name=f"{prefix}_final_displacement",
        )
        for name, path in export_strain_png(
            case.grid, final.epsilon, folder_name=output_dir, file_name=f"{prefix}_final",
        ).items():
            outputs[f"strain_{name}"] = path

        # Differentiate the peak increment using its PRECEDING committed history.
        peak_index = next(i for i, record in enumerate(records) if record.load_factor == 1.0)
        peak = records[peak_index]
        old = initial if peak_index == 0 else as_tpsa_state(records[peak_index - 1].state)
        matrix = setup.jacobian_at(peak, old)
        outputs["jacobian"] = export_jacobian_png(
            matrix, formulation=selected_formulation, num_cells=case.grid.num_cells,
            folder_name=output_dir, file_name=f"{prefix}_jacobian",
            title=f"TPSA Jacobian — {selected_formulation} / {jacobian}\nConverged peak loading, factor 1",
        )
    if verbose:
        for name, path in outputs.items():
            print(f"Output ({name}): {path}")
    return DirichletResult(case, controller, outputs, formulation=selected_formulation)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-per-axis", type=int, default=4)
    parser.add_argument("--formulation", choices=FORMULATION_CHOICES, default=DEFAULT_FORMULATION,
                        help="Discrete formulation: original, coupled strain, or retained face displacement (default: retained).")
    parser.add_argument("--jacobian", choices=["analytical", "finite-difference"], default="analytical")
    parser.add_argument("--adaptive", action="store_true", help="Enable load cutbacks and growth.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-export", action="store_true", help="Solve without writing VTK/PNG files.")
    args = parser.parse_args(argv)
    try:
        run_example(cells_per_axis=args.cells_per_axis, formulation=args.formulation, jacobian=args.jacobian,
                    adaptive=args.adaptive, output_dir=None if args.no_export else args.output_dir,
                    verbose=True)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Example failed: {error}\n")


if __name__ == "__main__":
    main()
