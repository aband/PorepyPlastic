"""Verified homogeneous J2 loading/unloading example; run with python -m examples.plastic_plane_strain."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from porepy.applications.convergence_analysis import ConvergenceAnalysis

from examples._solver_setup import prepare_solver, problem_type
from prototype.loading import RetainedLoadStepResult
from prototype.operators import RetainedFaceOperators
from prototype.state import RetainedState, assemble_state_residual
from utils.formulation import DEFAULT_FORMULATION, FORMULATION_CHOICES, FormulationName, select_formulation
from utils.load_schedule import AdaptiveLoadOptions, LoadStepError
from utils.newton_utils import print_newton_convergence
from utils.problem import PlaneStrainProblem
from utils.state import TpsaState, as_tpsa_state
from utils.transfer import CellToFaceTransfer
from utils.visualization import export_jacobian_png, export_plastic_history

if TYPE_CHECKING:
    from examples._solver_setup import Controller, Operators, StepRecord


class NumericalCheckError(RuntimeError):
    """A converged numerical state failed a physical or reference-solution check."""


@dataclass(eq=False)
class HomogeneousReference:
    """Uniform full 3D material fields for the proportional loading path."""

    epsilon: NDArray[np.float64]
    stress: NDArray[np.float64]
    plastic_strain: NDArray[np.float64]
    backstress: NDArray[np.float64]
    alpha: float


def homogeneous_reference(
    case: PlaneStrainProblem, factor: float, *, unloading: bool = False,
) -> HomogeneousReference:
    """Closed-form J2 solution from virgin history, independent of the return map.

    Loading is proportional to case.displacement_gradient for factors in [0,1].
    Mixed hardening must be linear. On unloading, plastic history is frozen at
    factor 1 and stress changes elastically; reject a target beyond reverse yield.
    This reference is for homogeneous affine loading, not general boundary data.
    """
    if not np.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError("Reference load factor must lie in [0, 1].")
    material, parameters = case.material, case.hardening_parameters
    if parameters.sigma_u != parameters.sigma_y:
        raise ValueError("The analytical reference requires linear mixed hardening.")
    mu, lam = material.isotropic_shear_modulus, material.lame_parameter
    # The geometric reference also checks that the case has affine boundary data.
    full_strain = np.asarray(case.reference_strain().to_numpy(), dtype=np.float64)
    epsilon = (1.0 if unloading else factor) * full_strain
    predictor = 2 * mu * epsilon + lam * np.trace(epsilon) * np.eye(3)
    deviator = predictor - np.trace(predictor) / 3 * np.eye(3)
    norm = float(np.linalg.norm(deviator))
    c = np.sqrt(2 / 3)
    gamma = max(0.0, norm - c * parameters.sigma_y) / (2 * mu + 2 / 3 * parameters.H_bar)
    plastic = np.zeros((3, 3)) if gamma == 0 else gamma * deviator / norm
    sigma = predictor - 2 * mu * plastic
    backstress = 2 / 3 * (1 - parameters.theta) * parameters.H_bar * plastic
    alpha = float(c * gamma)
    if unloading:
        depsilon = (factor - 1) * full_strain
        sigma += 2 * mu * depsilon + lam * np.trace(depsilon) * np.eye(3)
        shifted = sigma - np.trace(sigma) / 3 * np.eye(3) - backstress
        radius = c * (parameters.sigma_y + parameters.theta * parameters.H_bar * alpha)
        if np.linalg.norm(shifted) > radius + 1e-10 * parameters.sigma_y:
            raise ValueError("Unloading reaches reverse plastic yield; reduce unload_fraction.")
    return HomogeneousReference(factor * full_strain, sigma, plastic, backstress, alpha)


@dataclass(eq=False)
class StepCheck:
    """Diagnostics for one accepted, checked step; errors are absolute discrete L2 norms."""

    index: int
    phase: Literal["loading", "unloading"]
    load_factor: float
    iterations: int
    residual_norms: NDArray[np.float64]
    alpha_max: float
    errors: dict[str, float]
    limits: dict[str, float]


@dataclass(eq=False)
class PlasticPlaneStrainResult:
    """Prepared model, solver, accepted records, and checks for subsequent inspection."""

    case: PlaneStrainProblem
    operators: Operators
    transfer: CellToFaceTransfer
    controller: Controller
    checks: tuple[StepCheck, ...]
    outputs: dict[str, Path] = field(default_factory=dict)
    formulation: FormulationName = DEFAULT_FORMULATION


# Absolute L2 tolerance floors for this unit-square SI benchmark. Relative
# tolerance adds 1e-7 times the reference L2 norm; zero references use the floor.
_ERROR_FLOORS = {
    "displacement": 1e-12, "rotation_stress": 2.0, "total_pressure": 2.0,
    "strain": 1e-11, "stress": 2.0, "plastic_strain": 1e-11,
    "backstress": 0.01, "alpha": 1e-11, "traction": 1.0,
}
_ERROR_UNITS = {
    "displacement": "m^2", "rotation_stress": "Pa m", "total_pressure": "Pa m",
    "strain": "m", "stress": "Pa m", "plastic_strain": "m",
    "backstress": "Pa m", "alpha": "m", "traction": "N",
}


def check_step(
    case: PlaneStrainProblem,
    operators: Operators,
    record: StepRecord,
    previous: TpsaState | RetainedState,
    *,
    index: int,
    unloading: bool,
    peak: TpsaState | RetainedState | None = None,
) -> StepCheck:
    """Check equilibrium and every reference field; fail visibly on bad results.

    Reassemble residuals from accepted fields and numerical tractions, and compare
    each block to the actual Newton thresholds. Field errors use cell volumes
    (Frobenius tensor norms), or PorePy face dual volumes for integrated forces.
    During unloading also compare plastic histories directly to the numerical
    peak state. This checker never updates material points or accepts a state.
    """
    grid, state, factor = case.grid, as_tpsa_state(record.state), record.load_factor
    previous = as_tpsa_state(previous)
    peak = as_tpsa_state(peak) if peak is not None else None
    reference = homogeneous_reference(case, factor, unloading=unloading)
    nc = grid.num_cells
    if isinstance(record, RetainedLoadStepResult):
        if not isinstance(operators, RetainedFaceOperators):
            raise ValueError("Retained records require retained operators.")
        residual = assemble_state_residual(operators, record.state)
        blocks = np.split(residual, [2 * nc, 3 * nc, 4 * nc])
    else:
        from alternatives.coupled.residual import CoupledOperators, assemble_coupled_residual
        from alternatives.original.residual import TpsaOperators, assemble_global_residual

        if not isinstance(operators, TpsaOperators):
            raise ValueError("TPSA records require TPSA operators.")
        if isinstance(operators, CoupledOperators):
            residual = assemble_coupled_residual(operators, state, sources=record.sources)
            blocks = np.split(residual, [2 * nc, 3 * nc, 4 * nc])
        else:
            residual = assemble_global_residual(operators, state.x, state.bc_values, state.traction, sources=record.sources)
            blocks = np.split(residual, [2 * nc, 3 * nc])
    norms = np.array([np.linalg.norm(block) for block in blocks])
    thresholds = record.newton.thresholds
    if thresholds.shape != norms.shape or not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0):
        raise NumericalCheckError(f"Step {index}: invalid Newton convergence thresholds.")
    if not np.all(np.isfinite(norms)) or np.any(norms > thresholds):
        raise NumericalCheckError(f"Step {index}: residual norms {norms} exceed thresholds {thresholds}.")
    if isinstance(record, RetainedLoadStepResult) and isinstance(operators, RetainedFaceOperators):
        # Check BOTH outward forces; the shared-face average can hide an imbalance.
        exact_side = reference.stress[:2, :2] @ (operators.side_normals * operators.side_areas)
        side_error = np.abs(record.state.side_traction - exact_side)
        if not np.all(np.isfinite(side_error)) or np.any(side_error > 1.0 + 1e-7 * np.abs(exact_side)):
            raise NumericalCheckError(f"Step {index}: one-sided traction differs from the homogeneous reference.")

    fields = {
        name: np.stack([getattr(history, name).to_numpy() for history in state.material_states], axis=2)
        for name in ("stress", "plastic_strain", "backstress")
    }
    alpha = np.array([history.alpha for history in state.material_states])
    alpha_previous = np.array([history.alpha for history in previous.material_states])
    if np.any(alpha < -1e-11) or np.any(alpha < alpha_previous - 1e-11):
        raise NumericalCheckError(f"Step {index}: accumulated plastic strain decreased or became negative.")
    if not unloading and factor == 1 and np.any(alpha <= 0):
        raise NumericalCheckError(f"Step {index}: the peak load did not produce plastic strain in every cell.")
    if not np.allclose(state.epsilon[2], 0, atol=1e-14) or not np.allclose(state.epsilon[:, 2], 0, atol=1e-14):
        raise NumericalCheckError(f"Step {index}: total strain violates plane strain.")

    def repeated(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.repeat(values[:, :, None], nc, axis=2)

    comparisons = {
        "displacement": (state.x[:2 * nc].reshape((2, nc), order="F"), factor * case.reference_displacement(grid.cell_centers)),
        "rotation_stress": (state.x[2 * nc:3 * nc], np.full(nc, factor * case.reference_rotation_stress())),
        "total_pressure": (state.x[3 * nc:], np.full(nc, factor * case.reference_total_pressure())),
        "strain": (state.epsilon, repeated(reference.epsilon)),
        "stress": (fields["stress"], repeated(reference.stress)),
        "plastic_strain": (fields["plastic_strain"], repeated(reference.plastic_strain)),
        "backstress": (fields["backstress"], repeated(reference.backstress)),
        "alpha": (alpha, np.full(nc, reference.alpha)),
        "traction": (state.traction, reference.stress[:2, :2] @ grid.face_normals[:2]),
    }
    errors, limits = {}, {}
    for name, (numerical, exact) in comparisons.items():
        is_cc = name != "traction"
        error = float(ConvergenceAnalysis.lp_error(
            grid, exact.ravel(order="F"), numerical.ravel(order="F"),
            is_cc=is_cc, p=2, relative=False,
        ))
        scale = float(ConvergenceAnalysis.lp_error(
            grid, exact.ravel(order="F"), np.zeros(exact.size),
            is_cc=is_cc, p=2, relative=False,
        ))
        limit = _ERROR_FLOORS[name] + 1e-7 * scale
        if not np.isfinite(error) or error > limit:
            raise NumericalCheckError(f"Step {index}: {name} L2 error {error:.3e} exceeds {limit:.3e} {_ERROR_UNITS[name]}.")
        errors[name], limits[name] = error, limit

    if unloading:
        if peak is None:
            raise ValueError("Unloading checks require the accepted peak state.")
        for name, tolerance in (("alpha", 1e-11), ("plastic_strain", 1e-11), ("backstress", 0.01)):
            if name == "alpha":
                change = alpha - np.array([history.alpha for history in peak.material_states])
            else:
                peak_field = np.stack([getattr(history, name).to_numpy() for history in peak.material_states], axis=2)
                change = fields[name] - peak_field
            if not np.all(np.isfinite(change)) or np.max(np.abs(change)) > tolerance:
                raise NumericalCheckError(f"Step {index}: {name} changed during elastic unloading.")
    return StepCheck(index, "unloading" if unloading else "loading", factor, record.newton.iterations, norms, float(np.max(alpha)), errors, limits)


def run_example(
    *,
    cells_per_axis: int = 3,
    loading_steps: int = 20,
    unloading_steps: int = 2,
    peak_strain: float = 0.004,
    unload_fraction: float = 0.02,
    finite_difference_step: float = 1e-10,
    jacobian: str = "analytical",
    formulation: str = DEFAULT_FORMULATION,
    max_iterations: int = 20,
    adaptive: AdaptiveLoadOptions | None = None,
    verbose: bool = False,
    show_l2_errors: bool = False,
    output_dir: str | Path | None = None,
) -> PlasticPlaneStrainResult:
    """Run constrained x-extension on the unit square, then small elastic unloading.

    All boundary faces have u_x = factor * peak_strain * x, u_y = 0. Body forces
    vanish. Load factors rise from 1/loading_steps to 1, then decrease to
    1-unload_fraction. Validate that the peak yields and unloading stays elastic
    BEFORE solving. Return the model and every accepted state/check; exceptions
    propagate on either Newton failure or a failed numerical check. If output_dir
    is supplied, export the accepted history to VTK/PVD and PNG after all checks
    pass, including a Jacobian heatmap at the converged peak loading state.
    Every output filename includes the selected formulation and jacobian type.
    With output_dir=None (the Python default), no output files are created.
    jacobian selects "analytical" (default) or "finite-difference"; the latter
    alone uses finite_difference_step. Both solve all selected unknowns together.
    verbose=True prints per-iteration residual norms and observed momentum rates
    (face-balance rates for the retained formulation).
    show_l2_errors=True independently enables the L2 error summary (off by default).
    Numerical checks always run and retain their errors in the returned checks.
    adaptive=None keeps the fixed loading_steps/unloading_steps schedule. Supply
    AdaptiveLoadOptions to select substeps toward the exact targets 1 and
    1-unload_fraction; the two fixed step counts are then unused. Every accepted
    substep is checked before proceeding and is included in plots and exports.
    formulation defaults to "retained". "original" solves [u,r,p] for comparison.
    "coupled" solves [u,r,p,strain] with complementary weights and plastic face feedback.
    "retained" solves increments [du,dr,dp,d_free] with nested material updates.
    """
    selected_formulation = select_formulation(formulation)
    if adaptive is not None and not isinstance(adaptive, AdaptiveLoadOptions):
        raise ValueError("adaptive must be AdaptiveLoadOptions or None.")
    if jacobian not in ("analytical", "finite-difference"):
        raise ValueError("jacobian must be 'analytical' or 'finite-difference'.")
    for name, count in (("loading_steps", loading_steps), ("unloading_steps", unloading_steps)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(peak_strain) or peak_strain <= 0:
        raise ValueError("peak_strain must be finite and positive.")
    if not np.isfinite(unload_fraction) or not 0 < unload_fraction <= 1:
        raise ValueError("unload_fraction must lie in (0, 1].")
    case = problem_type(selected_formulation)(cells_per_axis=cells_per_axis, displacement_gradient=np.diag([peak_strain, 0.0]))
    if homogeneous_reference(case, 1).alpha <= 0:
        raise ValueError("peak_strain must be large enough to produce plastic loading.")
    homogeneous_reference(case, 1 - unload_fraction, unloading=True)
    case.prepare_simulation()
    setup = prepare_solver(
        case, selected_formulation, jacobian, finite_difference_step=finite_difference_step,
        max_iterations=max_iterations,
    )
    operators, controller = setup.operators, setup.controller
    transfer, initial = setup.transfer, setup.initial
    factors = np.r_[np.linspace(1 / loading_steps, 1, loading_steps),
                    np.linspace(1, 1 - unload_fraction, unloading_steps + 1)[1:]]
    if verbose:
        print(f"Formulation: {selected_formulation}")
        print(f"Plastic plane strain: {case.grid.num_cells} cells, {setup.num_dofs} unknowns")
        print(f"Peak epsilon_xx={peak_strain:g}; final load factor={1 - unload_fraction:g}; {jacobian} Jacobian")
        if adaptive is not None:
            print(f"Adaptive load stepping: initial={adaptive.initial_increment:g}, "
                  f"min={adaptive.min_increment:g}, max={adaptive.max_increment:g}")
        rate_block = "f" if selected_formulation == "retained" else "u"
        rate_name = "face-balance" if selected_formulation == "retained" else "momentum"
        print(f"Newton rates use the {rate_name} residual: ratio_{rate_block} = r[k]/r[k-1], "
              f"order_{rate_block} = log(ratio_{rate_block}[k])/log(ratio_{rate_block}[k-1]).")
        print("Order is diagnostic; -- means insufficient, nondecreasing, or near-noise-floor data.")
        print("step phase      factor  Newton       ||R_u||       ||R_r||       ||R_p||     alpha_max")
    checks: list[StepCheck] = []
    previous, peak = initial, None
    accepted = (
        (controller.advance(float(factor)) for factor in factors) if adaptive is None
        else controller.iter_adaptive([1.0, 1 - unload_fraction], options=adaptive)
    )
    reported_attempts = 0
    for index, record in enumerate(accepted, 1):
        check = check_step(case, operators, record, previous, index=index,
                           unloading=peak is not None, peak=peak)
        checks.append(check)
        previous = as_tpsa_state(record.state)
        if record.load_factor == 1.0:
            peak = as_tpsa_state(record.state)
        if verbose:
            for attempt in controller.attempts[reported_attempts:]:
                if not attempt.accepted:
                    print(f"  Cutback: rejected factor {attempt.load_factor:g} "
                          f"from {attempt.start_factor:g} (Newton nonconvergence)")
            reported_attempts = len(controller.attempts)
            ru, rr, rp = check.residual_norms[:3]
            print(f"{index:4d} {check.phase:9s} {record.load_factor:7.4f} {check.iterations:7d} {ru:13.3e} {rr:13.3e} {rp:13.3e} {check.alpha_max:13.3e}")
            print_newton_convergence(record.newton, extra_block="f" if selected_formulation == "retained" else "eps")
    if verbose and adaptive is not None:
        rejected = sum(not attempt.accepted for attempt in controller.attempts)
        print(f"Adaptive stepping: {len(checks)} accepted steps, {rejected} rejected attempts")
    if show_l2_errors:
        print("Maximum absolute discrete L2 errors over all accepted steps:")
        for name, unit in _ERROR_UNITS.items():
            print(f"  {name}: {max(check.errors[name] for check in checks):.3e} [{unit}]")
    if verbose:
        print("Numerical checks: PASS (equilibrium, analytical J2 response, plane strain, elastic unloading)")
    outputs: dict[str, Path] = {}
    if output_dir is not None:
        records: Sequence[StepRecord] = controller.steps
        peak_index = next(i for i, record in enumerate(records) if record.load_factor == 1.0)
        peak_record = records[peak_index]
        # Differentiate the peak increment from its PREVIOUS committed history.
        # Using the peak itself as old would turn it into a zero-increment tangent.
        old = initial if peak_index == 0 else as_tpsa_state(records[peak_index - 1].state)
        matrix = setup.jacobian_at(peak_record, old)
        file_name = f"plastic_plane_strain_{selected_formulation}_{jacobian}"
        outputs = export_plastic_history(
            case.grid, records, folder_name=output_dir, file_name=file_name,
        )
        outputs["jacobian"] = export_jacobian_png(
            matrix, formulation=selected_formulation, num_cells=case.grid.num_cells,
            folder_name=output_dir, file_name=f"{file_name}_jacobian",
            title=(f"TPSA Jacobian — {selected_formulation} / {jacobian}\n"
                   f"Peak loading: step {peak_index + 1}, factor {peak_record.load_factor:g} (converged)"),
        )
    if verbose:
        for name, path in outputs.items():
            print(f"Output ({name}): {path}")
    return PlasticPlaneStrainResult(case, operators, transfer, controller, tuple(checks), outputs,
                                   formulation=selected_formulation)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run and verify a homogeneous plastic TPSA loading/unloading example.")
    parser.add_argument("--cells-per-axis", type=int, default=3)
    parser.add_argument("--loading-steps", type=int, default=20, help="Number of loading steps in fixed-step mode.")
    parser.add_argument("--unloading-steps", type=int, default=2, help="Number of unloading steps in fixed-step mode.")
    parser.add_argument("--peak-strain", type=float, default=0.004)
    parser.add_argument("--unload-fraction", type=float, default=0.02)
    parser.add_argument("--formulation", choices=FORMULATION_CHOICES, default=DEFAULT_FORMULATION,
                        help="Discrete formulation: original, coupled strain, or retained face displacement (default: retained).")
    parser.add_argument("--jacobian", choices=("analytical", "finite-difference"), default="analytical")
    parser.add_argument("--fd-step", type=float, default=1e-10, help="Local perturbation for the finite-difference Jacobian only.")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--adaptive", action="store_true", help="Adapt increments between peak and final targets instead of using fixed step counts.")
    parser.add_argument("--initial-increment", type=float, default=0.05)
    parser.add_argument("--min-increment", type=float, default=1e-4)
    parser.add_argument("--max-increment", type=float, default=0.2)
    parser.add_argument("--cutback-factor", type=float, default=0.5)
    parser.add_argument("--growth-factor", type=float, default=1.5)
    parser.add_argument("--target-iterations", type=int, default=4, help="Grow adaptive increments after at most this many Newton corrections.")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum accepted substeps in an adaptive run.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="VTK/PNG directory (default: results).")
    parser.add_argument("--no-export", action="store_true", help="Run numerical checks without writing files.")
    parser.add_argument("--show-l2-errors", action="store_true", help="Print the L2 error summary (hidden by default; numerical checks always run).")
    args = parser.parse_args(argv)
    try:
        adaptive = AdaptiveLoadOptions(
            initial_increment=args.initial_increment, min_increment=args.min_increment,
            max_increment=args.max_increment, cutback_factor=args.cutback_factor,
            growth_factor=args.growth_factor, target_iterations=args.target_iterations,
            max_retries=args.max_retries, max_steps=args.max_steps,
        ) if args.adaptive else None
        run_example(
            cells_per_axis=args.cells_per_axis, loading_steps=args.loading_steps,
            unloading_steps=args.unloading_steps, peak_strain=args.peak_strain,
            unload_fraction=args.unload_fraction, finite_difference_step=args.fd_step,
            jacobian=args.jacobian, formulation=args.formulation,
            max_iterations=args.max_iterations, adaptive=adaptive, verbose=True,
            show_l2_errors=args.show_l2_errors,
            output_dir=None if args.no_export else args.output_dir,
        )
    except (ValueError, NotImplementedError, LoadStepError, NumericalCheckError, OSError) as error:
        parser.exit(1, f"Example failed: {error}\n")


if __name__ == "__main__":
    main()
