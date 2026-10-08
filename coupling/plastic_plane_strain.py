"""Verified homogeneous J2 loading/unloading example; run with python -m coupling.plastic_plane_strain."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from porepy.applications.convergence_analysis import ConvergenceAnalysis

from coupling.jacobian import AnalyticalJacobian, FiniteDifferenceJacobian
from coupling.loading import JacobianFactory, LoadStepController, LoadStepError, LoadStepResult
from coupling.newton import print_newton_convergence
from coupling.plane_strain import PlaneStrainTpsa
from coupling.residual import TpsaOperators, TpsaState, assemble_global_residual
from coupling.transfer import CellToFaceTransfer
from coupling.visualization import export_jacobian_png, export_plastic_history


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
    case: PlaneStrainTpsa, factor: float, *, unloading: bool = False,
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
    full_strain = np.zeros((3, 3))
    full_strain[:2, :2] = 0.5 * (case.displacement_gradient + case.displacement_gradient.T)
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

    case: PlaneStrainTpsa
    operators: TpsaOperators
    transfer: CellToFaceTransfer
    controller: LoadStepController
    checks: tuple[StepCheck, ...]
    outputs: dict[str, Path] = field(default_factory=dict)


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
    case: PlaneStrainTpsa,
    operators: TpsaOperators,
    record: LoadStepResult,
    previous: TpsaState,
    *,
    index: int,
    unloading: bool,
    peak: TpsaState | None = None,
) -> StepCheck:
    """Check equilibrium and every reference field; fail visibly on bad results.

    Reassemble residuals from accepted fields and numerical tractions, and compare
    each block to the actual Newton thresholds. Field errors use cell volumes
    (Frobenius tensor norms), or PorePy face dual volumes for integrated forces.
    During unloading also compare plastic histories directly to the numerical
    peak state. This checker never updates material points or accepts a state.
    """
    grid, state, factor = case.grid, record.state, record.load_factor
    reference = homogeneous_reference(case, factor, unloading=unloading)
    nc = grid.num_cells
    residual = assemble_global_residual(operators, state.x, state.bc_values, state.traction, sources=record.sources)
    norms = np.array([np.linalg.norm(residual[:2 * nc]), np.linalg.norm(residual[2 * nc:3 * nc]), np.linalg.norm(residual[3 * nc:])])
    thresholds = record.newton.thresholds
    if thresholds.shape != (3,) or not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0):
        raise NumericalCheckError(f"Step {index}: invalid Newton convergence thresholds.")
    if not np.all(np.isfinite(norms)) or np.any(norms > thresholds):
        raise NumericalCheckError(f"Step {index}: residual norms {norms} exceed thresholds {thresholds}.")

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
    max_iterations: int = 20,
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
    Every output filename includes the selected jacobian type.
    With output_dir=None (the Python default), no output files are created.
    jacobian selects "analytical" (default) or "finite-difference"; the latter
    alone uses finite_difference_step. Both keep the coupled [u,r,p] solve.
    verbose=True prints per-iteration residual norms and observed momentum rates.
    show_l2_errors=True independently enables the L2 error summary (off by default).
    Numerical checks always run and retain their errors in the returned checks.
    """
    if jacobian not in ("analytical", "finite-difference"):
        raise ValueError("jacobian must be 'analytical' or 'finite-difference'.")
    for name, count in (("loading_steps", loading_steps), ("unloading_steps", unloading_steps)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(peak_strain) or peak_strain <= 0:
        raise ValueError("peak_strain must be finite and positive.")
    if not np.isfinite(unload_fraction) or not 0 < unload_fraction <= 1:
        raise ValueError("unload_fraction must lie in (0, 1].")
    case = PlaneStrainTpsa(cells_per_axis=cells_per_axis, displacement_gradient=np.diag([peak_strain, 0.0]))
    if homogeneous_reference(case, 1).alpha <= 0:
        raise ValueError("peak_strain must be large enough to produce plastic loading.")
    homogeneous_reference(case, 1 - unload_fraction, unloading=True)
    case.prepare_simulation()
    operators, transfer = TpsaOperators(case), CellToFaceTransfer(case.grid)
    initial = TpsaState.zeros(case.grid)
    factory: JacobianFactory | None = None
    if jacobian == "analytical":
        factory = lambda old: AnalyticalJacobian(operators, transfer, case.material_points, old)
    controller = LoadStepController(
        operators, transfer, case.material_points, initial, case.bc_values,
        jacobian_factory=factory,
        finite_difference_step=finite_difference_step, max_iterations=max_iterations,
    )
    factors = np.r_[np.linspace(1 / loading_steps, 1, loading_steps),
                    np.linspace(1, 1 - unload_fraction, unloading_steps + 1)[1:]]
    if verbose:
        print(f"Plastic plane strain: {case.grid.num_cells} cells, {4 * case.grid.num_cells} unknowns")
        print(f"Peak epsilon_xx={peak_strain:g}; final load factor={1 - unload_fraction:g}; {jacobian} Jacobian")
        print("Newton rates use the momentum residual: ratio_u = r[k]/r[k-1], order_u = log(ratio_u[k])/log(ratio_u[k-1]).")
        print("Order is diagnostic; -- means insufficient, nondecreasing, or near-noise-floor data.")
        print("step phase      factor  Newton       ||R_u||       ||R_r||       ||R_p||     alpha_max")
    checks: list[StepCheck] = []
    previous, peak = initial, None
    for index, factor in enumerate(factors, 1):
        record = controller.advance(float(factor))
        check = check_step(case, operators, record, previous, index=index, unloading=index > loading_steps, peak=peak)
        checks.append(check)
        previous = record.state
        if index == loading_steps:
            peak = record.state
        if verbose:
            ru, rr, rp = check.residual_norms
            print(f"{index:4d} {check.phase:9s} {factor:7.4f} {check.iterations:7d} {ru:13.3e} {rr:13.3e} {rp:13.3e} {check.alpha_max:13.3e}")
            print_newton_convergence(record.newton)
    if show_l2_errors:
        print("Maximum absolute discrete L2 errors over all accepted steps:")
        for name, unit in _ERROR_UNITS.items():
            print(f"  {name}: {max(check.errors[name] for check in checks):.3e} [{unit}]")
    if verbose:
        print("Numerical checks: PASS (equilibrium, analytical J2 response, plane strain, elastic unloading)")
    outputs: dict[str, Path] = {}
    if output_dir is not None:
        records = controller.steps
        peak_record = records[loading_steps - 1]
        # Differentiate the peak increment from its PREVIOUS committed history.
        # Using the peak itself as old would turn it into a zero-increment tangent.
        old = initial if loading_steps == 1 else records[loading_steps - 2].state
        peak_jacobian = (
            AnalyticalJacobian(operators, transfer, case.material_points, old)
            if jacobian == "analytical" else FiniteDifferenceJacobian(
                operators, transfer, case.material_points, old, step=finite_difference_step,
            )
        )
        matrix = peak_jacobian(peak_record.state.x, peak_record.newton.evaluation)
        file_name = f"plastic_plane_strain_{jacobian}"
        outputs = export_plastic_history(
            case.grid, records, folder_name=output_dir, file_name=file_name,
        )
        outputs["jacobian"] = export_jacobian_png(
            matrix, folder_name=output_dir, file_name=f"{file_name}_jacobian",
            title=(f"Coupled Jacobian — {jacobian}\n"
                   f"Peak loading: step {loading_steps}, factor {peak_record.load_factor:g} (converged)"),
        )
    if verbose:
        for name, path in outputs.items():
            print(f"Output ({name}): {path}")
    return PlasticPlaneStrainResult(case, operators, transfer, controller, tuple(checks), outputs)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run and verify a homogeneous plastic TPSA loading/unloading example.")
    parser.add_argument("--cells-per-axis", type=int, default=3)
    parser.add_argument("--loading-steps", type=int, default=20)
    parser.add_argument("--unloading-steps", type=int, default=2)
    parser.add_argument("--peak-strain", type=float, default=0.004)
    parser.add_argument("--unload-fraction", type=float, default=0.02)
    parser.add_argument("--jacobian", choices=("analytical", "finite-difference"), default="analytical")
    parser.add_argument("--fd-step", type=float, default=1e-10, help="Local perturbation for the finite-difference Jacobian only.")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="VTK/PNG directory (default: results).")
    parser.add_argument("--no-export", action="store_true", help="Run numerical checks without writing files.")
    parser.add_argument("--show-l2-errors", action="store_true", help="Print the L2 error summary (hidden by default; numerical checks always run).")
    args = parser.parse_args(argv)
    try:
        run_example(
            cells_per_axis=args.cells_per_axis, loading_steps=args.loading_steps,
            unloading_steps=args.unloading_steps, peak_strain=args.peak_strain,
            unload_fraction=args.unload_fraction, finite_difference_step=args.fd_step,
            jacobian=args.jacobian, max_iterations=args.max_iterations, verbose=True,
            show_l2_errors=args.show_l2_errors,
            output_dir=None if args.no_export else args.output_dir,
        )
    except (ValueError, LoadStepError, NumericalCheckError, OSError) as error:
        parser.exit(1, f"Example failed: {error}\n")


if __name__ == "__main__":
    main()
