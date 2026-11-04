from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from numpy.typing import NDArray

from core.jacobian import AnalyticalJacobian
from core.layout import FloatArray
from core.residual import LoadIncrement, ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory
from utils.newton_utils import (
    NewtonConvergenceError,
    _residual_block_norms,
    _solve_linear_correction,
    _validate_newton_options,
    NewtonDivergenceError,
    BacktrackingOptions,
    LocalUpdateError,
)
from core.linear_solvers import LinearSolver, _checked

BlockTolerances = tuple[float, float, float, float]

@dataclass(eq=False)
class NewtonResult:
    """Converged candidate for one load increment.

    No material history is committed here.
    residual_norms contains rows for iterations 0, ..., iterations,
    with columns ordered [u, r, p, face].
    """

    z: FloatArray
    evaluation: ResidualEvaluation
    iterations: int
    residual_norms: FloatArray
    thresholds: FloatArray
    step_length: FloatArray = field(default_factory=lambda: np.empty(0))
    evaluations: int=0
    linear_iterations: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    linear_times: FloatArray = field(default_factory=lambda: np.empty(0))

def _merit_scales(
        norms: FloatArray,
        absolute: FloatArray,
    )-> FloatArray:
    """Fixed per-block merit scales, as in prototype.newton."""
    scales = np.maximum(norms, absolute)
    scales[[0,3]] = max(scales[0], scales[3]) # selecting u and face
    scales[scales == 0.0] = 1.0
    return scales

def solve_newton(
    assembler: ResidualAssembler,
    loads: LoadIncrement,
    committed: MaterialHistory,
    *,
    jacobian: AnalyticalJacobian | None = None,
    initial_z: FloatArray | None = None,
    max_iterations: int = 20,
    atol: BlockTolerances = (1e-5, 1e-12, 1e-12, 1e-5),
    rtol: BlockTolerances = (1e-8, 1e-8, 1e-8, 1e-8),
    committed_residual: FloatArray | None = None,
    line_search: BacktrackingOptions | None = None,
    linear_solver: LinearSolver | None = None,
) -> NewtonResult:
    """Solve one load increment with fixed committed material history.

    z contains load-step increments [du, dr, dp, d_free].
    Each Newton correction updates all unknowns together.
    Geometry and material parameters remain fixed during the solve.
    """
    absolute, relative = _validate_newton_options(
        max_iterations, atol, rtol, blocks=4,
    )

    if line_search is not None and not isinstance(line_search, BacktrackingOptions):
        raise ValueError("line_search must be BacktrackingOptions or None.")

    if linear_solver is not None and not isinstance(linear_solver, LinearSolver):
        raise ValueError("linear_solver must implement LinearSolver or be None.")

    if jacobian is not None and jacobian.assembler is not assembler:
        raise ValueError("Jacobian must use the supplied residual assembler.")

    layout = assembler.layout
    blocks = (
        layout.u_slice,
        layout.r_slice,
        layout.p_slice,
        layout.face_slice,
    )

    initial = (
        np.zeros(layout.size)
        if initial_z is None
        else np.asarray(initial_z)
    )

    # Validate before float conversion to avoid silently discarding complex data.
    assembler._validate_inputs(initial, loads)

    z = np.array(initial, dtype=np.float64, copy=True)
    old = committed.copy()
    if linear_solver is not None:
        linear_solver.begin_step()

    fixed_loads = LoadIncrement(
        np.array(loads.dirichlet_increment, dtype=np.float64, copy=True),
        np.array(loads.neumann_increment, dtype=np.float64, copy=True),
        np.array(loads.cell_source_increment, dtype=np.float64, copy=True),
    )

    size = assembler.layout.size
    offset = (
        np.zeros(size)
        if committed_residual is None
        else np.asarray(committed_residual)
    )

    if (
        offset.shape != (size,)
        or offset.dtype.kind not in ("f", "i", "u")
        or not np.all(np.isfinite(offset))
    ):
        raise ValueError("committed_residual must be finite.")

    fixed_residual = np.array(offset, dtype=np.float64, copy=True)

    evaluations = 0
    def evaluate(
        candidate: FloatArray,
    ) -> tuple[ResidualEvaluation, FloatArray, FloatArray]:
        nonlocal evaluations
        evaluations += 1

        result = assembler.evaluate(candidate, fixed_loads, old)

        if result.residual.shape != (layout.size,):
            raise RuntimeError("Newton residual has incorrect shape.")

        total_residual = fixed_residual + result.residual
        norms = _residual_block_norms(
            total_residual, layout.size, blocks,
        )
        return result, total_residual, norms

    evaluation, total_residual, norms = evaluate(z)

    # Reference norms remain fixed throughout this load increment.
    with np.errstate(over="ignore", invalid="ignore"):
        thresholds = absolute + relative * norms

    scales = _merit_scales(norms, absolute)

    def merit(values: FloatArray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(np.linalg.norm(values / scales))
        if not np.isfinite(value):
            raise NewtonDivergenceError("Newton line-search merit must remain finite.")
        return value

    if not np.all(np.isfinite(thresholds)):
        raise ValueError("Newton convergence thresholds must remain finite.")

    history = [norms]
    lengths: list[float] = []
    linear_iterations: list[int] = []
    linear_times: list[float] = []

    for iteration in range(max_iterations + 1):
        if np.all(norms <= thresholds):
            return NewtonResult(
                z=z.copy(),
                evaluation=evaluation,
                iterations=iteration,
                residual_norms=np.stack(history),
                thresholds=thresholds.copy(),
                step_length=np.array(lengths, dtype=np.float64),
                evaluations=evaluations,
                linear_iterations=np.array(linear_iterations, dtype=np.int64),
                linear_times=np.array(linear_times, dtype=np.float64)
            )

        if iteration == max_iterations:
            break

        # Construct fixed Jacobian operators only when a correction is needed.
        if jacobian is None:
            jacobian = AnalyticalJacobian(assembler)

        start = time.perf_counter()
        if linear_solver is None:
            # Default path: unchanged
            matrix = jacobian.evaluate(old, evaluation)

            # Solves J * correction = -R for the complete coupled system.
            correction = _solve_linear_correction(
                matrix, total_residual, iteration,
            )
            krylov_iterations = 0
        else:
            linear_solver.bind(assembler, jacobian)
            result = linear_solver.solve(old, evaluation, total_residual, iteration=iteration)
            correction = _checked(result, layout.size, iteration)
            krylov_iterations = result.iterations

        linear_iterations.append(krylov_iterations)
        linear_times.append(time.perf_counter() - start)

        alpha = 1.0
        current_merit = merit(norms) if line_search is not None else 0.0
        reductions = 0 if line_search is None else line_search.max_backtracks

        for backtrack in range(reductions + 1):
            with np.errstate(over="ignore", invalid="ignore"):
                candidate = z + alpha * correction
            if not np.all(np.isfinite(candidate)):
                raise NewtonDivergenceError(
                    f"Nonfinite Newton candidate at iteration {iteration + 1}."
                )

            if line_search is None:
                trial = evaluate(candidate)
                accepted = True
            else:
                try:
                    trial = evaluate(candidate)
                except LocalUpdateError:
                    # A too-long step can break the local return map;
                    # shorten it instead of abandoning the load step.
                    if backtrack == reductions:
                        raise
                    alpha *= line_search.factor
                    continue
                trial_norms = trial[2]
                accepted = bool(
                    np.all(trial_norms <= thresholds)
                    or merit(trial_norms)
                    <= np.sqrt(1.0 - 2.0 * line_search.armijo * alpha) * current_merit
                )

            if accepted:
                z = candidate
                evaluation, total_residual, norms = trial
                history.append(norms)
                lengths.append(alpha)
                break

            if backtrack == reductions:
                raise NewtonConvergenceError(
                    f"Line search failed at correction {iteration + 1} after "
                    f"{reductions} reductions; block norms [u, r, p, face] = "
                    f"{norms}; thresholds = {thresholds}."
                )
            assert line_search is not None
            alpha *= line_search.factor

    raise NewtonConvergenceError(
        f"Newton did not converge after {max_iterations} corrections. "
        f"Block norms [u, r, p, face] = {norms}; "
        f"thresholds = {thresholds}."
    )