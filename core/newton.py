from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.jacobian import AnalyticalJacobian
from core.layout import FloatArray
from core.residual import LoadIncrement, ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory
from utils.newton_utils import (
    NewtonConvergenceError,
    _residual_block_norms,
    _solve_linear_correction,
    _validate_newton_options,
)

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
) -> NewtonResult:
    """Solve one load increment with fixed committed material history.

    z contains load-step increments [du, dr, dp, d_free].
    Each Newton correction updates all unknowns together.
    Geometry and material parameters remain fixed during the solve.
    """
    absolute, relative = _validate_newton_options(
        max_iterations, atol, rtol, blocks=4,
    )

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
    fixed_loads = LoadIncrement(
        np.array(loads.dirichlet_increment, dtype=np.float64, copy=True),
        np.array(loads.neumann_increment, dtype=np.float64, copy=True),
        np.array(loads.cell_source_increment, dtype=np.float64, copy=True),
    )

    def evaluate(
        candidate: FloatArray,
    ) -> tuple[ResidualEvaluation, FloatArray]:
        result = assembler.evaluate(candidate, fixed_loads, old)
        norms = _residual_block_norms(
            result.residual, layout.size, blocks,
        )
        return result, norms

    evaluation, norms = evaluate(z)

    # Reference norms remain fixed throughout this load increment.
    with np.errstate(over="ignore", invalid="ignore"):
        thresholds = absolute + relative * norms

    if not np.all(np.isfinite(thresholds)):
        raise ValueError("Newton convergence thresholds must remain finite.")

    history = [norms]

    for iteration in range(max_iterations + 1):
        if np.all(norms <= thresholds):
            return NewtonResult(
                z=z.copy(),
                evaluation=evaluation,
                iterations=iteration,
                residual_norms=np.stack(history),
                thresholds=thresholds.copy(),
            )

        if iteration == max_iterations:
            break

        # Construct fixed Jacobian operators only when a correction is needed.
        if jacobian is None:
            jacobian = AnalyticalJacobian(assembler)

        matrix = jacobian.evaluate(old, evaluation)

        # Solves J * correction = -R for the complete coupled system.
        correction = _solve_linear_correction(
            matrix, evaluation.residual, iteration,
        )

        with np.errstate(over="ignore", invalid="ignore"):
            candidate = z + correction

        if not np.all(np.isfinite(candidate)):
            raise RuntimeError(
                f"Nonfinite Newton candidate at iteration {iteration + 1}."
            )

        z = candidate
        evaluation, norms = evaluate(z)
        history.append(norms)

    raise NewtonConvergenceError(
        f"Newton did not converge after {max_iterations} corrections. "
        f"Block norms [u, r, p, face] = {norms}; "
        f"thresholds = {thresholds}."
    )