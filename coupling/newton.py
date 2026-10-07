"""One fully coupled Newton load step with a caller-supplied Jacobian."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray

from coupling.residual import TpsaOperators, TpsaResidual, TpsaState, evaluate_global_residual
from coupling.transfer import CellToFaceTransfer
from material_state import MaterialPoint

JacobianMatrix: TypeAlias = sps.spmatrix | sps.sparray | NDArray[np.float64]
JacobianCallback: TypeAlias = Callable[[NDArray[np.float64], TpsaResidual], JacobianMatrix]
BlockTolerances: TypeAlias = tuple[float, float, float]


class NewtonConvergenceError(RuntimeError):
    """The last evaluated candidate did not meet all block residual tolerances."""


@dataclass(eq=False)
class NewtonResult:
    """Returned only after convergence; accepting trial remains the caller's choice.

    iterations counts Newton corrections, so an equilibrated initial guess uses
    zero. residual_norms has shape (iterations + 1, 3), including the initial
    evaluation. Its columns and the fixed thresholds are ordered [u, r, p].
    evaluation contains the final residual, state, and cell stress correction.
    """

    evaluation: TpsaResidual
    iterations: int
    residual_norms: NDArray[np.float64]
    thresholds: NDArray[np.float64]

    @property
    def trial(self) -> TpsaState:
        """The independent converged candidate, including its numerical tractions."""
        return self.evaluation.trial


def newton_convergence_rates(
    residual_norms: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return reduction ratios and observed orders for each residual block.

    Input and both outputs have shape (evaluations, 3), ordered [u,r,p].
    For r_k = ||R_block(x_k)||, ratio_k = r_k/r_(k-1) and
    order_k = log(r_k/r_(k-1)) / log(r_(k-1)/r_(k-2)). Each stencil stays within
    ONE load step. Blockwise rates avoid combining residuals with different units.

    Undefined entries are NaN. Order requires three positive, strictly decreasing
    norms above 100*machine_epsilon*max(history) for that block and a resolvable
    logarithmic denominator. This relative floor is a heuristic for numerical
    noise, not a solver tolerance. Ratios remain available below that floor.
    These residual-based estimates do not prove asymptotic convergence order,
    especially at yield switches or when only a few corrections are needed.
    """
    norms = np.asarray(residual_norms, dtype=np.float64)
    if (norms.ndim != 2 or norms.shape[1] != 3 or norms.shape[0] < 1
            or not np.all(np.isfinite(norms)) or np.any(norms < 0)):
        raise ValueError("Expected finite nonnegative residual norms with shape (evaluations, 3).")
    ratios, orders = np.full_like(norms, np.nan), np.full_like(norms, np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(norms[1:], norms[:-1], out=ratios[1:], where=norms[:-1] > 0)
    ratios[~np.isfinite(ratios)] = np.nan
    logs = np.zeros_like(norms)
    np.log(norms, out=logs, where=norms > 0)
    previous = logs[1:-1] - logs[:-2]
    current = logs[2:] - logs[1:-1]
    epsilon = np.finfo(np.float64).eps
    floor = 100 * epsilon * np.max(norms, axis=0)
    valid = (
        (norms[:-2] > norms[1:-1]) & (norms[1:-1] > norms[2:])
        & (norms[2:] > floor) & (np.abs(previous) > np.sqrt(epsilon))
    )
    np.divide(current, previous, out=orders[2:], where=valid)
    return ratios, orders


def print_newton_convergence(result: NewtonResult, *, indent: str = "    ") -> None:
    """Print one load step's iteration history, with rates for the momentum block.

    The retained rotation/pressure equations are linear, so their norms are shown
    separately while ratio_u/order_u describe the nonlinear force equation.
    Initial evaluation is iteration 0; -- denotes an unavailable order or ratio.
    """
    ratios, orders = newton_convergence_rates(result.residual_norms)
    print(f"{indent}iter       ||R_u||       ||R_r||       ||R_p||      ratio_u   order_u")
    for iteration, (ru, rr, rp) in enumerate(result.residual_norms):
        ratio, order = ratios[iteration, 0], orders[iteration, 0]
        ratio_text = f"{ratio:.3e}" if np.isfinite(ratio) else "--"
        order_text = f"{order:.3f}" if np.isfinite(order) else "--"
        print(f"{indent}{iteration:4d} {ru:13.3e} {rr:13.3e} {rp:13.3e} {ratio_text:>12} {order_text:>9}")


def _block_norms(residual: NDArray[np.float64], nc: int) -> NDArray[np.float64]:
    if residual.shape != (4 * nc,) or not np.all(np.isfinite(residual)):
        raise RuntimeError("Newton evaluation returned a non-finite or incorrectly shaped residual.")
    norms = np.array([
        np.linalg.norm(residual[:2 * nc]),
        np.linalg.norm(residual[2 * nc:3 * nc]),
        np.linalg.norm(residual[3 * nc:]),
    ])
    if not np.all(np.isfinite(norms)):
        raise RuntimeError("Newton block residual norms are not finite.")
    return norms


def solve_newton(
    operators: TpsaOperators,
    transfer: CellToFaceTransfer,
    material_points: Sequence[MaterialPoint],
    committed: TpsaState,
    bc_values: NDArray[np.float64],
    jacobian: JacobianCallback,
    *,
    initial_x: NDArray[np.float64] | None = None,
    sources: NDArray[np.float64] | None = None,
    max_iterations: int = 20,
    atol: BlockTolerances = (1e-5, 1e-12, 1e-12),
    rtol: BlockTolerances = (1e-8, 1e-8, 1e-8),
) -> NewtonResult:
    """Solve one target load for all [u, r, p], without changing committed history.

    Snapshot committed state, boundary data, sources, and the initial guess.
    The guess defaults to committed.x. Operators, transfer, and material model
    parameters must remain fixed. sources uses the integrated, absolute target
    convention of evaluate_global_residual(); omitted sources are zero.

    jacobian(x, evaluation) returns the full (4*nc, 4*nc) derivative at the
    current candidate, as a sparse matrix or dense array. It must treat its
    arguments as read-only; x is passed through a read-only view. Closures can
    supply fixed operators/history to the callback. No Jacobian is assumed by
    default. The elastic reference operators.A is exact on an elastic branch.
    coupling.jacobian provides AnalyticalJacobian and FiniteDifferenceJacobian;
    construct either callback from the same committed state as this solve.

    For each block b, convergence requires ||R_b||_2 <= atol_b + rtol_b*||R_b0||_2.
    These are Euclidean norms of the raw integrated residual, not volume-weighted
    field errors. The reference norms are fixed at the initial evaluation.
    atol carries each block's units; the defaults suit this project's SI elastic
    example. rtol is dimensionless and each entry must lie in [0, 1).

    Each correction solves J delta_x = -R and takes the full step. There is no
    damping, line search, load adaptation, or automatic commit. max_iterations
    counts corrections, not residual evaluations; zero allows only an initial
    convergence check. Nonconvergence raises NewtonConvergenceError. Invalid
    Jacobians, linear failures, non-finite updates, and material exceptions also
    propagate without accepting any candidate or modifying input histories.
    """
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 0
    ):
        raise ValueError("max_iterations must be a nonnegative integer.")
    absolute, relative = np.array(atol, dtype=np.float64), np.array(rtol, dtype=np.float64)
    for name, values in (("atol", absolute), ("rtol", relative)):
        if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{name} must contain three finite nonnegative tolerances for [u, r, p].")
    if np.any(relative >= 1):
        raise ValueError("Each rtol must be smaller than one.")

    nc, nf = operators.num_cells, operators.num_faces
    fixed_state = committed.copy()
    x = np.array(fixed_state.x if initial_x is None else initial_x, dtype=np.float64, copy=True)
    if x.shape != (4 * nc,) or not np.all(np.isfinite(x)):
        raise ValueError(f"Expected finite initial_x with shape ({4 * nc},).")
    bf = operators.boundary_faces
    if bc_values.shape != (2, nf) or not np.all(np.isfinite(bc_values[:, bf])):
        raise ValueError(f"Expected bc_values (2, {nf}), finite on boundary faces.")
    boundary = np.zeros((2, nf))
    boundary[:, bf] = bc_values[:, bf]
    load = np.zeros(4 * nc) if sources is None else np.array(sources, dtype=np.float64, copy=True)
    if load.shape != (4 * nc,) or not np.all(np.isfinite(load)):
        raise ValueError(f"Expected finite sources with shape ({4 * nc},).")

    history: list[NDArray[np.float64]] = []
    thresholds = np.zeros(3)
    for iteration in range(max_iterations + 1):
        evaluation = evaluate_global_residual(
            operators, transfer, material_points, fixed_state, x, boundary, sources=load,
        )
        norms = _block_norms(evaluation.residual, nc)
        history.append(norms)
        if iteration == 0:
            thresholds = absolute + relative * norms
            if not np.all(np.isfinite(thresholds)):
                raise ValueError("Newton convergence thresholds must remain finite.")
        if np.all(norms <= thresholds):
            return NewtonResult(evaluation, iteration, np.stack(history), thresholds)
        if iteration == max_iterations:
            break

        callback_x = x.view()
        callback_x.flags.writeable = False
        matrix = sps.csc_array(jacobian(callback_x, evaluation), dtype=np.float64, copy=True)
        matrix.sum_duplicates()
        if matrix.shape != (4 * nc, 4 * nc) or not np.all(np.isfinite(matrix.data)):
            raise ValueError(f"Jacobian must be finite with shape ({4 * nc}, {4 * nc}).")
        try:
            factor = sps.linalg.splu(matrix)
            correction = np.asarray(factor.solve(-evaluation.residual), dtype=np.float64)
        except RuntimeError as error:
            raise RuntimeError(f"Newton linear solve failed at correction {iteration + 1}.") from error
        with np.errstate(over="ignore", invalid="ignore"):
            updated = x + correction
        if correction.shape != x.shape or not np.all(np.isfinite(correction)) or not np.all(np.isfinite(updated)):
            raise RuntimeError(f"Newton produced a non-finite or invalid correction at iteration {iteration + 1}.")
        x = updated

    raise NewtonConvergenceError(
        f"Newton did not converge after {max_iterations} corrections: "
        f"block norms [u, r, p] = {history[-1]}, thresholds = {thresholds}."
    )
