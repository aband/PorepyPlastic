"""One fully coupled Newton load step with a caller-supplied Jacobian."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from alternatives.coupled.residual import CoupledOperators, evaluate_coupled_residual
from alternatives.original.residual import TpsaOperators, TpsaResidual, evaluate_global_residual
from utils.state import TpsaState
from utils.transfer import CellToFaceTransfer
from material.material_state import MaterialPoint

from utils.newton_utils import (
    JacobianMatrix as JacobianMatrix,
    BlockTolerances as BlockTolerances,
    NewtonConvergenceError as NewtonConvergenceError,
    newton_convergence_rates as newton_convergence_rates,
    ConvergenceHistory as ConvergenceHistory,
    print_newton_convergence as print_newton_convergence,
    _residual_block_norms as _residual_block_norms,
    _validate_newton_options as _validate_newton_options,
    _solve_linear_correction as _solve_linear_correction,
)

JacobianCallback: TypeAlias = Callable[[NDArray[np.float64], TpsaResidual], JacobianMatrix]


@dataclass(eq=False)
class NewtonResult:
    """Returned only after convergence; accepting trial remains the caller's choice.

    iterations counts Newton corrections, so an equilibrated initial guess uses
    zero. residual_norms includes the initial evaluation. Its columns and the
    fixed thresholds are [u, r, p], with a fourth strain column for CoupledOperators.
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


def _block_norms(residual: NDArray[np.float64], nc: int, *, coupled: bool = False) -> NDArray[np.float64]:
    blocks = [slice(0, 2 * nc), slice(2 * nc, 3 * nc), slice(3 * nc, 4 * nc)]
    if coupled:
        blocks.append(slice(4 * nc, 7 * nc))
    return _residual_block_norms(residual, (7 if coupled else 4) * nc, blocks)


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
    """Solve one target load without changing committed history.

    Snapshot committed state, boundary data, sources, and the initial guess.
    The guess defaults to committed.x, or operators.unknowns(committed) for
    CoupledOperators, which appends independent Mandel strain. Operators and model
    parameters must remain fixed. sources uses the integrated, absolute target
    convention of evaluate_global_residual(); omitted sources are zero.

    jacobian(x, evaluation) returns the full derivative at the current candidate:
    4*nc square for the original formulation, 7*nc square for CoupledOperators.
    It may be sparse or dense and must treat its
    arguments as read-only; x is passed through a read-only view. Closures can
    supply fixed operators/history to the callback. No Jacobian is assumed by
    default. The original elastic reference operators.A is exact on an elastic
    branch; the coupled elastic Jacobian additionally includes the strain constraint.
    alternatives.original.jacobian provides AnalyticalJacobian and FiniteDifferenceJacobian;
    alternatives.coupled.jacobian supplies the corresponding coupled callbacks.
    Construct the callback from the same committed state as this solve.

    For each block b, convergence requires ||R_b||_2 <= atol_b + rtol_b*||R_b0||_2.
    These are Euclidean norms of the raw integrated residual, not volume-weighted
    field errors. The reference norms are fixed at the initial evaluation.
    atol carries each block's units; the defaults suit this project's SI elastic
    example. rtol is dimensionless and each entry must lie in [0, 1). For coupled
    solves, three-entry tolerances append strain defaults atol=1e-12, rtol=1e-8;
    pass four entries to specify strain tolerances explicitly.

    Each correction solves J delta_x = -R and takes the full step. There is no
    damping, line search, load adaptation, or automatic commit. max_iterations
    counts corrections, not residual evaluations; zero allows only an initial
    convergence check. Nonconvergence raises NewtonConvergenceError. Invalid
    Jacobians, linear failures, non-finite updates, and material exceptions also
    propagate without accepting any candidate or modifying input histories.
    """
    coupled = isinstance(operators, CoupledOperators)
    blocks = 4 if coupled else 3
    size = (7 if coupled else 4) * operators.num_cells
    if coupled:
        if len(atol) == 3:
            atol = (*atol, 1e-12)
        if len(rtol) == 3:
            rtol = (*rtol, 1e-8)
    absolute, relative = _validate_newton_options(max_iterations, atol, rtol, blocks=blocks)

    nc, nf = operators.num_cells, operators.num_faces
    fixed_state = committed.copy()
    if isinstance(operators, CoupledOperators):
        operators.validate_committed(fixed_state, material_points)
        operators.validate_transfer(transfer)
        initial = operators.unknowns(fixed_state)
    else:
        initial = fixed_state.x
    x = np.array(initial if initial_x is None else initial_x, dtype=np.float64, copy=True)
    if x.shape != (size,) or not np.all(np.isfinite(x)):
        raise ValueError(f"Expected finite initial_x with shape ({size},).")
    bf = operators.boundary_faces
    if bc_values.shape != (2, nf) or not np.all(np.isfinite(bc_values[:, bf])):
        raise ValueError(f"Expected bc_values (2, {nf}), finite on boundary faces.")
    boundary = np.zeros((2, nf))
    boundary[:, bf] = bc_values[:, bf]
    load = np.zeros(4 * nc) if sources is None else np.array(sources, dtype=np.float64, copy=True)
    if load.shape != (4 * nc,) or not np.all(np.isfinite(load)):
        raise ValueError(f"Expected finite sources with shape ({4 * nc},).")

    history: list[NDArray[np.float64]] = []
    thresholds = np.zeros(blocks)
    for iteration in range(max_iterations + 1):
        if isinstance(operators, CoupledOperators):
            evaluation = evaluate_coupled_residual(
                operators, transfer, material_points, fixed_state, x, boundary, sources=load,
            )
        else:
            evaluation = evaluate_global_residual(
                operators, transfer, material_points, fixed_state, x, boundary, sources=load,
            )
        norms = _block_norms(evaluation.residual, nc, coupled=coupled)
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
        correction = _solve_linear_correction(jacobian(callback_x, evaluation), evaluation.residual, iteration)
        with np.errstate(over="ignore", invalid="ignore"):
            updated = x + correction
        if correction.shape != x.shape or not np.all(np.isfinite(correction)) or not np.all(np.isfinite(updated)):
            raise RuntimeError(f"Newton produced a non-finite or invalid correction at iteration {iteration + 1}.")
        x = updated

    raise NewtonConvergenceError(
        f"Newton did not converge after {max_iterations} corrections: "
        f"block norms {'[u, r, p, strain]' if coupled else '[u, r, p]'} = {history[-1]}, thresholds = {thresholds}."
    )
