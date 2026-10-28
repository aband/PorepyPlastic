"""One nested Newton load step with independent cell and face increments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from utils.newton_utils import JacobianMatrix, NewtonConvergenceError, _residual_block_norms, _solve_linear_correction, _validate_newton_options
from utils.state import TpsaState
from material.material_state import MaterialPoint
from prototype.jacobian import RetainedAnalyticalJacobian
from prototype.operators import RetainedFaceOperators
from prototype.residual import RetainedFaceResidual, evaluate_global_residual

JacobianCallback: TypeAlias = Callable[[NDArray[np.float64], RetainedFaceResidual], JacobianMatrix]
BlockTolerances: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True)
class BacktrackingOptions:
    """Armijo search: try 1, factor, factor**2, ... for each Newton correction.

    max_backtracks counts reductions AFTER the full-step trial. Material and
    linear-solve errors propagate; only finite residuals are compared for search.
    """

    factor: float = 0.5
    armijo: float = 1e-4
    max_backtracks: int = 12

    def __post_init__(self) -> None:
        if not np.isfinite(self.factor) or not 0 < self.factor < 1:
            raise ValueError("Backtracking factor must lie strictly between zero and one.")
        if not np.isfinite(self.armijo) or not 0 < self.armijo < 0.5:
            raise ValueError("Armijo coefficient must lie strictly between zero and 0.5.")
        if (isinstance(self.max_backtracks, bool)
                or not isinstance(self.max_backtracks, (int, np.integer)) or self.max_backtracks < 0):
            raise ValueError("max_backtracks must be a nonnegative integer.")


@dataclass(eq=False)
class RetainedNewtonResult:
    """A converged load-step candidate; no committed state has been changed.

    evaluation contains z, total trial strain, material histories, Q, and
    one-sided traction INCREMENTS. It does not contain accumulated tractions or
    an accepted global state. Acceptance belongs to the calling load controller.
    residual_norms has columns [u,r,p,face], including iteration zero and each
    accepted Newton iterate. step_lengths excludes iteration zero and rejected
    search trials; residual_evaluations counts every trial, including rejections.
    utils.newton_utils.newton_convergence_rates can process this four-column history.
    """

    evaluation: RetainedFaceResidual
    iterations: int
    residual_norms: NDArray[np.float64]
    thresholds: NDArray[np.float64]
    step_lengths: NDArray[np.float64]
    residual_evaluations: int

    @property
    def z(self) -> NDArray[np.float64]:
        """Converged physical increments [delta_u,delta_r,delta_p,d_free]."""
        return self.evaluation.z


def solve_newton(
    operators: RetainedFaceOperators,
    material_points: Sequence[MaterialPoint],
    committed: TpsaState,
    bc_values: NDArray[np.float64],
    jacobian: JacobianCallback | None = None,
    *,
    initial_z: NDArray[np.float64] | None = None,
    body_force_increment: NDArray[np.float64] | None = None,
    neumann_increment: NDArray[np.float64] | None = None,
    committed_residual: NDArray[np.float64] | None = None,
    max_iterations: int = 20,
    atol: BlockTolerances = (1e-5, 1e-12, 1e-12, 1e-5),
    rtol: BlockTolerances = (1e-8, 1e-8, 1e-8, 1e-8),
    line_search: BacktrackingOptions | None = None,
) -> RetainedNewtonResult:
    """Solve Section 4.2 of Plasticity_revised.pdf at fixed committed history.

    Every residual evaluation reconstructs strain and reruns the existing return
    maps from H_n, including rejected line-search candidates. All four blocks of
    z are solved simultaneously with K delta_z = -R. No auxiliary elimination
    or independent material-iterate correction is performed. initial_z defaults
    to zero LOAD-STEP increments, even when the committed state is nonzero.

    Boundary data is absolute; body_force_increment is integrated (dim,nc) and
    committed_residual is a known old imbalance (num_dofs,). These inputs and
    committed history are copied before iteration. Geometry and model parameters
    must stay fixed. A supplied jacobian(z, evaluation) uses the same history;
    its arguments are read-only by contract. z is passed through a read-only view.
    The default is RetainedAnalyticalJacobian, constructed once when first needed.

    Each block requires ||R_b||_2 <= atol_b + rtol_b*||R_b(initial)||_2, with
    fixed thresholds and four explicit entries [u,r,p,face]. Norms are Euclidean
    norms of integrated residuals, not spatial L2 errors. The defaults use SI
    force/kinematic units. An empty free-face block has norm zero.

    Full steps are the default. Optional backtracking uses the fixed scaled
    merit phi = 0.5*sum_b (||R_b||_2/s_b)**2. Start with
    s_b=max(initial norm,atol_b), then give the two FORCE blocks (u,face) their
    common maximum scale. This prevents an initially balanced face block from
    amplifying finite-difference noise during unloading. Zero scales become one;
    block convergence tolerances remain independent and unchanged.
    Armijo requires phi_new <= (1-2*c*alpha)*phi;
    a candidate satisfying all convergence thresholds is also accepted.

    max_iterations counts Newton corrections; zero only checks the initial
    guess. Iteration/search exhaustion raises NewtonConvergenceError. Invalid
    matrices, non-finite data, and material/linear failures propagate immediately.
    No input histories or MaterialPoint slots are committed, on success or failure.
    """
    absolute, relative = _validate_newton_options(max_iterations, atol, rtol, blocks=4)
    if line_search is not None and not isinstance(line_search, BacktrackingOptions):
        raise ValueError("line_search must be BacktrackingOptions or None.")
    size, nf, nd = operators.num_dofs, operators.num_faces, operators.dim
    old, points = committed.copy(), tuple(material_points)
    z = np.zeros(size) if initial_z is None else np.array(initial_z, dtype=np.float64, copy=True)
    operators._validate_unknowns(z)
    values = np.asarray(bc_values, dtype=np.float64)
    bf = operators.dirichlet_faces
    if values.shape != (nd, nf) or not np.all(np.isfinite(values[:, bf])):
        raise ValueError(f"Expected bc_values ({nd}, {nf}), finite on boundary faces.")
    boundary = np.zeros((nd, nf))
    boundary[:, bf] = values[:, bf]
    body = None if body_force_increment is None else np.array(body_force_increment, dtype=np.float64, copy=True)
    neumann = None if neumann_increment is None else np.array(neumann_increment, dtype=np.float64, copy=True)
    imbalance = None if committed_residual is None else np.array(committed_residual, dtype=np.float64, copy=True)
    blocks = (operators.u_slice, operators.r_slice, operators.p_slice, operators.face_slice)
    evaluations = 0

    def evaluate(candidate: NDArray[np.float64]) -> tuple[RetainedFaceResidual, NDArray[np.float64]]:
        nonlocal evaluations
        result = evaluate_global_residual(
            operators, points, old, candidate, boundary,
            body_force_increment=body, committed_residual=imbalance,
            **({"neumann_increment": neumann} if neumann is not None else {}),
        )
        evaluations += 1
        return result, _residual_block_norms(result.residual, size, blocks)

    evaluation, norms = evaluate(z)
    with np.errstate(over="ignore", invalid="ignore"):
        thresholds = absolute + relative * norms
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("Newton convergence thresholds must remain finite.")
    scales = np.maximum(norms, absolute)
    scales[[0, 3]] = max(scales[0], scales[3])
    scales[scales == 0] = 1.0
    history = [norms]
    lengths: list[float] = []
    callback = jacobian

    def scaled_norm(values: NDArray[np.float64]) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            norm = float(np.linalg.norm(values / scales))
        if not np.isfinite(norm):
            raise RuntimeError("Newton line-search merit must remain finite.")
        return norm

    for iteration in range(max_iterations + 1):
        if np.all(norms <= thresholds):
            return RetainedNewtonResult(
                evaluation, iteration, np.stack(history), thresholds,
                np.array(lengths, dtype=np.float64), evaluations,
            )
        if iteration == max_iterations:
            break
        if callback is None:
            callback = RetainedAnalyticalJacobian(operators, points, old)
        view = z.view()
        view.flags.writeable = False
        correction = _solve_linear_correction(callback(view, evaluation), evaluation.residual, iteration)
        alpha = 1.0
        merit_norm = scaled_norm(norms) if line_search is not None else 0.0
        reductions = 0 if line_search is None else line_search.max_backtracks
        for backtrack in range(reductions + 1):
            with np.errstate(over="ignore", invalid="ignore"):
                candidate = z + alpha * correction
            if not np.all(np.isfinite(candidate)):
                raise RuntimeError(f"Newton produced a non-finite candidate at iteration {iteration + 1}.")
            trial, trial_norms = evaluate(candidate)
            if (line_search is None or np.all(trial_norms <= thresholds)
                    or scaled_norm(trial_norms) <= np.sqrt(1 - 2 * line_search.armijo * alpha) * merit_norm):
                z, evaluation, norms = candidate, trial, trial_norms
                history.append(norms)
                lengths.append(alpha)
                break
            if backtrack == reductions:
                raise NewtonConvergenceError(
                    f"Newton line search failed at correction {iteration + 1} after {reductions} backtracks."
                )
            alpha *= line_search.factor

    raise NewtonConvergenceError(
        f"Newton did not converge after {max_iterations} corrections: "
        f"block norms [u, r, p, face] = {history[-1]}, thresholds = {thresholds}."
    )
