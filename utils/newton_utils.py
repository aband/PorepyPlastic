"""Shared Newton diagnostics, tolerance validation, and linear corrections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, TypeAlias

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray

JacobianMatrix: TypeAlias = sps.spmatrix | sps.sparray | NDArray[np.float64]
BlockTolerances: TypeAlias = tuple[float, float, float] | tuple[float, float, float, float]


class NewtonConvergenceError(RuntimeError):
    """The last evaluated candidate did not meet all block residual tolerances."""


def newton_convergence_rates(
    residual_norms: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return reduction ratios and observed orders for each residual block.

    Input and outputs have shape (evaluations, 3 or 4), ordered [u,r,p,(extra)].
    The optional block may be coupled strain or retained-face balance.
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
    if (norms.ndim != 2 or norms.shape[1] not in (3, 4) or norms.shape[0] < 1
            or not np.all(np.isfinite(norms)) or np.any(norms < 0)):
        raise ValueError("Expected finite nonnegative residual norms with shape (evaluations, 3) or (evaluations, 4).")
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


class ConvergenceHistory(Protocol):
    @property
    def residual_norms(self) -> NDArray[np.float64]: ...


def print_newton_convergence(
    result: ConvergenceHistory, *, indent: str = "    ",
    extra_block: Literal["eps", "f"] = "eps",
) -> None:
    """Print one load step's iteration history and nonlinear block convergence rates.

    All block norms are shown separately. Rates describe momentum unless
    extra_block='f', which selects the nonlinear retained-face balance instead.
    Initial evaluation is iteration 0; -- denotes an unavailable order or ratio.
    """
    ratios, orders = newton_convergence_rates(result.residual_norms)
    rate_index, rate_label = (3, "f") if extra_block == "f" else (0, "u")
    if rate_index >= result.residual_norms.shape[1]:
        raise ValueError("Face convergence rates require four residual blocks.")
    strain_header = f"{f'||R_{extra_block}||':>14}" if result.residual_norms.shape[1] == 4 else ""
    print(f"{indent}iter       ||R_u||       ||R_r||       ||R_p||{strain_header}      ratio_{rate_label}   order_{rate_label}")
    for iteration, norms in enumerate(result.residual_norms):
        ru, rr, rp = norms[:3]
        ratio, order = ratios[iteration, rate_index], orders[iteration, rate_index]
        ratio_text = f"{ratio:.3e}" if np.isfinite(ratio) else "--"
        order_text = f"{order:.3f}" if np.isfinite(order) else "--"
        strain_text = f" {norms[3]:13.3e}" if norms.size == 4 else ""
        print(f"{indent}{iteration:4d} {ru:13.3e} {rr:13.3e} {rp:13.3e}{strain_text} {ratio_text:>12} {order_text:>9}")


def _residual_block_norms(
    residual: NDArray[np.float64], size: int, blocks: Sequence[slice],
) -> NDArray[np.float64]:
    """Validate and measure integrated residual blocks without mixing their units."""
    if residual.shape != (size,) or not np.all(np.isfinite(residual)):
        raise RuntimeError("Newton evaluation returned a non-finite or incorrectly shaped residual.")
    norms = np.array([np.linalg.norm(residual[block]) for block in blocks])
    if not np.all(np.isfinite(norms)):
        raise RuntimeError("Newton block residual norms are not finite.")
    return norms


def _validate_newton_options(
    max_iterations: int, atol: Sequence[float], rtol: Sequence[float], *, blocks: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shared iteration-limit and block-tolerance validation."""
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer))
            or max_iterations < 0):
        raise ValueError("max_iterations must be a nonnegative integer.")
    absolute, relative = np.array(atol, dtype=np.float64), np.array(rtol, dtype=np.float64)
    for name, values in (("atol", absolute), ("rtol", relative)):
        if values.shape != (blocks,) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{name} must contain {blocks} finite nonnegative tolerances for the residual blocks.")
    if np.any(relative >= 1):
        raise ValueError("Each rtol must be smaller than one.")
    return absolute, relative


def _solve_linear_correction(
    jacobian: JacobianMatrix, residual: NDArray[np.float64], iteration: int,
) -> NDArray[np.float64]:
    """Solve the full Newton system, preserving the callback's matrix."""
    size = residual.size
    matrix = sps.csc_array(jacobian, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"Jacobian must be finite with shape ({size}, {size}).")
    try:
        correction = np.asarray(sps.linalg.splu(matrix).solve(-residual), dtype=np.float64)
    except RuntimeError as error:
        raise RuntimeError(f"Newton linear solve failed at correction {iteration + 1}.") from error
    if correction.shape != residual.shape or not np.all(np.isfinite(correction)):
        raise RuntimeError(f"Newton produced a non-finite or invalid correction at iteration {iteration + 1}.")
    return correction
