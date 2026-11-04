"""Cellwise-vectorized J2 radial return and consistent-tangent correction.

Array counterparts of vonMisesModel.radial_return_map_with_data and
vonMisesModel.consistent_tangent for many independent material points.
Tensors have shape (3, 3, nc) and scalars shape (nc,). Every cell runs the
same local Newton iteration, tolerances and branch selection as the
per-cell model; only the loop over cells is vectorized.

Hardening laws must accept array alpha and parameters whose fields are
scalars or (nc,) arrays, as the laws in material.scalar_hardening do.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from material.J2 import ReturnMapConvergenceError
from material.scalar_hardening import HardeningParameters, ScalarHardeningLaw

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SQRT_TWO_THIRDS = np.sqrt(2.0 / 3.0)


class BatchReturnMapConvergenceError(ReturnMapConvergenceError):
    """Local Newton failed in the listed cells."""

    def __init__(self, cells: IntArray, residuals: FloatArray) -> None:
        self.cells = cells
        super().__init__(
            f"Local Newton iteration did not converge in {cells.size} cell(s) "
            f"(first: {cells[0]}); max |R|={np.max(np.abs(residuals)):.3e}."
        )


@dataclass(frozen=True, eq=False)
class ReturnMapBatch:
    """Updated history and intermediate quantities for one cellwise trial."""

    stress: FloatArray             # (3, 3, nc)
    plastic_strain: FloatArray     # (3, 3, nc)
    backstress: FloatArray         # (3, 3, nc)
    alpha: FloatArray              # (nc,)
    elastic_predictor: FloatArray  # (3, 3, nc)
    dgamma: FloatArray             # (nc,), exactly zero on the elastic branch
    trial_norm: FloatArray         # (nc,), ||dev(predictor) - dev(backstress_n)||
    flow_direction: FloatArray     # (3, 3, nc), zero on the elastic branch


def _deviatoric(tensors: FloatArray) -> FloatArray:
    mean = (tensors[0, 0] + tensors[1, 1] + tensors[2, 2]) / 3.0
    result = tensors.copy()
    for i in range(3):
        result[i, i] -= mean
    return result


def _parameters_at(
    parameters: HardeningParameters, cells: IntArray,
) -> HardeningParameters:
    """Restrict array-valued parameter fields to the given cells."""
    values = {}
    for item in fields(parameters):
        value = getattr(parameters, item.name)
        values[item.name] = value[cells] if np.ndim(value) else value
    return HardeningParameters(**values)


def radial_return_batch(
    stress_n: FloatArray,
    plastic_strain_n: FloatArray,
    backstress_n: FloatArray,
    alpha_n: FloatArray,
    strain_increment: FloatArray,
    *,
    mu: float | FloatArray,
    lam: float | FloatArray,
    parameters: HardeningParameters,
    K_law: ScalarHardeningLaw,
    H_law: ScalarHardeningLaw,
    tol: float = 1.0e-10,
    rtol: float = 1.0e-12,
    max_iteration: int = 20,
) -> ReturnMapBatch:
    """Return-map every cell from its committed history and strain increment.

    Inputs are not modified. tol and rtol match vonMisesModel.consistency_parameter:
    |R| <= tol + rtol * max(|q|, |sqrt(2/3) K(alpha_n)|) per cell.
    """
    if (
        not np.isfinite(tol) or tol < 0.0
        or not np.isfinite(rtol) or rtol < 0.0
    ):
        raise ValueError("Convergence tolerances must be finite and non-negative.")

    shape = alpha_n.shape
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), shape)
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), shape)
    c = SQRT_TWO_THIRDS

    # Isotropic elastic predictor, C_e : delta_epsilon for symmetric strain.
    trace = strain_increment[0, 0] + strain_increment[1, 1] + strain_increment[2, 2]
    predictor = stress_n + 2.0 * mu * strain_increment
    for i in range(3):
        predictor[i, i] += lam * trace

    shifted = _deviatoric(predictor) - _deviatoric(backstress_n)
    trial_norm = np.sqrt(np.einsum("ijc,ijc->c", shifted, shifted))

    K_n = np.broadcast_to(K_law(alpha_n, parameters)[0], shape)
    H_n = np.broadcast_to(H_law(alpha_n, parameters)[0], shape)
    residual_tolerance = tol + rtol * np.maximum(np.abs(trial_norm), np.abs(c * K_n))

    # Local Newton on the cells whose trial state violates the yield condition.
    dgamma = np.zeros(shape, dtype=np.float64)
    active = np.flatnonzero(trial_norm - c * K_n > 0.0)

    for iteration in range(max_iteration + 1):
        if active.size == 0:
            break

        local = _parameters_at(parameters, active)
        current = dgamma[active]
        alpha = alpha_n[active] + c * current

        K, K_prime = K_law(alpha, local)
        H, H_prime = H_law(alpha, local)

        residual = (
            trial_norm[active]
            - 2.0 * mu[active] * current
            - c * (K + H - H_n[active])
        )
        converged = np.isfinite(residual) & (
            np.abs(residual) <= residual_tolerance[active]
        )

        # The last pass only checks the final update, as in the per-cell model.
        if iteration == max_iteration:
            if not np.all(converged):
                raise BatchReturnMapConvergenceError(
                    active[~converged], residual[~converged]
                )
            break

        pending = ~converged
        slope = np.broadcast_to(K_prime + H_prime, residual.shape)[pending]
        active = active[pending]
        derivative = -2.0 * mu[active] - (2.0 / 3.0) * slope

        dgamma[active] = np.maximum(
            0.0, current[pending] - residual[pending] / derivative
        )

    plastic = dgamma > 0.0
    direction = np.zeros_like(shifted)
    direction[:, :, plastic] = shifted[:, :, plastic] / trial_norm[plastic]

    alpha = alpha_n + c * dgamma
    H = np.broadcast_to(H_law(alpha, parameters)[0], shape)

    return ReturnMapBatch(
        stress=predictor - (2.0 * mu * dgamma) * direction,
        plastic_strain=plastic_strain_n + dgamma * direction,
        backstress=backstress_n + c * (H - H_n) * direction,
        alpha=alpha,
        elastic_predictor=predictor,
        dgamma=dgamma,
        trial_norm=trial_norm,
        flow_direction=direction,
    )


def tangent_correction_batch(
    result: ReturnMapBatch,
    *,
    mu: float | FloatArray,
    parameters: HardeningParameters,
    K_law: ScalarHardeningLaw,
    H_law: ScalarHardeningLaw,
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Return plastic cells and the coefficients of C_e - C_alg on them.

    C_e - C_alg = c_dev * P_dev + c_nn * (n outer n), with
    c_dev = 4 mu^2 dgamma/q and c_nn = 4 mu^2 (1/D - dgamma/q),
    D = 2 mu + (2/3)(K'(alpha) + H'(alpha)) at the updated alpha.
    Elastic cells (dgamma == 0) contribute no correction.
    """
    cells = np.flatnonzero(result.dgamma > 0.0)
    if cells.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return cells, empty, empty

    shear = np.broadcast_to(np.asarray(mu, dtype=np.float64), result.dgamma.shape)[cells]
    q = result.trial_norm[cells]
    if not np.all(np.isfinite(q)) or np.any(q <= 0.0):
        raise ValueError("Plastic tangent requires a nonzero shifted trial stress.")

    local = _parameters_at(parameters, cells)
    alpha = result.alpha[cells]
    _, K_prime = K_law(alpha, local)
    _, H_prime = H_law(alpha, local)
    denominator = 2.0 * shear + (2.0 / 3.0) * (K_prime + H_prime)
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError(
            "Plastic tangent requires a finite positive consistency denominator."
        )

    ratio = result.dgamma[cells] / q
    scale = 4.0 * shear**2
    return cells, scale * ratio, scale * (1.0 / denominator - ratio)
