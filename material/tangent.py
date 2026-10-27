"""Analytical and finite-difference tangents of isolated material-point updates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from material.material_state import MaterialPoint, MaterialState
from material.tensor import strain, stress


def _validate_dimension(dim: int) -> None:
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim not in (2, 3):
        raise ValueError("Tangent dimension must be 2 or 3.")


def _validate_step(step: float) -> None:
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Finite-difference step must be finite and positive.")


def finite_difference_material_tangent(
    point: MaterialPoint,
    committed: MaterialState,
    strain_increment: NDArray[np.float64],
    *,
    dim: int = 2,
    step: float = 1e-10,
) -> NDArray[np.float64]:
    """Return d sigma[i,j] / d G[k,l], shape (3,3,dim,dim), by forward differences.

    dim=2 varies the in-plane displacement gradient; dim=3 varies all components.
    Perturbing G[k,l] by step changes strain by step * sym(e_k outer e_l), so
    shear perturbations change BOTH symmetric strain entries by step/2.
    Strain increments, stress output and histories remain full 3D in either case.
    dim selects derivative directions; it does not restrict the supplied strain
    increment or perform a plane-stress reduction.

    Each of 1 + dim**2 local updates (baseline plus perturbations) starts from
    the supplied committed history. point supplies only model/parameters; its
    committed/trial slots and all inputs remain unchanged, including on failure.
    step is an absolute dimensionless gradient perturbation, defaulting to the
    paper's 1e-10. At a yield switch this is a one-sided numerical derivative.
    """
    _validate_dimension(dim)
    _validate_step(step)
    increment = np.array(strain_increment, dtype=np.float64, copy=True)
    if increment.shape != (3, 3) or not np.all(np.isfinite(increment)):
        raise ValueError("Expected a finite strain increment with shape (3, 3).")
    local = MaterialPoint(
        material=point.material, parameters=point.parameters, model=point.model,
        K_law=point.K_law, H_law=point.H_law, committed=committed,
    )

    def stress_at(values: NDArray[np.float64]) -> NDArray[np.float64]:
        history, _ = local.update(strain(values))
        stress = np.asarray(history.stress.to_numpy(), dtype=np.float64)
        if stress.shape != (3, 3) or not np.all(np.isfinite(stress)):
            raise ValueError("Material update returned a non-finite or invalid stress.")
        return stress

    baseline = stress_at(increment)
    tangent = np.empty((3, 3, dim, dim))
    for k in range(dim):
        for l in range(dim):
            basis = np.zeros((3, 3))
            basis[k, l] += 0.5
            basis[l, k] += 0.5
            perturbed = increment + step * basis
            if not np.all(np.isfinite(perturbed)) or np.array_equal(perturbed, increment):
                raise ValueError("Finite-difference step cannot produce a finite, distinct perturbation.")
            tangent[:, :, k, l] = (stress_at(perturbed) - baseline) / step
    if not np.all(np.isfinite(tangent)):
        raise ValueError("Finite-difference material tangent must be finite.")
    return tangent


def analytical_material_tangent(
    point: MaterialPoint,
    committed: MaterialState,
    strain_increment: NDArray[np.float64],
    *,
    dim: int = 2,
    returned: MaterialState | None = None,
) -> NDArray[np.float64]:
    """Return exact J2 d sigma[i,j]/d G[k,l], with shape (3,3,dim,dim).

    Use the same 3D radial return and scalar hardening laws as MaterialPoint.
    dim=2 selects in-plane gradient variations; dim=3 selects all components.
    The supplied strain increment and material histories remain full 3D;
    dim only selects derivative directions, without a plane-stress reduction.
    Optionally reuse returned from that SAME increment and committed history;
    otherwise perform one isolated local update. No finite differences are used.
    The supplied point's committed/trial slots and all input states are preserved.
    Both shear columns act on sym(delta_G), with no engineering-shear factor.
    Out-of-plane stress rows are retained. At a yield switch use the branch
    selected by the return map, rather than claiming a unique smooth derivative.
    """
    _validate_dimension(dim)
    increment = np.asarray(strain_increment, dtype=np.float64)
    if (increment.shape != (3, 3) or not np.all(np.isfinite(increment))
            or not np.allclose(increment, increment.T, rtol=1e-10, atol=1e-14)):
        raise ValueError("Expected a finite symmetric strain increment with shape (3, 3).")
    if returned is None:
        local = MaterialPoint(
            point.material, point.parameters, point.model, point.K_law, point.H_law,
            committed=committed,
        )
        returned, _ = local.update(strain(increment))
    # Recover the multiplier in THIS implementation's unit-flow convention.
    dgamma = (returned.alpha - committed.alpha) / np.sqrt(2 / 3)
    elastic = point.material.elastic_tensor.to_numpy(copy=False)
    predictor = committed.stress.to_numpy() + np.einsum("ijkl,kl->ij", elastic, increment)
    tangent = point.model.consistent_tangent(
        stress(predictor), committed.backstress, returned.alpha, float(dgamma),
        point.material, point.K_law, point.H_law, point.parameters,
    )
    return tangent.to_numpy()[:, :, :dim, :dim]
