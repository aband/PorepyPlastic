"""Isolated cellwise constitutive updates at total plane strain or 3D strain."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from material.material_state import MaterialPoint, MaterialState
from material.tensor import strain


@dataclass(eq=False)
class MaterialTrial:
    """Cell results from one trial evaluation, with independent arrays and histories.

    epsilon and stress_correction have shape (3, 3, nc). The correction is the
    elastic predictor minus returned stress for the current load increment.
    Numerical face tractions and force equilibrium are evaluated separately.
    """

    epsilon: NDArray[np.float64]
    material_states: list[MaterialState]
    stress_correction: NDArray[np.float64]


class _CellLayout(Protocol):
    """The supplied-strain material update only needs the number of cells."""

    @property
    def num_cells(self) -> int: ...


class _MaterialHistory(Protocol):
    """History shared by the legacy TPSA state and dimension-aware material state."""

    @property
    def epsilon(self) -> NDArray[np.float64]: ...

    @property
    def material_states(self) -> Sequence[MaterialState]: ...


def evaluate_material_at_strain(
    layout: _CellLayout, material_points: Sequence[MaterialPoint],
    committed: _MaterialHistory, epsilon: NDArray[np.float64],
    *, dim: int = 2, tol: float = 1.0e-10, rtol: float = 1.0e-12,
) -> MaterialTrial:
    """Evaluate material trials at supplied TOTAL strain, using fixed history.

    Only num_cells is required from layout; the caller supplies reconstructed
    strain. dim=2 imposes plane strain, while dim=3 allows general symmetric strain.
    Material stresses and plastic histories remain full 3D tensors in either case.

    Each return map receives epsilon - committed.epsilon and starts from the
    supplied committed material state. Original material-point history and trial
    slots are untouched, including after a local failure. Returned fields own their
    storage. No commit or tangent computation is performed here.

    Q = sigma_n + C_e:(epsilon - epsilon_n) - sigma_returned. tol and rtol control
    absolute and relative local return-map convergence. Material parameters and
    committed MaterialState fields should be validated during problem setup;
    reference elastic coefficients must agree with the global discretization.
    """
    if dim not in (2, 3):
        raise ValueError("Expected dimension 2 or 3.")
    # The new material history carries dim; the legacy TPSA state does not.
    if getattr(committed, "dim", dim) != dim:
        raise ValueError("Committed history has a different dimension.")

    nc = layout.num_cells
    if nc < 1 or len(material_points) != nc or len(committed.material_states) != nc:
        raise ValueError("Expected one material point and history per cell.")

    def checked_strain(name: str, value: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(value)
        if values.dtype.kind not in ("f", "i", "u"):
            raise ValueError(f"{name} must contain real numeric values.")
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (3, 3, nc) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite with shape (3, 3, {nc}).")
        if not np.allclose(values, values.swapaxes(0, 1), rtol=1e-10, atol=1e-14):
            raise ValueError(f"{name} must be symmetric.")
        if dim == 2 and (
            not np.allclose(values[2], 0.0, atol=1e-14)
            or not np.allclose(values[:, 2], 0.0, atol=1e-14)
        ):
            raise ValueError(f"{name} must satisfy plane strain.")
        return values

    total = checked_strain("epsilon", epsilon)
    previous = checked_strain("committed.epsilon", committed.epsilon)
    with np.errstate(over="ignore", invalid="ignore"):
        increments = total - previous
    if not np.all(np.isfinite(increments)):
        raise ValueError("Strain increments must remain finite.")

    histories: list[MaterialState] = []
    correction = np.empty((3, 3, nc), dtype=np.float64)
    for cell, point in enumerate(material_points):
        old = committed.material_states[cell]
        increment = strain(increments[:, :, cell])
        # A local point copies history and isolates its mutable trial slot.
        local = MaterialPoint(
            material=point.material, parameters=point.parameters, model=point.model,
            K_law=point.K_law, H_law=point.H_law, committed=old,
        )
        history, _ = local.update(increment, tol=tol, rtol=rtol)
        elastic_increment = point.material.elastic_tensor.double_contract(increment)
        with np.errstate(over="ignore", invalid="ignore"):
            correction[:, :, cell] = (
                old.stress.to_numpy(copy=False) + elastic_increment.to_numpy(copy=False)
                - history.stress.to_numpy(copy=False)
            )
        histories.append(history)
    if not np.all(np.isfinite(correction)):
        raise ValueError("Stress corrections must remain finite.")
    return MaterialTrial(total.copy(), histories, correction)
