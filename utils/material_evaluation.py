"""Isolated cellwise constitutive updates at supplied total plane strain."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from material.material_state import MaterialPoint, MaterialState
from material.tensor import strain
from utils.state import TpsaState


@dataclass(eq=False)
class TpsaMaterialTrial:
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


def evaluate_material_at_strain(
    operators: _CellLayout, material_points: Sequence[MaterialPoint],
    committed: TpsaState, epsilon: NDArray[np.float64],
) -> TpsaMaterialTrial:
    """Isolated local return maps at supplied total plane strain, with fixed history.

    Works with both eliminated-face and retained-face layouts; only num_cells
    is required from operators. The caller supplies the strain reconstruction.
    """
    nc = operators.num_cells
    if len(material_points) != nc or len(committed.material_states) != nc:
        raise ValueError("Expected one material point and history per cell.")
    if (epsilon.shape != (3, 3, nc) or not np.all(np.isfinite(epsilon))
            or not np.allclose(epsilon, epsilon.swapaxes(0, 1), atol=1e-14)
            or not np.allclose(epsilon[2], 0, atol=1e-14) or not np.allclose(epsilon[:, 2], 0, atol=1e-14)):
        raise ValueError("Expected finite symmetric total plane strain with shape (3, 3, nc).")
    histories: list[MaterialState] = []
    correction = np.empty_like(epsilon)
    for cell, point in enumerate(material_points):
        old = committed.material_states[cell]
        increment = strain(epsilon[:, :, cell] - committed.epsilon[:, :, cell])
        # A local point copies history and isolates its mutable trial slot.
        local = MaterialPoint(
            material=point.material, parameters=point.parameters, model=point.model,
            K_law=point.K_law, H_law=point.H_law, committed=old,
        )
        history, _ = local.update(increment)
        elastic_increment = point.material.elastic_tensor.double_contract(increment)
        correction[:, :, cell] = (
            old.stress.to_numpy(copy=False) + elastic_increment.to_numpy(copy=False)
            - history.stress.to_numpy(copy=False)
        )
        histories.append(history)
    return TpsaMaterialTrial(epsilon, histories, correction)
