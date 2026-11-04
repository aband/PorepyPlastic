"""Isolated cellwise constitutive updates at total plane strain or 3D strain."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from material.J2 import ReturnMapData, ReturnMapConvergenceError
from material.material_state import MaterialPoint, MaterialState 
from material.tensor import strain
from utils.newton_utils import LocalUpdateError

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
    elastic_predictor: NDArray[np.float64] | None = None
    return_map_data: tuple[ReturnMapData, ...] | None = None

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

def _evaluate_material(
    layout: _CellLayout, 
    material_points: Sequence[MaterialPoint],
    committed: _MaterialHistory,
    strain_values: NDArray[np.float64],
    *, 
    input_is_increment: bool,
    dim: int = 2, 
    tol: float = 1.0e-10, 
    rtol: float = 1.0e-12,
) -> MaterialTrial:
    """Evaluate material trials at supplied total strain or only increment."""

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

    #total = checked_strain("epsilon", epsilon)
    #previous = checked_strain("committed.epsilon", committed.epsilon)
    #with np.errstate(over="ignore", invalid="ignore"):
    #    increments = total - previous
    #if not np.all(np.isfinite(increments)):
    #    raise ValueError("Strain increments must remain finite.")

    name = "strain_increment" if input_is_increment else "epsilon"

    values = checked_strain(name, strain_values)
    previous = checked_strain("committed.epsilon", committed.epsilon)

    with np.errstate(over="ignore", invalid="ignore"):
        if input_is_increment:
            increments = values
            total = previous + increments
        else:
            total = values.copy()
            increments = total - previous

    if input_is_increment:
        if not np.all(np.isfinite(total)):
            raise ValueError("Total strain must remain finite.")
    else:
        if not np.all(np.isfinite(increments)):
            raise ValueError("Strain increments must remain finite.")
 
    histories: list[MaterialState] = []
    correction = np.empty((3, 3, nc), dtype=np.float64)
    predictors = np.empty((3, 3, nc), dtype=np.float64)
    datas: list[ReturnMapData] = []

    for cell, point in enumerate(material_points):
        old = committed.material_states[cell]
        increment = strain(increments[:, :, cell])

        #history, predictor_tensor = point.evaluate_from(
        #    old, increment, tol=tol, rtol=rtol
        #)
        try:
            history, predictor_tensor, data = point.evaluate_from_with_data(
                old, increment, tol=tol, rtol=rtol
            )
        except ReturnMapConvergenceError as error:
            raise LocalUpdateError(f"Return map failed in cell {cell}.") from error

        predictor = predictor_tensor.to_numpy(copy=False)
        predictors[:, :, cell] = predictor

        with np.errstate(over="ignore", invalid="ignore"):
            correction[:, :, cell] = (
                predictor - history.stress.to_numpy(copy=False)
            )
 
        histories.append(history)
        datas.append(data)

    if not np.all(np.isfinite(correction)):
        raise ValueError("Stress corrections must remain finite.")

    return MaterialTrial(
        epsilon=total,
        material_states=histories,
        stress_correction=correction,
        elastic_predictor=predictors,
        return_map_data=tuple(datas),
    )

def evaluate_material_at_strain(
    layout: _CellLayout,
    material_points: Sequence[MaterialPoint],
    committed: _MaterialHistory,
    epsilon: NDArray[np.float64],
    *,
    dim: int = 2,
    tol: float = 1.0e-10,
    rtol: float = 1.0e-12,
) -> MaterialTrial:
    """Evaluate at total strain using fixed committed history."""
    return _evaluate_material(
        layout, 
        material_points, 
        committed, 
        epsilon,
        input_is_increment=False,
        dim=dim,
        tol=tol,
        rtol=rtol,
    )

def evaluate_material_at_increment(
    layout: _CellLayout,
    material_points: Sequence[MaterialPoint],
    committed: _MaterialHistory,
    strain_increment: NDArray[np.float64],
    *,
    dim: int=2,
    tol: float=1.0e-10,
    rtol: float=1.0e-12,
) -> MaterialTrial:
    """Evaluate at a strain increment measured from committed history."""
    return _evaluate_material(
        layout,
        material_points,
        committed,
        strain_increment,
        input_is_increment=True,
        dim=dim,
        tol=tol,
        rtol=rtol,
    )