"""Shared retained-face solver setup for the example runners."""

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from porepy.grids.grid import Grid
from porepy.params.bc import BoundaryConditionVectorial

from material.material_state import MaterialPoint

from utils.state import TpsaState
from prototype.jacobian import RetainedAnalyticalJacobian, RetainedFiniteDifferenceJacobian
from prototype.loading import JacobianFactory, LoadStepController
from prototype.newton import BacktrackingOptions, JacobianCallback
from prototype.operators import RetainedFaceOperators
from prototype.state import RetainedState


class PreparedRetainedProblem(Protocol):
    """Minimal prepared 2D/3D problem; no shared plane-strain runner required."""

    @property
    def grid(self) -> Grid: ...

    @property
    def bc(self) -> BoundaryConditionVectorial: ...

    @property
    def material_points(self) -> Sequence[MaterialPoint]: ...

    @property
    def bc_values(self) -> NDArray[np.float64]: ...


@dataclass(eq=False)
class RetainedSetup:
    """Operators, virgin state, and matching tangent/controller for a prepared case."""

    operators: RetainedFaceOperators
    initial: RetainedState
    jacobian_factory: JacobianFactory
    controller: LoadStepController


def prepare_retained_solver(
    case: PreparedRetainedProblem, jacobian: str, *, finite_difference_step: float = 1e-10,
    max_iterations: int = 20,
    line_search: BacktrackingOptions | None = None,
) -> RetainedSetup:
    """Reuse the existing material points and Dirichlet data without requiring elastic TPSA matrices."""
    if jacobian not in ("analytical", "finite-difference"):
        raise ValueError("jacobian must be 'analytical' or 'finite-difference'.")
    operators = RetainedFaceOperators(
        case.grid, case.bc,
        mu=np.array([p.material.isotropic_shear_modulus for p in case.material_points]),
        lam=np.array([p.material.lame_parameter for p in case.material_points]),
    )
    initial = RetainedState.zeros(operators)

    def factory(old: TpsaState) -> JacobianCallback:
        if jacobian == "analytical":
            return RetainedAnalyticalJacobian(operators, case.material_points, old)
        return RetainedFiniteDifferenceJacobian(
            operators, case.material_points, old, step=finite_difference_step,
        )

    controller = LoadStepController(
        operators, case.material_points, initial, case.bc_values,
        jacobian_factory=factory, max_iterations=max_iterations, line_search=line_search,
    )
    return RetainedSetup(operators, initial, factory, controller)
