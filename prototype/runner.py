"""Shared retained-face solver setup for the example runners."""

from dataclasses import dataclass

from utils.problem import PlaneStrainProblem
from utils.state import TpsaState
from prototype.jacobian import RetainedAnalyticalJacobian, RetainedFiniteDifferenceJacobian
from prototype.loading import JacobianFactory, LoadStepController
from prototype.newton import JacobianCallback
from prototype.operators import RetainedFaceOperators
from prototype.state import RetainedState


@dataclass(eq=False)
class RetainedSetup:
    """Operators, virgin state, and matching tangent/controller for a prepared case."""

    operators: RetainedFaceOperators
    initial: RetainedState
    jacobian_factory: JacobianFactory
    controller: LoadStepController


def prepare_retained_solver(
    case: PlaneStrainProblem, jacobian: str, *, finite_difference_step: float = 1e-10,
    max_iterations: int = 20,
) -> RetainedSetup:
    """Reuse the existing material points and Dirichlet data without requiring elastic TPSA matrices."""
    if jacobian not in ("analytical", "finite-difference"):
        raise ValueError("jacobian must be 'analytical' or 'finite-difference'.")
    operators = RetainedFaceOperators(
        case.grid, case.bc, mu=case.material.isotropic_shear_modulus,
        lam=case.material.lame_parameter,
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
        jacobian_factory=factory, max_iterations=max_iterations,
    )
    return RetainedSetup(operators, initial, factory, controller)
