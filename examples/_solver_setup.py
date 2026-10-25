"""Shared example setup; alternative solvers are imported only when selected."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from prototype.loading import LoadStepController as RetainedController, RetainedLoadStepResult
from prototype.operators import RetainedFaceOperators
from prototype.runner import prepare_retained_solver
from utils.formulation import FormulationName
from utils.newton_utils import JacobianMatrix
from utils.problem import PlaneStrainProblem
from utils.state import TpsaState
from utils.transfer import CellToFaceTransfer

if TYPE_CHECKING:
    from alternatives.coupled.jacobian import CoupledAnalyticalJacobian, CoupledFiniteDifferenceJacobian
    from alternatives.loading import LoadStepController, LoadStepResult
    from alternatives.original.jacobian import AnalyticalJacobian, FiniteDifferenceJacobian
    from alternatives.original.residual import TpsaOperators

    StepRecord: TypeAlias = LoadStepResult | RetainedLoadStepResult
    Controller: TypeAlias = LoadStepController | RetainedController
    Operators: TypeAlias = TpsaOperators | RetainedFaceOperators


@dataclass(eq=False)
class SolverSetup:
    """Prepared solver and tangent evaluation using the preceding committed state."""

    operators: Operators
    transfer: CellToFaceTransfer
    initial: TpsaState
    controller: Controller
    num_dofs: int
    jacobian_at: Callable[[StepRecord, TpsaState], JacobianMatrix]


def problem_type(formulation: FormulationName) -> type[PlaneStrainProblem]:
    """Only the alternative formulations need the elastic TPSA matrix setup."""
    if formulation == "retained":
        return PlaneStrainProblem
    from alternatives.plane_strain import PlaneStrainTpsa

    return PlaneStrainTpsa


def prepare_solver(
    case: PlaneStrainProblem, formulation: FormulationName, jacobian: str, *,
    finite_difference_step: float = 1e-10, max_iterations: int = 20,
) -> SolverSetup:
    """Build the selected operators/controller for an already prepared problem."""
    if jacobian not in ("analytical", "finite-difference"):
        raise ValueError("jacobian must be 'analytical' or 'finite-difference'.")
    if formulation == "retained":
        retained = prepare_retained_solver(
            case, jacobian, finite_difference_step=finite_difference_step,
            max_iterations=max_iterations,
        )
        transfer = retained.operators.rotation_transfer
        assert transfer is not None  # Elastic coefficients were supplied above.

        def retained_tangent(record: StepRecord, old: TpsaState) -> JacobianMatrix:
            if not isinstance(record, RetainedLoadStepResult):
                raise ValueError("Retained setup requires a retained load-step record.")
            return retained.jacobian_factory(old)(record.newton.z, record.newton.evaluation)

        return SolverSetup(
            retained.operators, transfer, retained.initial.tpsa, retained.controller,
            retained.operators.num_dofs, retained_tangent,
        )

    from alternatives.loading import LoadStepController
    from alternatives.newton import JacobianCallback
    from alternatives.plane_strain import PlaneStrainTpsa
    from alternatives.original.residual import TpsaOperators

    if not isinstance(case, PlaneStrainTpsa):
        raise ValueError("Alternative formulations require elastic TPSA matrices.")
    operators: TpsaOperators
    analytical_type: type[AnalyticalJacobian] | type[CoupledAnalyticalJacobian]
    numerical_type: type[FiniteDifferenceJacobian] | type[CoupledFiniteDifferenceJacobian]
    if formulation == "coupled":
        from alternatives.coupled.residual import CoupledOperators
        from alternatives.coupled.jacobian import CoupledAnalyticalJacobian, CoupledFiniteDifferenceJacobian

        coupled = CoupledOperators(case)
        operators, transfer = coupled, coupled.transfer
        analytical_type = CoupledAnalyticalJacobian
        numerical_type = CoupledFiniteDifferenceJacobian
    else:
        from alternatives.original.jacobian import AnalyticalJacobian, FiniteDifferenceJacobian

        operators, transfer = TpsaOperators(case), CellToFaceTransfer(case.grid)
        analytical_type = AnalyticalJacobian
        numerical_type = FiniteDifferenceJacobian
    initial = TpsaState.zeros(case.grid, store_faces=formulation == "coupled")

    def factory(old: TpsaState) -> JacobianCallback:
        if jacobian == "analytical":
            return analytical_type(operators, transfer, case.material_points, old)
        return numerical_type(operators, transfer, case.material_points, old, step=finite_difference_step)

    controller = LoadStepController(
        operators, transfer, case.material_points, initial, case.bc_values,
        jacobian_factory=factory, finite_difference_step=finite_difference_step,
        max_iterations=max_iterations,
    )

    def alternative_tangent(record: StepRecord, old: TpsaState) -> JacobianMatrix:
        if isinstance(record, RetainedLoadStepResult):
            raise ValueError("Alternative setup requires a TPSA load-step record.")
        candidate = coupled.unknowns(record.state) if formulation == "coupled" else record.state.x
        return factory(old)(candidate, record.newton.evaluation)

    return SolverSetup(
        operators, transfer, initial, controller,
        (7 if formulation == "coupled" else 4) * case.grid.num_cells, alternative_tangent,
    )
