"""Prescribed proportional load steps with atomic acceptance after coupled Newton."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from alternatives.coupled.jacobian import CoupledFiniteDifferenceJacobian
from alternatives.coupled.residual import CoupledOperators
from alternatives.original.jacobian import FiniteDifferenceJacobian
from utils.newton_utils import BlockTolerances, NewtonConvergenceError
from alternatives.newton import JacobianCallback, NewtonResult, solve_newton
from alternatives.original.residual import TpsaOperators
from utils.state import TpsaState
from utils.transfer import CellToFaceTransfer
from material.material_state import MaterialPoint

from utils.load_schedule import (
    LoadStepError as LoadStepError,
    AdaptiveLoadOptions as AdaptiveLoadOptions,
    LoadAttempt as LoadAttempt,
    AdaptiveLoadStepError as AdaptiveLoadStepError,
    _LoadStepSchedule as _LoadStepSchedule,
)

JacobianFactory: TypeAlias = Callable[[TpsaState], JacobianCallback]


@dataclass(eq=False)
class LoadStepResult:
    """One accepted absolute load factor, source vector, and Newton evaluation.

    sources is the integrated target [body force, rotation, pressure] vector.
    newton retains iteration count, residual norms, thresholds, and the complete
    converged state. Returned records own their data independently of controller
    state and the stored history.
    """

    load_factor: float
    sources: NDArray[np.float64]
    newton: NewtonResult

    @property
    def state(self) -> TpsaState:
        """Accepted displacement, auxiliary fields, strain, material history, traction."""
        return self.newton.trial


class LoadStepController(_LoadStepSchedule[LoadStepResult]):
    """Advance a full-Dirichlet TPSA model through prescribed proportional loads.

    Targets are g = factor * reference_bc_values and b = factor * reference_sources;
    sources are absolute cell-integrated values, not increments or densities.
    Factors may increase, decrease, repeat, or be negative. advance/run use fixed
    steps; iter_adaptive/run_adaptive insert substeps with cutback and growth.
    Every mode uses the same fully coupled Newton solve and atomic acceptance.

    Supply a compatible committed state and its initial_load_factor (default 0).
    Nonzero starting states must include their old boundary data and numerical
    tractions, plus stored face displacement for CoupledOperators. Geometry,
    elastic operators, transfer, and material parameters
    remain fixed. State and reference loads are copied; material points supply
    model parameters and their own committed/trial slots are never advanced.

    A fresh Jacobian is built at every step. By default it uses local finite
    differences with finite_difference_step. Alternatively jacobian_factory(old)
    returns a Newton callback, with fixed operators/model data captured in its
    closure. old is an independent copy of the last accepted state.

    Successful Newton completion accepts the ENTIRE trial state together. Failures
    raise LoadStepError without changing state, load_factor, or accepted records.
    state and steps return independent snapshots, so failed/rejected evaluations
    and edits to returned records cannot change the stored committed history.
    """

    def __init__(
        self,
        operators: TpsaOperators,
        transfer: CellToFaceTransfer,
        material_points: Sequence[MaterialPoint],
        committed: TpsaState,
        reference_bc_values: NDArray[np.float64],
        *,
        reference_sources: NDArray[np.float64] | None = None,
        initial_load_factor: float = 0.0,
        jacobian_factory: JacobianFactory | None = None,
        finite_difference_step: float = 1e-10,
        max_iterations: int = 20,
        atol: BlockTolerances = (1e-5, 1e-12, 1e-12),
        rtol: BlockTolerances = (1e-8, 1e-8, 1e-8),
    ) -> None:
        nc, nf = operators.num_cells, operators.num_faces
        if (transfer.num_cells, transfer.num_faces) != (nc, nf):
            raise ValueError("Transfer must match the operators' cell and face counts.")
        if len(material_points) != nc or len(committed.material_states) != nc:
            raise ValueError("Expected one material point and committed history per cell.")
        # Revalidate mutable input and copy all state fields before storing anything.
        state = TpsaState(
            committed.x, committed.bc_values, committed.epsilon,
            committed.material_states, committed.traction, face_displacement=committed.face_displacement,
        )
        if state.bc_values.shape != (2, nf):
            raise ValueError("Committed boundary data must match the operators' face count.")
        if isinstance(operators, CoupledOperators):
            operators.validate_committed(state, material_points)
            operators.validate_transfer(transfer)
        values = np.asarray(reference_bc_values, dtype=np.float64)
        bf = operators.boundary_faces
        if values.shape != (2, nf) or not np.all(np.isfinite(values[:, bf])):
            raise ValueError(f"Expected reference_bc_values (2, {nf}), finite on boundary faces.")
        boundary = np.zeros((2, nf))
        boundary[:, bf] = values[:, bf]
        sources = np.zeros(4 * nc) if reference_sources is None else np.array(
            reference_sources, dtype=np.float64, copy=True,
        )
        if sources.shape != (4 * nc,) or not np.all(np.isfinite(sources)):
            raise ValueError(f"Expected finite reference_sources with shape ({4 * nc},).")
        if not np.isfinite(initial_load_factor):
            raise ValueError("initial_load_factor must be finite.")
        if not np.isfinite(finite_difference_step) or finite_difference_step <= 0:
            raise ValueError("finite_difference_step must be finite and positive.")
        if not np.allclose(state.bc_values[:, bf], initial_load_factor * boundary[:, bf], rtol=1e-10, atol=1e-14):
            raise ValueError("Committed boundary data must match initial_load_factor times reference data.")

        self._operators, self._transfer = operators, transfer
        self._points = tuple(material_points)
        self._committed = state
        self._boundary, self._sources = boundary, sources
        self._load_factor = float(initial_load_factor)
        self._jacobian_factory = jacobian_factory
        self._finite_difference_step = finite_difference_step
        self._max_iterations, self._atol, self._rtol = max_iterations, atol, rtol
        self._steps: list[LoadStepResult] = []
        self._attempts: list[LoadAttempt] = []

    @property
    def state(self) -> TpsaState:
        """Independent snapshot of the last accepted state (initial state before step 1)."""
        return self._committed.copy()

    @property
    def load_factor(self) -> float:
        """Last accepted factor, or initial_load_factor before the first step."""
        return self._load_factor

    @property
    def steps(self) -> tuple[LoadStepResult, ...]:
        """Independent accepted records in input order; the initial state is excluded."""
        return tuple(deepcopy(self._steps))

    @property
    def attempts(self) -> tuple[LoadAttempt, ...]:
        """Accepted and rejected attempts in chronological order, in every mode."""
        return tuple(self._attempts)

    def advance(self, load_factor: float) -> LoadStepResult:
        """Solve and accept one absolute target, starting from the previous solution."""
        factor = float(load_factor)
        if not np.isfinite(factor):
            raise ValueError("load_factor must be finite.")
        try:
            with np.errstate(over="raise", invalid="raise"):
                boundary, sources = factor * self._boundary, factor * self._sources
            old = self._committed.copy()
            default_jacobian = CoupledFiniteDifferenceJacobian if isinstance(self._operators, CoupledOperators) else FiniteDifferenceJacobian
            jacobian = (
                default_jacobian(
                    self._operators, self._transfer, self._points, old,
                    step=self._finite_difference_step,
                )
                if self._jacobian_factory is None else self._jacobian_factory(old.copy())
            )
            result = solve_newton(
                self._operators, self._transfer, self._points, old, boundary, jacobian,
                sources=sources, max_iterations=self._max_iterations,
                atol=self._atol, rtol=self._rtol,
            )
        except Exception as error:
            self._attempts.append(LoadAttempt(self._load_factor, factor, None, str(error)))
            raise LoadStepError(len(self._steps) + 1, factor, error) from error
        record = LoadStepResult(factor, sources, result)
        accepted = result.trial.copy()
        stored = deepcopy(record)
        self._steps.append(stored)
        self._attempts.append(LoadAttempt(self._load_factor, factor, result.iterations))
        self._committed, self._load_factor = accepted, factor
        return record
