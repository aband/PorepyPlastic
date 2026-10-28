"""Load stepping and atomic acceptance for the retained-face nested solver."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from utils.load_schedule import LoadAttempt, LoadStepError, _LoadStepSchedule
from utils.newton_utils import NewtonConvergenceError, _residual_block_norms, _validate_newton_options
from utils.state import TpsaState
from prototype.tpsa_state import TpsaState as SpatialTpsaState
from material.material_state import MaterialPoint
from prototype.newton import (
    BacktrackingOptions, BlockTolerances, JacobianCallback, RetainedNewtonResult, solve_newton,
)
from prototype.operators import RetainedFaceOperators
from prototype.state import RetainedState, assemble_state_residual

JacobianFactory: TypeAlias = Callable[[TpsaState], JacobianCallback]


@dataclass(eq=False)
class RetainedLoadStepResult:
    """One accepted absolute factor, accumulated state, and incremental Newton result.

    state and newton own independent arrays/history. Edits to returned records
    do not affect the controller or its stored records.
    """

    load_factor: float
    state: RetainedState
    newton: RetainedNewtonResult


class LoadStepController(_LoadStepSchedule[RetainedLoadStepResult]):
    """Proportional Dirichlet, Neumann and body loads with fixed or adaptive stepping.

    Targets are g=factor*reference_bc_values and b=factor*reference_body_force.
    Newton receives absolute g but INCREMENTAL b_target-b_n.
    Neumann reference values are outward integrated forces (dim, n_neumann),
    supplied in operators.neumann_faces order. They are differenced against
    the accepted state at each attempt. Newton also receives the
    actual old imbalance assembled from stored total histories. A fresh callback
    uses H_n at every attempt, including adaptive retries. The default is the
    analytical retained-face Jacobian; jacobian_factory(old_tpsa) can override it.

    On convergence, accumulate u,r,p, shared face displacement, and every side
    traction once. Take total strain and material histories from the solved
    return maps; assign exact target boundary values and body load. Validate the
    complete state before replacing any accepted data. Failures preserve state,
    load factor, and records, and append only a failed LoadAttempt.

    State and loads are copied. MaterialPoint slots remain untouched. Reuse
    utils.load_schedule.AdaptiveLoadOptions with inherited run/iter_adaptive/
    run_adaptive; only NewtonConvergenceError triggers adaptive cutback.
    Geometry, side ordering, elasticity, and material parameters must remain fixed.
    """

    def __init__(
        self, operators: RetainedFaceOperators, material_points: Sequence[MaterialPoint],
        committed: RetainedState, reference_bc_values: NDArray[np.float64],
        *, reference_body_force: NDArray[np.float64] | None = None,
        reference_neumann: NDArray[np.float64] | None = None,
        initial_load_factor: float = 0.0,
        jacobian_factory: JacobianFactory | None = None,
        max_iterations: int = 20,
        atol: BlockTolerances = (1e-5, 1e-12, 1e-12, 1e-5),
        rtol: BlockTolerances = (1e-8, 1e-8, 1e-8, 1e-8),
        line_search: BacktrackingOptions | None = None,
    ) -> None:
        nc, nf, nd = operators.num_cells, operators.num_faces, operators.dim
        state = RetainedState(operators, committed.tpsa, committed.side_traction, committed.body_force,
                              neumann_values=committed.neumann_values)
        if operators.mu is None or operators.lam is None:
            raise ValueError("Retained load stepping requires mu and lam.")
        if len(material_points) != nc:
            raise ValueError("Expected one material point per cell.")
        for cell, point in enumerate(material_points):
            if not np.allclose(
                [point.material.isotropic_shear_modulus, point.material.lame_parameter],
                [operators.mu[cell], operators.lam[cell]], rtol=1e-12, atol=0,
            ):
                raise ValueError("Material elastic coefficients must match the retained operators.")
        _validate_newton_options(max_iterations, atol, rtol, blocks=4)
        if line_search is not None and not isinstance(line_search, BacktrackingOptions):
            raise ValueError("line_search must be BacktrackingOptions or None.")
        values = np.asarray(reference_bc_values, dtype=np.float64)
        bf = operators.dirichlet_faces
        if values.shape != (nd, nf) or not np.all(np.isfinite(values[:, bf])):
            raise ValueError(f"Expected reference_bc_values ({nd}, {nf}), finite on boundary faces.")
        boundary = np.zeros((nd, nf))
        boundary[:, bf] = values[:, bf]
        body = np.zeros((nd, nc)) if reference_body_force is None else np.array(
            reference_body_force, dtype=np.float64, copy=True,
        )
        if body.shape != (nd, nc) or not np.all(np.isfinite(body)):
            raise ValueError(f"Expected finite reference_body_force with shape ({nd}, {nc}).")
        neumann = (np.zeros((nd, operators.neumann_faces.size)) if reference_neumann is None
                   else np.array(reference_neumann, dtype=np.float64, copy=True))
        if neumann.shape != (nd, operators.neumann_faces.size) or not np.all(np.isfinite(neumann)):
            raise ValueError("Expected finite reference_neumann (dim, num_neumann_faces).")
        if not np.isfinite(initial_load_factor):
            raise ValueError("initial_load_factor must be finite.")
        with np.errstate(over="raise", invalid="raise"):
            initial_bc, initial_body = initial_load_factor * boundary, initial_load_factor * body
        if not np.allclose(state.tpsa.bc_values, initial_bc, rtol=1e-10, atol=1e-14):
            raise ValueError("Committed boundary data must match initial_load_factor times reference data.")
        if not np.allclose(state.body_force, initial_body, rtol=1e-10, atol=1e-14):
            raise ValueError("Committed body force must match initial_load_factor times reference data.")
        if not np.allclose(state.neumann_values, initial_load_factor * neumann, rtol=1e-10, atol=1e-14):
            raise ValueError("Committed Neumann data must match initial_load_factor times reference data.")
        self._neumann = neumann
        self._operators, self._points = operators, tuple(material_points)
        self._committed, self._boundary, self._body = state, boundary, body
        self._load_factor = float(initial_load_factor)
        self._jacobian_factory = jacobian_factory
        self._max_iterations, self._atol, self._rtol = max_iterations, deepcopy(atol), deepcopy(rtol)
        self._line_search = line_search
        self._steps: list[RetainedLoadStepResult] = []
        self._attempts: list[LoadAttempt] = []

    @property
    def state(self) -> RetainedState:
        """Independent snapshot of the last accepted total state."""
        return self._committed.copy()

    @property
    def load_factor(self) -> float:
        return self._load_factor

    @property
    def steps(self) -> tuple[RetainedLoadStepResult, ...]:
        return tuple(deepcopy(self._steps))

    @property
    def attempts(self) -> tuple[LoadAttempt, ...]:
        return tuple(self._attempts)

    def advance(self, load_factor: float) -> RetainedLoadStepResult:
        """Solve one absolute target and accept all fields together, or preserve H_n."""
        factor = float(load_factor)
        if not np.isfinite(factor):
            raise ValueError("load_factor must be finite.")
        ops = self._operators
        try:
            old = self._committed.copy()
            with np.errstate(over="raise", invalid="raise"):
                boundary, body = factor * self._boundary, factor * self._body
                delta_body = body - old.body_force
                neumann = factor * self._neumann
                delta_neumann = neumann - old.neumann_values
            imbalance = assemble_state_residual(ops, old)
            callback = None if self._jacobian_factory is None else self._jacobian_factory(old.tpsa.copy())
            result = solve_newton(
                ops, self._points, old.tpsa, boundary, callback,
                body_force_increment=delta_body, committed_residual=imbalance,
                max_iterations=self._max_iterations, atol=self._atol, rtol=self._rtol,
                line_search=self._line_search,
                **({"neumann_increment": delta_neumann} if ops.neumann_faces.size else {}),
            )
            evaluation = result.evaluation
            assert old.tpsa.face_displacement is not None
            with np.errstate(over="raise", invalid="raise"):
                x = old.tpsa.x + result.z[ops.cell_slice]
                faces = old.tpsa.face_displacement + ops.face_displacement_increment(
                    result.z, boundary, old.tpsa.bc_values,
                )
                side = old.side_traction + evaluation.traction_increment
            faces[:, ops.dirichlet_faces] = boundary[:, ops.dirichlet_faces]
            material = evaluation.material_trial
            tpsa = SpatialTpsaState(x, boundary, material.epsilon, material.material_states,
                             old.tpsa.traction, face_displacement=faces)
            accepted = RetainedState(ops, tpsa, side, body, neumann_values=neumann)
            norms = _residual_block_norms(assemble_state_residual(ops, accepted), ops.num_dofs,
                (ops.u_slice, ops.r_slice, ops.p_slice, ops.face_slice))
            if np.any(norms > result.thresholds):
                raise NewtonConvergenceError("Accumulated state exceeds Newton block residual thresholds.")
            record = RetainedLoadStepResult(factor, accepted.copy(), result)
            stored = deepcopy(record)
        except Exception as error:
            self._attempts.append(LoadAttempt(self._load_factor, factor, None, str(error)))
            raise LoadStepError(len(self._steps) + 1, factor, error) from error
        self._steps.append(stored)
        self._attempts.append(LoadAttempt(self._load_factor, factor, result.iterations))
        self._committed, self._load_factor = accepted, factor
        return record
