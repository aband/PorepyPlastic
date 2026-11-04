"""Textbook strip footing solved with the coupled cell/strain TPSA alternative.

Geometry, material and reference loads reuse the prototype benchmark definition;
all discretization, residual, Jacobian and load solves use alternatives/.
Boundary data follow PorePy: displacement on Dirichlet components and OUTWARD
integrated force on Neumann components. Units are m, kPa and kN/m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from porepy.utils.common_constants import PARAMETERS

from alternatives.coupled.jacobian import CoupledAnalyticalJacobian
from alternatives.coupled.residual import CoupledOperators, assemble_coupled_residual
from alternatives.loading import LoadStepController, LoadStepResult
from alternatives.plane_strain import (
    KEYWORD, PlaneStrainTpsa, _assemble_matrices, _discretize_get_matrices,
    _set_uniform_parameters,
)
from prototype.strip_footing import (
    COHESION, FINAL_SETTLEMENT, HALF_WIDTH, POISSON_RATIO, WIDTH, YOUNG_MODULUS,
    RESPONSE_COLUMNS as PROTOTYPE_COLUMNS, make_case as prototype_case,
)
from utils.load_schedule import AdaptiveLoadOptions, LoadAttempt
from utils.state import TpsaState

RESPONSE_COLUMNS = tuple("R_strain" if name == "R_face" else name for name in PROTOTYPE_COLUMNS)


class StripFootingCase(PlaneStrainTpsa):
    """Identical benchmark data, with PorePy's condensed TPSA reference matrices."""

    def __init__(self, cells_under_footing: int = 6, *, growth: float = 1.25,
                 settlement: float = FINAL_SETTLEMENT) -> None:
        super().__init__(young_modulus=YOUNG_MODULUS, poisson_ratio=POISSON_RATIO)
        self.cells_under_footing, self.growth, self.settlement = cells_under_footing, growth, settlement
        self.prepare_simulation()

    def prepare_simulation(self) -> None:
        data = prototype_case(self.cells_under_footing, growth=self.growth, settlement=self.settlement)
        self.grid, self.bc, self.bc_values = data.grid, data.bc, data.bc_values
        self.material_points = data.material_points
        self.material = self.material_points[0].material
        self.hardening_parameters = self.material_points[0].parameters
        self.footing_faces = data.footing_faces
        self.x_nodes, self.y_nodes = data.x_nodes, data.y_nodes
        self.bound_vec = self.bc_values.ravel("F")
        self.x = None
        self.d = _set_uniform_parameters(self.grid, mu=self.material.isotropic_shear_modulus,
                                         lmbda=self.material.lame_parameter)
        self.d[PARAMETERS][KEYWORD]["bc"] = self.bc
        self.matrices = _discretize_get_matrices(self.grid, self.d)
        self.face_discretization, self.rhs_matrix, self.div, self.accum = _assemble_matrices(
            self.matrices, self.grid, self.d)
        self.operators = CoupledOperators(self)


@dataclass
class StripFootingResult:
    case: StripFootingCase
    state: TpsaState
    steps: tuple[LoadStepResult, ...]
    attempts: tuple[LoadAttempt, ...]
    response: NDArray[np.float64]
    requested_steps: int
    adaptive: bool


def run_case(case: StripFootingCase, *, steps: int = 14, adaptive: bool = True,
             max_iterations: int = 40, progress: bool = False) -> StripFootingResult:
    """Use the same uniform settlement targets as the prototype run."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    ops = case.operators
    controller = LoadStepController(
        ops, ops.transfer, case.material_points, TpsaState.zeros(case.grid, store_faces=True), case.bc_values,
        jacobian_factory=lambda old: CoupledAnalyticalJacobian(ops, ops.transfer, case.material_points, old),
        max_iterations=max_iterations, atol=(1e-6, 1e-11, 1e-11, 1e-11), rtol=(1e-9,) * 4,
    )
    targets = np.linspace(0.0, 1.0, steps + 1)[1:]
    increment = float(np.max(np.diff(np.r_[0.0, targets])))
    records = (controller.iter_adaptive(targets, options=AdaptiveLoadOptions(
        initial_increment=increment, max_increment=increment,
        min_increment=min(1e-6, 0.01 / steps), max_retries=15, max_steps=2000,
    )) if adaptive else (controller.advance(float(factor)) for factor in targets))
    rows = [np.zeros(len(RESPONSE_COLUMNS))]
    accepted: list[LoadStepResult] = []
    for step, record in enumerate(records, 1):
        state = record.state
        outward = state.traction * ops.boundary_signs
        half_reaction = -float(outward[1, case.footing_faces].sum())
        pressure = half_reaction / HALF_WIDTH
        residual = assemble_coupled_residual(ops, state)
        norms = [np.linalg.norm(b) for b in np.split(residual, np.array([2, 3, 4]) * ops.num_cells)]
        imbalance = float(np.linalg.norm(outward[:, ops.boundary_faces].sum(axis=1)))
        settlement = record.load_factor * case.settlement
        alpha = max(s.alpha for s in state.material_states)
        rows.append(np.array([step, record.load_factor, settlement, settlement / WIDTH, pressure,
                              pressure / COHESION, half_reaction, record.newton.iterations,
                              *norms, imbalance, alpha]))
        accepted.append(record)
        if progress:
            print(f"alternatives coupled strip footing: step={step}, u/B={settlement / WIDTH:.7g}, "
                  f"P/c={pressure / COHESION:.6f}, Newton={record.newton.iterations}, "
                  f"force imbalance={imbalance:.2e} kN/m", flush=True)
    return StripFootingResult(case, controller.state, tuple(accepted), controller.attempts,
                             np.stack(rows), steps, adaptive)
