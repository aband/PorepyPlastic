"""de Souza Neto, Peric & Owen (2008), Section 7.5.4, printed pp. 252--255.

Small-strain plane strain, weightless perfectly plastic J2 soil, smooth rigid
footing. Coordinates are x in [0,5], y in [-5,0] m; only half the footing is
modelled. Global-component roller conditions reproduce Figure 7.25. The mesh
uses orthogonal graded rectangles, not the book's 135 quadratic FEM elements.
Units: m and kPa, hence integrated face forces are kN/m of out-of-plane length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from porepy.grids.structured import TensorGrid
from porepy.params.bc import BoundaryConditionVectorial

from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import HardeningParameters, isotropic_hardening_K, kinematic_hardening_H
from prototype.loading import LoadStepController, RetainedLoadStepResult
from prototype.newton import BacktrackingOptions
from prototype.operators import RetainedFaceOperators
from prototype.state import RetainedState, assemble_state_residual
from utils.load_schedule import AdaptiveLoadOptions, LoadAttempt

FloatArray = NDArray[np.float64]
WIDTH = 1.0
HALF_WIDTH = WIDTH / 2
DOMAIN_SIZE = 5.0
YOUNG_MODULUS = 1.0e7
POISSON_RATIO = 0.48
YIELD_STRESS = 848.7
COHESION = YIELD_STRESS / np.sqrt(3.0)
LIMIT_FACTOR = 2.0 + np.pi
FINAL_SETTLEMENT = 0.002
RESPONSE_COLUMNS = (
    "step", "load_factor", "settlement_m", "settlement_over_B", "pressure_kPa",
    "pressure_over_c", "half_reaction_kN_per_m", "newton_iterations", "R_u",
    "R_r", "R_p", "R_face", "force_imbalance_kN_per_m", "max_alpha",
)


@dataclass
class StripFootingCase:
    grid: TensorGrid
    bc: BoundaryConditionVectorial
    operators: RetainedFaceOperators
    material_points: list[MaterialPoint]
    bc_values: FloatArray
    footing_faces: NDArray[np.int64]
    x_nodes: FloatArray
    y_nodes: FloatArray
    cells_under_footing: int
    growth: float
    settlement: float
    refinement_level: int = 0


@dataclass
class StripFootingResult:
    case: StripFootingCase
    state: RetainedState
    steps: tuple[RetainedLoadStepResult, ...]
    attempts: tuple[LoadAttempt, ...]
    response: FloatArray
    requested_steps: int
    adaptive: bool


def make_case(
    cells_under_footing: int = 6, *, growth: float = 1.25,
    settlement: float = FINAL_SETTLEMENT,
    refinement_level: int = 0,
) -> StripFootingCase:
    """Grade cells away from x=0.5 m, then optionally bisect every cell axis.

    Each refinement level halves every original interval, including the far
    field. ``cells_under_footing`` specifies the unrefined half-footing mesh.
    """
    if isinstance(cells_under_footing, bool) or not isinstance(cells_under_footing, int) or cells_under_footing < 1:
        raise ValueError("cells_under_footing must be a positive integer.")
    if not np.isfinite(growth) or growth < 1.0:
        raise ValueError("growth must be finite and >= 1.")
    if not np.isfinite(settlement) or settlement <= 0:
        raise ValueError("settlement must be finite and positive.")
    if isinstance(refinement_level, bool) or not isinstance(refinement_level, int) or refinement_level < 0:
        raise ValueError("refinement_level must be a nonnegative integer.")
    h = HALF_WIDTH / cells_under_footing
    length = DOMAIN_SIZE - HALF_WIDTH
    n_outer = (int(np.ceil(length / h)) if growth == 1.0 else
               int(np.ceil(np.log1p(length * (growth - 1.0) / h) / np.log(growth))))
    widths = growth ** np.arange(n_outer, dtype=float)
    widths *= length / widths.sum()
    coordinates = np.r_[np.linspace(0.0, HALF_WIDTH, cells_under_footing + 1),
                        HALF_WIDTH + np.cumsum(widths)]
    coordinates[-1] = DOMAIN_SIZE
    subdivisions = 2**refinement_level
    if refinement_level:
        fractions = np.arange(subdivisions, dtype=float) / subdivisions
        coordinates = np.r_[
            (coordinates[:-1, None] + np.diff(coordinates)[:, None] * fractions).ravel(),
            DOMAIN_SIZE,
        ]
    x_nodes, y_nodes = coordinates, -coordinates[::-1]
    grid = TensorGrid(x_nodes, y_nodes)
    grid.compute_geometry()
    boundary = grid.get_all_boundary_faces()
    x, y = grid.face_centers[:2]
    left = boundary[np.isclose(x[boundary], 0.0)]
    right = boundary[np.isclose(x[boundary], DOMAIN_SIZE)]
    bottom = boundary[np.isclose(y[boundary], -DOMAIN_SIZE)]
    top = boundary[np.isclose(y[boundary], 0.0)]
    footing = top[x[top] < HALF_WIDTH]
    bc = BoundaryConditionVectorial(grid, boundary, "neu")
    # Symmetry and far side: ux=0, ty=0. Bottom: uy=0, tx=0.
    # Smooth rigid footing: uy=-settlement, tx=0; ux remains unknown.
    for component, faces in ((0, np.r_[left, right]), (1, np.r_[bottom, footing])):
        bc.is_dir[component, faces] = True
        bc.is_neu[component, faces] = False
    material = _materialProperty()
    material.young_modulus, material.poisson_ratio = YOUNG_MODULUS, POISSON_RATIO
    points = [MaterialPoint(material, HardeningParameters(YIELD_STRESS, YIELD_STRESS, 0.0, 1.0, 0.0),
                            vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H)
              for _ in range(grid.num_cells)]
    operators = RetainedFaceOperators(grid, bc, mu=material.isotropic_shear_modulus, lam=material.lame_parameter)
    values = np.zeros((2, grid.num_faces))
    values[1, footing] = -settlement
    return StripFootingCase(grid, bc, operators, points, values, footing, x_nodes, y_nodes,
                           cells_under_footing * subdivisions, growth, settlement, refinement_level)


def run_case(
    case: StripFootingCase, *, steps: int = 14, adaptive: bool = True,
    max_iterations: int = 40, progress: bool = False,
) -> StripFootingResult:
    """Apply uniform target settlements; optional cutbacks insert extra substeps.

    The book specifies 14 increments and the endpoint, but not the complete
    increment sequence. This runner uses 14 equally spaced targets by default.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    ops = case.operators
    controller = LoadStepController(
        ops, case.material_points, RetainedState.zeros(ops), case.bc_values,
        max_iterations=max_iterations, atol=(1e-6, 1e-11, 1e-11, 1e-6), rtol=(1e-9,) * 4,
        line_search=BacktrackingOptions(max_backtracks=16),
    )
    targets = np.linspace(0.0, 1.0, steps + 1)[1:]
    increment = float(np.max(np.diff(np.r_[0.0, targets])))
    records = (controller.iter_adaptive(targets, options=AdaptiveLoadOptions(
        initial_increment=increment, max_increment=increment,
        min_increment=min(1e-6, 0.01 / steps), max_retries=15, max_steps=2000,
    )) if adaptive else (controller.advance(float(factor)) for factor in targets))
    rows = [np.zeros(len(RESPONSE_COLUMNS))]
    accepted: list[RetainedLoadStepResult] = []
    boundary_sides = np.isin(ops.side_faces, ops.boundary_faces)
    footing_sides = np.isin(ops.side_faces, case.footing_faces)
    for step, record in enumerate(records, 1):
        state = record.state
        # Downward traction acts on the soil; the reaction on the footing is upward.
        half_reaction = -float(state.side_traction[1, footing_sides].sum())
        pressure = half_reaction / HALF_WIDTH  # 2*R_half / B, NOT R_half / B.
        residual = assemble_state_residual(ops, state)
        norms = [np.linalg.norm(residual[s]) for s in (ops.u_slice, ops.r_slice, ops.p_slice, ops.face_slice)]
        imbalance = float(np.linalg.norm(state.side_traction[:, boundary_sides].sum(axis=1)))
        settlement = record.load_factor * case.settlement
        alpha = max(s.alpha for s in state.tpsa.material_states)
        rows.append(np.array([step, record.load_factor, settlement, settlement / WIDTH, pressure,
                              pressure / COHESION, half_reaction, record.newton.iterations,
                              *norms, imbalance, alpha]))
        accepted.append(record)
        if progress:
            print(f"prototype strip footing: step={step}, u/B={settlement / WIDTH:.7g}, "
                  f"P/c={pressure / COHESION:.6f}, Newton={record.newton.iterations}, "
                  f"force imbalance={imbalance:.2e} kN/m", flush=True)
    return StripFootingResult(case, controller.state, tuple(accepted), controller.attempts,
                             np.stack(rows), steps, adaptive)
