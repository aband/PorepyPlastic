"""Nagler et al. (2022), section 4.1: spatial 3D cantilever using retained TPSA.

Units are N, mm and MPa. Geometry 100 x 10 x 1 mm, E=210000 MPa,
nu=0.3, Y=240 MPa, H_bar=7000 MPa, theta=0.5. The x=0 face is clamped;
the x=100 face carries uniform z traction; all other faces are traction free.
The 40-step path 0 -> +0.6 -> -0.6 -> 0 MPa follows Figure 3. Reported tip
motion is the area-weighted face average, not a corner or cell displacement.

TPSA uses face-centre displacement unknowns. At least two cells in both width
and thickness are used to constrain all rigid rotations at the clamp. This
is a discretization difference from the paper's higher-order one-cell-wide
FEM meshes. Geometric grading is controlled by largest/smallest x-cell width
ratio, toward the clamp or symmetrically toward both ends. The paper does not
specify its grading ratio; grading at both ends is an additional mesh experiment.
All material laws are reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from porepy.grids.structured import TensorGrid
from porepy.params.bc import BoundaryConditionVectorial

from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import HardeningParameters, isotropic_hardening_K, kinematic_hardening_H
from prototype.jacobian import RetainedAnalyticalJacobian, RetainedFiniteDifferenceJacobian
from prototype.loading import LoadStepController
from prototype.newton import BacktrackingOptions
from prototype.operators import RetainedFaceOperators
from prototype.state import RetainedState
from utils.state import TpsaState

FloatArray = NDArray[np.float64]
Grading = Literal["clamp", "both-ends"]


@dataclass
class CantileverCase:
    grid: TensorGrid
    operators: RetainedFaceOperators
    material_points: list[MaterialPoint]
    boundary: FloatArray
    reference_neumann: FloatArray
    tip_faces: NDArray[np.int64]
    shape: tuple[int, int, int]
    grading_ratio: float
    axes: tuple[FloatArray, FloatArray, FloatArray]
    grading: Grading = "clamp"


@dataclass
class CantileverResult:
    case: CantileverCase
    response: FloatArray
    state_A: RetainedState
    state_B: RetainedState
    final_state: RetainedState
    jacobian: str


RESPONSE_COLUMNS = (
    "step", "load_factor", "traction_mpa", "mean_tip_uz_mm", "max_alpha",
    "plastic_cell_fraction", "newton_iterations", "force_imbalance_N",
)


def load_factors(steps: int = 40) -> FloatArray:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 4 or steps % 4:
        raise ValueError("steps must be a positive multiple of four.")
    q = steps // 4
    return np.concatenate((np.linspace(0., 1., q + 1)[1:],
                           np.linspace(1., -1., 2 * q + 1)[1:],
                           np.linspace(-1., 0., q + 1)[1:]))


def make_case(nx: int = 10, ny: int = 2, nz: int = 6, *, grading_ratio: float = 8.,
              grading: Grading = "clamp") -> CantileverCase:
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in (nx, ny, nz)):
        raise ValueError("Use at least two cells per axis to resolve the 3D clamp with face-centre DOFs.")
    if not np.isfinite(grading_ratio) or grading_ratio < 1.:
        raise ValueError("grading_ratio must be finite and at least one.")
    if grading not in ("clamp", "both-ends"):
        raise ValueError("grading must be 'clamp' or 'both-ends'.")
    if grading == "both-ends":
        if nx == 2 and grading_ratio > 1.:
            raise ValueError("Nonuniform both-ends grading requires at least three x cells.")
        half = np.geomspace(1., grading_ratio, (nx + 1) // 2)
        # Odd nx has one central cell; even nx has two equal central cells.
        widths = np.concatenate((half, half[-2::-1] if nx % 2 else half[::-1]))
    else:
        widths = np.geomspace(1., grading_ratio, nx)
    x = np.r_[0., 100. * np.cumsum(widths) / widths.sum()]
    y, z = np.linspace(-5., 5., ny + 1), np.linspace(-.5, .5, nz + 1)
    grid = TensorGrid(x, y, z)
    grid.compute_geometry()
    bf = grid.get_all_boundary_faces()
    fixed = bf[np.isclose(grid.face_centers[0, bf], 0., atol=1e-12)]
    tip = np.asarray(bf[np.isclose(grid.face_centers[0, bf], 100., atol=1e-12)], dtype=np.int64)
    bc = BoundaryConditionVectorial(grid, fixed, "dir")
    material = _materialProperty()
    material.young_modulus, material.poisson_ratio = 210000., .3
    parameters = HardeningParameters(240., 240., 7000., .5, 0.)
    points = [MaterialPoint(material, parameters, vonMisesModel("J2"),
                           isotropic_hardening_K, kinematic_hardening_H)
              for _ in range(grid.num_cells)]
    ops = RetainedFaceOperators(grid, bc, mu=material.isotropic_shear_modulus, lam=material.lame_parameter)
    neumann = np.zeros((3, ops.neumann_faces.size))
    at_tip = np.isin(ops.neumann_faces, tip)
    neumann[2, at_tip] = .6 * grid.face_areas[ops.neumann_faces[at_tip]]
    return CantileverCase(grid, ops, points, np.zeros((3, grid.num_faces)), neumann,
                          tip, (nx, ny, nz), grading_ratio, (x, y, z), grading=grading)


def mean_tip_displacement(case: CantileverCase, state: RetainedState) -> float:
    faces = state.tpsa.face_displacement
    if faces is None:
        raise ValueError("The accepted state must retain face displacements.")
    return float(np.average(faces[2, case.tip_faces], weights=case.grid.face_areas[case.tip_faces]))


def run_case(case: CantileverCase, *, steps: int = 40, jacobian: str = "analytical",
             progress: bool = False) -> CantileverResult:
    if jacobian not in ("analytical", "finite-difference"):
        raise ValueError("Unknown Jacobian choice.")
    ops = case.operators
    initial = RetainedState.zeros(ops)

    def factory(old: TpsaState) -> RetainedAnalyticalJacobian:
        if jacobian == "finite-difference":
            return RetainedFiniteDifferenceJacobian(ops, case.material_points, old, step=1e-8)
        return RetainedAnalyticalJacobian(ops, case.material_points, old)

    controller = LoadStepController(
        ops, case.material_points, initial, case.boundary, reference_neumann=case.reference_neumann,
        jacobian_factory=factory, max_iterations=35, line_search=BacktrackingOptions(max_backtracks=16),
        # Force tolerances exceed cancellation noise for the slender beam in N/mm units.
        atol=(2e-6, 2e-7, 2e-7, 2e-6), rtol=(1e-9, 1e-9, 1e-9, 1e-9),
    )
    rows = [np.zeros(len(RESPONSE_COLUMNS))]
    state_A, state_B = initial, initial
    boundary_sides = np.isin(ops.side_faces, ops.boundary_faces)
    for step, factor in enumerate(load_factors(steps), 1):
        record = controller.advance(float(factor))
        state = record.state
        alpha = np.array([p.alpha for p in state.tpsa.material_states])
        balance = float(np.linalg.norm(np.sum(state.side_traction[:, boundary_sides], axis=1)))
        if balance > 2e-5:
            raise RuntimeError(f"Boundary force balance failed: {balance:g} N.")
        rows.append(np.array([step, factor, .6 * factor, mean_tip_displacement(case, state),
                              np.max(alpha), np.mean(alpha > 1e-10), record.newton.iterations, balance]))
        if np.isclose(factor, 1.):
            state_A = state.copy()
        elif np.isclose(factor, -1.):
            state_B = state.copy()
        if progress:
            print(f"{case.shape}, grading={case.grading}, ratio={case.grading_ratio:g}: step {step:02d}/{steps}, "
                  f"t_z={.6*factor:+.3f} MPa, mean u_z={rows[-1][3]:+.5f} mm, "
                  f"Newton={record.newton.iterations}", flush=True)
    return CantileverResult(case, np.stack(rows), state_A, state_B, controller.state, jacobian)
