"""OGS Cook's membrane: elastic plane strain on orthogonal staircase grids.

The exact polygon is (0,0), (48,44), (48,60), (0,44) mm. A Cartesian
subgrid approximates its sloped edges; this is NOT the conforming OGS FEM mesh.
The left/right segments remain exactly 44/16 mm long at every refinement.
E=240.565 MPa, nu=0.4999 and t_y=6.25 MPa reproduce OGS 6.5.7's project
parameters in N/mm units. The resultant is 100 N per mm of thickness, or
100 N for the benchmark's unit (1 mm) thickness. All other tractions vanish.

This benchmark is linear elastic. Reuse the existing material-point interface
with an inactive 1e12 MPa yield guard and explicitly reject any plastic update.
The top-right displacement is linearly extrapolated from the last two loaded
face centres to (48,60), not replaced by a tip average or a cell value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from porepy.grids.grid import Grid
from porepy.grids.partition import extract_subgrid
from porepy.grids.structured import CartGrid
from porepy.params.bc import BoundaryConditionVectorial

from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import HardeningParameters, isotropic_hardening_K, kinematic_hardening_H
from prototype.loading import LoadStepController
from prototype.operators import RetainedFaceOperators
from prototype.state import RetainedState, assemble_state_residual

FloatArray = NDArray[np.float64]
SOURCE_URL = "https://www.opengeosys.org/6.5.7/docs/benchmarks/small-deformations/cooksmembranebbar/"
PROJECT_URL = "https://raw.githubusercontent.com/ufz/ogs/6.5.7/Tests/Data/Mechanics/CooksMembrane/CooksMembrane.prj"
GEOMETRY_URL = "https://raw.githubusercontent.com/ufz/ogs/6.5.7/Tests/Data/Mechanics/CooksMembrane/boundary.gml"
POLYGON = np.array([[0., 0.], [48., 44.], [48., 60.], [0., 44.]])
RESPONSE_COLUMNS = ("step", "load_factor", "top_right_uy_mm", "mean_right_uy_mm",
                    "newton_iterations", "R_u", "R_r", "R_p", "R_face", "force_imbalance_N_per_mm")


@dataclass
class CooksMembraneCase:
    grid: Grid
    operators: RetainedFaceOperators
    material_points: list[MaterialPoint]
    bc_values: FloatArray
    reference_neumann: FloatArray
    tip_faces: NDArray[np.int64]
    refinement: int
    spacing: float


@dataclass
class CooksMembraneResult:
    case: CooksMembraneCase
    state: RetainedState
    response: FloatArray


def make_case(refinement: int = 2) -> CooksMembraneCase:
    """Build h=4/2**refinement mm squares whose centres lie in the exact polygon."""
    if isinstance(refinement, bool) or not isinstance(refinement, int) or refinement < 0:
        raise ValueError("refinement must be a nonnegative integer.")
    h = 4. / 2**refinement
    parent = CartGrid(np.array([12, 15]) * 2**refinement, physdims=np.array([48., 60.]))
    parent.compute_geometry()
    x, y = parent.cell_centers[:2]
    keep = np.flatnonzero((y >= (44./48.) * x) & (y <= 44. + (16./48.) * x))
    grid, _, _ = extract_subgrid(parent, keep)
    bf = grid.get_all_boundary_faces()
    fixed = bf[np.isclose(grid.face_centers[0, bf], 0., rtol=0., atol=1e-12)]
    tip = np.asarray(bf[np.isclose(grid.face_centers[0, bf], 48., rtol=0., atol=1e-12)], dtype=np.int64)
    tip = tip[np.argsort(grid.face_centers[1, tip])]
    bc = BoundaryConditionVectorial(grid, fixed, "dir")
    material = _materialProperty()
    material.young_modulus, material.poisson_ratio = 240.565, .4999
    points = [MaterialPoint(material, HardeningParameters(1e12, 1e12, 0., 1., 0.),
                            vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H)
              for _ in range(grid.num_cells)]
    ops = RetainedFaceOperators(grid, bc, mu=material.isotropic_shear_modulus, lam=material.lame_parameter)
    neumann = np.zeros((2, ops.neumann_faces.size))
    at_tip = np.isin(ops.neumann_faces, tip)
    neumann[1, at_tip] = 6.25 * grid.face_areas[ops.neumann_faces[at_tip]]
    return CooksMembraneCase(grid, ops, points, np.zeros((2, grid.num_faces)), neumann,
                             tip, refinement, h)


def top_right_displacement(case: CooksMembraneCase, face_displacement: FloatArray) -> float:
    """First-order extrapolation in y of the retained right-face u_y to y=60 mm."""
    if face_displacement.shape != (2, case.grid.num_faces) or not np.isfinite(face_displacement).all():
        raise ValueError("Expected finite face displacement (2, num_faces).")
    bottom, top = case.tip_faces[-2:]
    y0, y1 = case.grid.face_centers[1, [bottom, top]]
    u0, u1 = face_displacement[1, [bottom, top]]
    return float(u1 + (60. - y1) * (u1 - u0) / (y1 - y0))


def run_case(case: CooksMembraneCase, *, steps: int = 1, progress: bool = False) -> CooksMembraneResult:
    """Ramp the dead traction with the prototype analytical Newton/load controller."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    ops = case.operators
    controller = LoadStepController(ops, case.material_points, RetainedState.zeros(ops),
        case.bc_values, reference_neumann=case.reference_neumann,
        max_iterations=5, atol=(1e-7, 1e-8, 1e-8, 1e-7), rtol=(1e-10,)*4)
    rows = [np.zeros(len(RESPONSE_COLUMNS))]
    boundary_sides = np.isin(ops.side_faces, ops.boundary_faces)
    for step, factor in enumerate(np.linspace(0., 1., steps + 1)[1:], 1):
        record = controller.advance(float(factor))
        state = record.state
        assert state.tpsa.face_displacement is not None
        if any(s.alpha != 0. for s in state.tpsa.material_states):
            raise RuntimeError("The OGS benchmark must remain purely elastic.")
        residual = assemble_state_residual(ops, state)
        norms = [np.linalg.norm(residual[s]) for s in (ops.u_slice, ops.r_slice, ops.p_slice, ops.face_slice)]
        imbalance = float(np.linalg.norm(state.side_traction[:, boundary_sides].sum(axis=1)))
        if imbalance > 1e-5:
            raise RuntimeError(f"Boundary force balance failed: {imbalance:g} N/mm.")
        tip = top_right_displacement(case, state.tpsa.face_displacement)
        mean = float(np.average(state.tpsa.face_displacement[1, case.tip_faces], weights=case.grid.face_areas[case.tip_faces]))
        rows.append(np.array([step, factor, tip, mean, record.newton.iterations, *norms, imbalance]))
        if progress:
            print(f"prototype Cook: h={case.spacing:g} mm, cells={ops.num_cells}, DOFs={ops.num_dofs}, "
                  f"step={step}/{steps}, top-right uy={tip:.6f} mm, Newton={record.newton.iterations}", flush=True)
    return CooksMembraneResult(case, controller.state, np.stack(rows))
