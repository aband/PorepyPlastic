"""Galin/Yarushina circular-hole benchmark on Cartesian staircase meshes.

Source: Yarushina et al. (2010), doi:10.1029/2010GC003130, Eqs. (10),
(15)--(18), (22), (24), (41). Compression is positive P, tensile stress positive.
R=k=1, p=0; E/k=100. The prototype's full 3D J2 law uses sigma_y=sqrt(3)*k,
zero hardening and nu close to 0.5. Galin is an exact J2 reference only in the
incompressible limit: finite nu is an explicitly retained modelling error.

Outer displacement comes from Eq. (17), removing remote-boundary truncation
error for the circular reference domain. The actual hole is a staircase with
zero traction on its horizontal/vertical faces. No radial normals are imposed
on those faces. Displacements inside the plastic region are NOT supplied as an
analytical reference (Eqs. (13)--(14) apply to tensile loading only).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from porepy.grids.grid import Grid
from porepy.grids.partition import extract_subgrid
from porepy.grids.structured import CartGrid
from porepy.params.bc import BoundaryConditionVectorial

from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import HardeningParameters, isotropic_hardening_K, kinematic_hardening_H
from prototype.newton import BacktrackingOptions, solve_newton
from prototype.operators import RetainedFaceOperators
from prototype.state import RetainedState, assemble_state_residual
from prototype.tpsa_state import TpsaState

FloatArray = NDArray[np.float64]
SOURCE = "https://doi.org/10.1029/2010GC003130"
RESPONSE_COLUMNS = ("step", "pressure_over_k", "tau_infinity_over_k", "newton_iterations",
                    "R_u", "R_r", "R_p", "R_face", "mean_stress_relative_L2",
                    "max_shear_relative_L2", "plastic_area", "elapsed_seconds")


def load_path(pressure: float) -> float:
    """Hydrostatic up to P/k=1, then tau/k=-0.3*(P/k-1), ending at (2,-0.3)."""
    if not np.isfinite(pressure) or not 0. <= pressure <= 2.:
        raise ValueError("pressure must lie in [0, 2] for this benchmark load path.")
    return -.3 * max(pressure - 1., 0.)


def plastic_axes(pressure: float) -> tuple[float, float]:
    """Actual x/y semiaxes, not major/minor ordering; zero before first yield."""
    tau = load_path(pressure)
    if pressure < 1.:
        return 0., 0.
    c = np.exp((pressure - 1.) / 2.)
    return float(c*(1.-tau)), float(c*(1.+tau))


def _elastic_potentials(z: NDArray[np.complex128], pressure: float
                        ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Return Phi, (bar(z) Phi_z + Psi), phi and psi outside the plastic zone.

For P<=1 this path is axisymmetric: use Lamé/Kirsch with sigma_rr(R)=0
and far-field stress=-P I, rather than the inconsistent pressure signs printed
in the purely elastic paragraph on p.3 of the paper. For P>=1, Eq.(16) matches
that limit continuously. Choose the exterior inverse-map root in ALL quadrants.
"""
    tau = load_path(pressure)
    if pressure <= 1.:
        phi = -.5*pressure*z
        psi = pressure/z
        return np.full_like(z, -.5*pressure), -pressure/z**2, phi, psi
    c = np.exp((pressure - 1.)/2.)
    discriminant = np.sqrt((z/c)**2 + 4.*tau)
    first, second = .5*(z/c+discriminant), .5*(z/c-discriminant)
    s = np.where(abs(first) >= abs(second), first, second)
    w_prime = c*(1.+tau/s**2)
    w_tilde = c*(1./s-tau*s)
    logarithm = np.log(z/s)
    big_phi = -.5-logarithm
    phi_prime = -(w_prime/z-1./s)
    big_psi = -w_tilde/(s*w_prime)
    q = np.conj(z)*phi_prime/w_prime+big_psi
    phi = -z*(logarithm+.5)-2.*c*tau/s
    psi = c*(1./s+tau*s)
    return big_phi, q, phi, psi


def analytical_stress(xy: FloatArray, pressure: float) -> FloatArray:
    """In-plane stress (2,2,n) at reference material points r>=R=1."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite coordinates (2, n).")
    r = np.linalg.norm(xy, axis=0)
    if np.any(r < 1.-1e-12):
        raise ValueError("Analytical stresses require r>=R=1.")
    ax, ay = plastic_axes(pressure)
    plastic = ((xy[0]/ax)**2+(xy[1]/ay)**2 <= 1.) if ax else np.zeros(r.size, dtype=bool)
    result = np.zeros((2, 2, r.size))
    elastic = ~plastic
    if np.any(elastic):
        z = np.asarray(xy[0, elastic]+1j*xy[1, elastic], dtype=np.complex128)
        phi, q, _, _ = _elastic_potentials(z, pressure)
        result[0, 0, elastic] = 2.*phi.real-q.real
        result[1, 1, elastic] = 2.*phi.real+q.real
        result[0, 1, elastic] = result[1, 0, elastic] = q.imag
    if np.any(plastic):
        n = xy[:, plastic]/r[plastic]
        rr = -2.*np.log(r[plastic])
        result[:, :, plastic] = (rr-2.)[None, None, :]*np.eye(2)[:, :, None]+2.*np.einsum("in,jn->ijn", n, n)
    return result


def analytical_elastic_displacement(xy: FloatArray, pressure: float, *, young: float = 100., nu: float = .4999) -> FloatArray:
    """Eq.(17) outside the plastic ellipse; not a plastic displacement solution."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite coordinates (2, n).")
    if not np.isfinite(young) or young <= 0. or not 0. <= nu <= .5:
        raise ValueError("Expected positive E and 0<=nu<=0.5.")
    ax, ay = plastic_axes(pressure)
    if np.any(np.linalg.norm(xy, axis=0) < 1.-1e-12) or (ax and np.any((xy[0]/ax)**2+(xy[1]/ay)**2 < 1.-1e-12)):
        raise ValueError("Displacement reference is available only in the elastic region.")
    z = np.asarray(xy[0]+1j*xy[1], dtype=np.complex128)
    big_phi, _, phi, psi = _elastic_potentials(z, pressure)
    u = ((3.-4.*nu)*phi-z*np.conj(big_phi)-np.conj(psi))/(young/(1.+nu))
    return np.stack((u.real, u.imag))


def stress_measures(stress: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Paper's 2D mean and maximum shear; neither is the 3D mean/von Mises stress."""
    return .5*(stress[0, 0]+stress[1, 1]), np.sqrt(.25*(stress[0, 0]-stress[1, 1])**2+stress[0, 1]**2)


@dataclass
class CircularHoleCase:
    grid: Grid
    operators: RetainedFaceOperators
    material_points: list[MaterialPoint]
    level: int
    spacing: float
    half_width: float
    nu: float


@dataclass
class CircularHoleResult:
    case: CircularHoleCase
    state: RetainedState
    response: FloatArray
    stress: FloatArray
    reference_stress: FloatArray
    pressure: float


def make_case(level: int = 1, *, half_width: float = 3., nu: float = .4999) -> CircularHoleCase:
    """Keep square-cell centres outside r=1 in [-L,L]^2; all outer faces Dirichlet."""
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("level must be a nonnegative integer.")
    if not np.isfinite(half_width) or half_width < 2.5 or not 0. < nu < .5:
        raise ValueError("Expected half_width>=2.5 and 0<nu<0.5.")
    h = .25/2**level
    n = round(2.*half_width/h)
    if not np.isclose(n*h, 2.*half_width):
        raise ValueError("Domain width must be a multiple of cell spacing.")
    parent = CartGrid(np.array([n, n]), physdims=np.array([2.*half_width, 2.*half_width]))
    parent.nodes[:2] -= half_width
    parent.compute_geometry()
    grid, _, _ = extract_subgrid(parent, np.flatnonzero(np.linalg.norm(parent.cell_centers[:2], axis=0) >= 1.))
    bf = grid.get_all_boundary_faces()
    outer = bf[np.any(np.isclose(abs(grid.face_centers[:2, bf]), half_width), axis=0)]
    bc = BoundaryConditionVectorial(grid, outer, "dir")
    material = _materialProperty()
    material.young_modulus, material.poisson_ratio = 100., nu
    hardening = HardeningParameters(np.sqrt(3.), np.sqrt(3.), 0., 1., 0.)
    points = [MaterialPoint(material, hardening, vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H)
              for _ in range(grid.num_cells)]
    ops = RetainedFaceOperators(grid, bc, mu=material.isotropic_shear_modulus, lam=material.lame_parameter)
    return CircularHoleCase(grid, ops, points, level, h, half_width, nu)


def relative_l2(actual: FloatArray, exact: FloatArray, volumes: FloatArray) -> float:
    denominator = float(np.sum(volumes*exact**2))
    return float(np.sqrt(np.sum(volumes*(actual-exact)**2)/denominator)) if denominator else 0.


def run_case(case: CircularHoleCase, *, steps: int = 20, final_pressure: float = 2., progress: bool = False) -> CircularHoleResult:
    """Reuse prototype Newton and state containers for this nonproportional BC path.

No hidden hardening, finite-difference Jacobian, or artificial body force is used.
The accepted imbalance is included at each increment; new state is installed
only after convergence and validation. Failure propagates, without export of a
supposedly converged result. Targets include P=1 exactly when the path crosses it.
"""
    load_path(final_pressure)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or final_pressure <= 0.:
        raise ValueError("Expected positive steps and final pressure.")
    ops, g = case.operators, case.grid
    state = RetainedState.zeros(ops)
    targets = np.linspace(0., final_pressure, steps+1)[1:]
    if final_pressure > 1.:
        targets = np.unique(np.r_[targets, 1.])
    rows = [np.zeros(len(RESPONSE_COLUMNS))]
    started = perf_counter()
    for step, pressure in enumerate(targets, 1):
        boundary = np.zeros((2, ops.num_faces))
        boundary[:, ops.dirichlet_faces] = analytical_elastic_displacement(
            g.face_centers[:2, ops.dirichlet_faces], float(pressure), nu=case.nu)
        result = solve_newton(ops, case.material_points, state.tpsa, boundary,
            committed_residual=assemble_state_residual(ops, state),
            max_iterations=35, atol=(1e-8, 1e-10, 1e-10, 1e-8), rtol=(1e-9,)*4,
            line_search=BacktrackingOptions(max_backtracks=15))
        trial = result.evaluation.material_trial
        assert state.tpsa.face_displacement is not None
        faces = state.tpsa.face_displacement+ops.face_displacement_increment(result.z, boundary, state.tpsa.bc_values)
        faces[:, ops.dirichlet_faces] = boundary[:, ops.dirichlet_faces]
        tpsa = TpsaState(state.tpsa.x+result.z[ops.cell_slice], boundary, trial.epsilon, trial.material_states,
                         state.tpsa.traction, face_displacement=faces)
        candidate = RetainedState(ops, tpsa, state.side_traction+result.evaluation.traction_increment, state.body_force)
        residual = assemble_state_residual(ops, candidate)
        norms = np.array([np.linalg.norm(residual[s]) for s in (ops.u_slice, ops.r_slice, ops.p_slice, ops.face_slice)])
        if np.any(norms > result.thresholds*1.01+1e-12):
            raise RuntimeError("Accumulated state failed the Newton equilibrium thresholds.")
        state = candidate
        stress = np.stack([s.stress.to_numpy() for s in state.tpsa.material_states], axis=2)
        reference = analytical_stress(g.cell_centers[:2], float(pressure))
        mean, shear = stress_measures(stress)
        exact_mean, exact_shear = stress_measures(reference)
        errors = [relative_l2(mean, exact_mean, g.cell_volumes), relative_l2(shear, exact_shear, g.cell_volumes)]
        plastic = np.array([s.alpha > 1e-8 for s in state.tpsa.material_states])
        rows.append(np.array([step, pressure, load_path(float(pressure)), result.iterations, *norms, *errors,
                              g.cell_volumes[plastic].sum(), perf_counter()-started]))
        if progress:
            print(f"prototype circle: h={case.spacing:g}, cells={g.num_cells}, P/k={pressure:.3f}, "
                  f"tau/k={load_path(float(pressure)):.3f}, Newton={result.iterations}, "
                  f"L2 mean/shear={errors[0]:.3%}/{errors[1]:.3%}", flush=True)
    return CircularHoleResult(case, state, np.stack(rows), stress, reference, final_pressure)
