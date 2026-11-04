"""Elastic bending verification for the retained TPSA prototype (N, mm, MPa).

The large studies solve PorePy's cell-only elastic Schur complement, recover
the retained face unknowns, and check the original prototype residual. This
avoids building a large retained Newton matrix for a linear problem. Tests
compare this path with the full retained matrix and the prototype load solver.

Pure bending has an exact three-dimensional polynomial solution. The paper
cantilever uses beam theory only as an engineering stress reference; it is not
an exact 3D solution near the clamp, loaded end, or lateral edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import porepy as pp
import scipy.sparse as sps
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import splu

from prototype.cantilever import Grading, make_case
from prototype.operators import RetainedFaceOperators
from prototype.residual import assemble_global_residual, evaluate_one_sided_traction
from utils.newton_utils import _solve_linear_correction

FloatArray = NDArray[np.float64]
Shape = tuple[int, int, int]


@dataclass
class ElasticBendingCase:
    kind: Literal["pure_bending", "cantilever"]
    operators: RetainedFaceOperators
    bc: pp.BoundaryConditionVectorial
    boundary: FloatArray
    neumann: FloatArray
    axes: tuple[FloatArray, FloatArray, FloatArray]
    shape: Shape
    young: float = 210000.
    poisson: float = .3
    yield_stress: float = 240.
    curvature: float = 0.
    traction: float = 0.

    @property
    def dimensions(self) -> FloatArray:
        return np.array([axis[-1] - axis[0] for axis in self.axes])


@dataclass
class ElasticBendingResult:
    case: ElasticBendingCase
    x: FloatArray
    face_displacement: FloatArray
    strain: FloatArray
    stress: FloatArray
    side_traction: FloatArray
    reference_stress: FloatArray
    metrics: dict[str, float]
    face_sections: FloatArray
    cell_sections: FloatArray


def pure_bending_fields(
    coordinates: FloatArray, curvature: float, young: float = 210000., poisson: float = .3,
    *, variances: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Exact u and sigma, with optional rectangular averaging of quadratic u.

    u = (-k*x*z, nu*k*y*z, k/2 * (x^2 + nu*(z^2-y^2))).
    epsilon = diag(-k*z, nu*k*z, nu*k*z); sigma_xx = -E*k*z.
    Thus div(sigma)=0 and every shear stress is zero. variances are squared
    rectangle side lengths / 12; mixed-coordinate covariances vanish.
    """
    x, y, z = coordinates
    u = curvature * np.vstack((-x*z, poisson*y*z, .5*(x*x + poisson*(z*z-y*y))))
    if variances is not None:
        u[2] += .5 * curvature * (variances[0] + poisson*(variances[2]-variances[1]))
    stress = np.zeros((3, 3, x.size))
    stress[0, 0] = -young * curvature * z
    return u, stress


def _face_variances(grid: pp.TensorGrid) -> FloatArray:
    """Exact coordinate variances on axis-aligned quadrilateral faces."""
    nodes = grid.nodes[:, grid.face_nodes.indices].reshape((3, 4, grid.num_faces), order="F")
    return np.ptp(nodes, axis=1)**2 / 12.


def make_pure_bending_case(
    shape: Shape = (4, 2, 2), *, dimensions: tuple[float, float, float] = (4., 2., 1.),
    curvature: float = 1e-4,
) -> ElasticBendingCase:
    """Exact bending on a block: exact displacement left, exact traction elsewhere.

    The left displacement includes Poisson warping; this is deliberately a
    manufactured verification problem, not the fully clamped paper cantilever.
    Both Dirichlet face averages and integrated Neumann loads are exact.
    """
    if len(shape) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in shape):
        raise ValueError("shape must contain three integer cell counts of at least two.")
    if len(dimensions) != 3 or not np.all(np.isfinite(dimensions)) or min(dimensions) <= 0:
        raise ValueError("dimensions must contain three finite positive lengths.")
    if not np.isfinite(curvature) or curvature <= 0:
        raise ValueError("curvature must be finite and positive.")
    length, width, height = dimensions
    axes = (np.linspace(0., length, shape[0]+1),
            np.linspace(-width/2, width/2, shape[1]+1),
            np.linspace(-height/2, height/2, shape[2]+1))
    grid = pp.TensorGrid(*axes)
    grid.compute_geometry()
    boundary_faces = grid.get_all_boundary_faces()
    fixed = boundary_faces[np.isclose(grid.face_centers[0, boundary_faces], 0.)]
    bc = pp.BoundaryConditionVectorial(grid, fixed, "dir")
    young, nu = 210000., .3
    ops = RetainedFaceOperators(grid, bc, mu=young/(2*(1+nu)), lam=young*nu/((1+nu)*(1-2*nu)))
    displacement, stress = pure_bending_fields(grid.face_centers, curvature, young, nu,
                                              variances=_face_variances(grid))
    sides = np.searchsorted(ops.side_faces, ops.neumann_faces)
    neumann = np.einsum("ijf,jf->if", stress[:, :, ops.neumann_faces], ops.side_normals[:, sides])
    neumann *= grid.face_areas[ops.neumann_faces]
    boundary = np.zeros_like(displacement)
    boundary[:, fixed] = displacement[:, fixed]
    return ElasticBendingCase("pure_bending", ops, bc, boundary, neumann, axes, shape, curvature=curvature)


def make_elastic_cantilever(
    shape: Shape = (128, 2, 4), *, traction: float = .06,
    grading_ratio: float = 1., grading: Grading = "clamp",
) -> ElasticBendingCase:
    """Reuse the paper geometry and BCs at a small positive tip traction."""
    if not np.isfinite(traction) or traction <= 0:
        raise ValueError("traction must be finite and positive.")
    original = make_case(*shape, grading_ratio=grading_ratio, grading=grading)
    grid, ops = original.grid, original.operators
    bc = pp.BoundaryConditionVectorial(grid, ops.dirichlet_faces, "dir")
    return ElasticBendingCase("cantilever", ops, bc, original.boundary,
        original.reference_neumann * (traction/.6), original.axes, shape, traction=traction)


def reference_stress(case: ElasticBendingCase, coordinates: FloatArray) -> FloatArray:
    """Exact manufactured stress, or Saint-Venant beam approximation for the cantilever."""
    if case.kind == "pure_bending":
        return pure_bending_fields(coordinates, case.curvature, case.young, case.poisson)[1]
    length, width, height = case.dimensions
    force, inertia = case.traction*width*height, width*height**3/12.
    x, _, z = coordinates
    stress = np.zeros((3, 3, x.size))
    stress[0, 0] = -force*(length-x)*z/inertia
    stress[0, 2] = stress[2, 0] = 1.5*case.traction*(1.-4*z*z/height**2)
    return stress


def _cell_system(case: ElasticBendingCase) -> tuple[sps.csr_array, FloatArray]:
    """Use upstream TPSA's elastic cell Schur complement, retaining its signs and units."""
    ops, grid = case.operators, case.operators.grid
    nc, nf = grid.num_cells, grid.num_faces
    data = {pp.PARAMETERS: {"mechanics": {
        "fourth_order_tensor": pp.FourthOrderTensor(ops.mu, ops.lam), "bc": case.bc,
    }}, pp.DISCRETIZATION_MATRICES: {"mechanics": {}}}
    pp.Tpsa("mechanics").discretize(grid, data)
    m = data[pp.DISCRETIZATION_MATRICES]["mechanics"]
    flux = sps.block_array([
        [m["stress"], m["stress_rotation"], m["stress_total_pressure"]],
        [m["rotation_displacement"], m["rotation_rotation"], sps.csr_array((3*nf, nc))],
        [m["solid_mass_displacement"], sps.csr_array((nf, 3*nc)), m["solid_mass_total_pressure"]],
    ], format="csr")
    div = sps.block_diag([grid.divergence(dim=3), grid.divergence(dim=3), grid.divergence(dim=1)], format="csr")
    accum = sps.diags_array(np.r_[np.zeros(3*nc), np.repeat(grid.cell_volumes/ops.mu, 3), grid.cell_volumes/ops.lam])
    bound = sps.vstack([m["bound_stress"], m["bound_rotation_displacement"], m["bound_mass_displacement"]], format="csr")
    values = case.boundary.copy()
    values[:, ops.neumann_faces] = case.neumann
    return sps.csr_array(div @ flux - accum), np.asarray(-div @ (bound @ values.ravel("F")))


def _recover_faces(case: ElasticBendingCase, x: FloatArray) -> FloatArray:
    """Solve the diagonal free-face balance equations exactly (Q=0)."""
    ops, grid = case.operators, case.operators.grid
    u = x[ops.u_slice].reshape((3, grid.num_cells), order="F")
    rotation = ops.rotation_transfer.apply(x[ops.r_slice].reshape((3, grid.num_cells), order="F"))
    sides, cells, n, a = ops.side_faces, ops.side_cells, ops.side_normals, ops.side_stiffness
    numerator = a*u[:, cells] - np.cross(rotation[:, sides].T, n.T).T - x[ops.p_slice][cells]*n
    summed = np.vstack([np.bincount(sides, weights=row, minlength=grid.num_faces) for row in numerator])
    summed[:, ops.neumann_faces] += case.neumann / grid.face_areas[ops.neumann_faces]
    faces = summed / np.bincount(sides, weights=a, minlength=grid.num_faces)
    faces[:, ops.dirichlet_faces] = case.boundary[:, ops.dirichlet_faces]
    return faces


def _solve_banded_system(case: ElasticBendingCase, matrix: sps.csr_array, rhs: FloatArray) -> FloatArray:
    """Equivalent direct solve for the largest, narrow cantilever grids.

    Rescale rotation/pressure unknowns by mu and momentum rows by 1/mu,
    then symmetrically reorder with reverse Cuthill-McKee to limit bandwidth.
    Natural-order LU with diagonal pivot preference preserves that bandwidth.
    This is restricted to this elastic benchmark; it is not a general Newton
    linear solver. Check backward error and report the original physical
    residual after recovering the unknowns. Large cases can still need >10 GB.
    """
    ops = case.operators
    mu = float(ops.mu[0])
    columns = np.ones(matrix.shape[1])
    columns[ops.r_slice] = columns[ops.p_slice] = mu
    rows = np.ones(matrix.shape[0])
    rows[ops.u_slice] = 1/mu
    scaled = sps.csr_array(matrix.multiply(rows[:, None]).multiply(columns[None, :]))
    permutation = reverse_cuthill_mckee(scaled, symmetric_mode=False)
    ordered = sps.csc_matrix(scaled[permutation, :][:, permutation])
    factor = splu(ordered, permc_spec="NATURAL", diag_pivot_thresh=0.)
    y = np.empty_like(rhs)
    y[permutation] = factor.solve((rows*rhs)[permutation])
    residual = scaled @ y - rows*rhs
    scale = np.max(np.asarray(abs(scaled).sum(axis=1)))*np.linalg.norm(y, np.inf) + np.linalg.norm(rows*rhs, np.inf)
    if not np.all(np.isfinite(y)) or np.linalg.norm(residual, np.inf) > 1e-10*scale:
        raise RuntimeError("Banded elastic solve failed its backward-error check.")
    return columns*y


def relative_l2(actual: FloatArray, reference: FloatArray, weights: FloatArray) -> float:
    """Relative discrete L2, with spatial volume/area weights on the last axis."""
    denominator = np.sum(reference**2 * weights)
    if denominator <= 0:
        raise ValueError("A relative L2 error requires a nonzero reference.")
    return float(np.sqrt(np.sum((actual-reference)**2 * weights) / denominator))


def section_data(case: ElasticBendingCase, stress: FloatArray, traction: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Face: x, M_numeric, M_continuum, shear. Cell: x, M_material, M_continuum, M_sampled_reference.

    Moments use face/cell centers. Even exact linear stress has a midpoint
    quadrature deficit of 1/nz^2; the sampled reference separates this error
    from reconstruction error. All moments have units N mm.
    """
    ops, grid = case.operators, case.operators.grid
    length, width, height = case.dimensions
    inertia, force = width*height**3/12., case.traction*width*height
    expected = lambda x: case.young*inertia*case.curvature if case.kind == "pure_bending" else force*(length-x)
    # TensorGrid numbering is x-fastest; select the positive-x side except at x=0.
    mask = ops.side_normals[0] > .5
    mask |= (ops.side_normals[0] < -.5) & np.isclose(grid.face_centers[0, ops.side_faces], 0.)
    sides = np.flatnonzero(mask)
    positions = np.rint(np.interp(grid.face_centers[0, ops.side_faces[sides]],
                                  case.axes[0], np.arange(len(case.axes[0])))).astype(int)
    sign = ops.side_normals[0, sides]
    moment = np.bincount(positions, weights=-sign*grid.face_centers[2, ops.side_faces[sides]]*traction[0, sides], minlength=len(case.axes[0]))
    shear = np.bincount(positions, weights=sign*traction[2, sides], minlength=len(case.axes[0]))
    face_sections = np.column_stack((case.axes[0], moment, [expected(x) for x in case.axes[0]], shear))
    nx = case.shape[0]
    ix = np.arange(grid.num_cells) % nx
    areas = grid.cell_volumes / np.diff(case.axes[0])[ix]
    xc = .5*(case.axes[0][1:] + case.axes[0][:-1])
    material = np.bincount(ix, weights=-stress[0, 0]*grid.cell_centers[2]*areas, minlength=nx)
    ref = reference_stress(case, grid.cell_centers)
    sampled = np.bincount(ix, weights=-ref[0, 0]*grid.cell_centers[2]*areas, minlength=nx)
    return face_sections, np.column_stack((xc, material, [expected(x) for x in xc], sampled))


def solve_elastic(case: ElasticBendingCase) -> ElasticBendingResult:
    """Solve once, reconstruct prototype strain/stress, and retain verification metrics."""
    grid, ops = case.operators.grid, case.operators
    length, width, height = case.dimensions
    estimated_max_stress = (case.young*case.curvature*height/2 if case.kind == "pure_bending"
                            else 6*case.traction*length/height)
    if estimated_max_stress >= case.yield_stress:
        raise ValueError("Reference loading reaches yield; reduce the load for an elastic study.")
    matrix, rhs = _cell_system(case)
    # General-purpose LU has excessive fill on the largest wide cross-sections.
    x = (_solve_banded_system(case, matrix, rhs)
         if case.kind == "cantilever" and case.shape[1]*case.shape[2] > 64
         else _solve_linear_correction(matrix, -rhs, 0))
    faces = _recover_faces(case, x)
    z = ops.pack(x, faces[:, ops.free_faces])
    zero = np.zeros_like(case.boundary)
    strain = ops.strain_increment(z, case.boundary, zero)
    stress = 2*ops.mu*strain + ops.lam*np.eye(3)[:, :, None]*np.einsum("iic->c", strain)
    traction = evaluate_one_sided_traction(ops, z, case.boundary, zero, np.zeros_like(stress))
    residual = assemble_global_residual(ops, z, case.boundary, zero, traction, neumann_increment=case.neumann)
    reference = reference_stress(case, grid.cell_centers)
    dev = stress - np.eye(3)[:, :, None]*np.einsum("iic->c", stress)/3.
    vm = np.sqrt(1.5*np.einsum("ijc,ijc->c", dev, dev))
    if not np.all(np.isfinite(x)) or np.max(vm) >= case.yield_stress:
        raise ValueError("Computed stress is not safely elastic; reduce the load.")
    tip = ops.boundary_faces[np.isclose(grid.face_centers[0, ops.boundary_faces], length)]
    tip_value = float(np.average(faces[2, tip], weights=grid.face_areas[tip]))
    inertia = width*height**3/12.
    if case.kind == "pure_bending":
        exact_faces, _ = pure_bending_fields(grid.face_centers, case.curvature, case.young, case.poisson,
                                            variances=_face_variances(grid))
        reference_tip = float(np.average(exact_faces[2, tip], weights=grid.face_areas[tip]))
    else:
        reference_tip = case.traction*width*height*length**3/(3*case.young*inertia)
    boundary_sides = np.isin(ops.side_faces, ops.boundary_faces)
    force_imbalance = np.sum(traction[:, boundary_sides], axis=1)
    moment_imbalance = np.cross(grid.face_centers[:, ops.side_faces[boundary_sides]].T, traction[:, boundary_sides].T).sum(axis=0)
    fs, cs = section_data(case, stress, traction)
    # Keep end effects out of the engineering beam-stress comparison.
    interior = (grid.cell_centers[0] >= .1*length) & (grid.cell_centers[0] <= .9*length)
    fm = (fs[:, 0] >= .1*length) & (fs[:, 0] <= .9*length)
    dx = np.diff(case.axes[0])
    face_weights = .5*np.r_[dx[0], dx[:-1]+dx[1:], dx[-1]]
    metrics = {
        "cells": float(grid.num_cells), "retained_dofs": float(ops.num_dofs),
        "dx_max_mm": float(np.diff(case.axes[0]).max()),
        "dy_max_mm": float(np.diff(case.axes[1]).max()), "dz_max_mm": float(np.diff(case.axes[2]).max()),
        "tip_mm": tip_value, "reference_tip_mm": reference_tip,
        "tip_relative_error": abs(tip_value-reference_tip)/abs(reference_tip),
        "stress_relative_l2": relative_l2(stress, reference, grid.cell_volumes),
        "interior_sigma_xx_relative_l2": relative_l2(stress[0, 0, interior], reference[0, 0, interior], grid.cell_volumes[interior]),
        "interior_face_moment_relative_l2": relative_l2(fs[fm, 1], fs[fm, 2], face_weights[fm]),
        "material_moment_sampled_relative_l2": relative_l2(cs[:, 1], cs[:, 3], np.diff(case.axes[0])),
        "max_vm_MPa": float(vm.max()), "max_abs_sigma_xx_MPa": float(abs(stress[0, 0]).max()),
        "force_imbalance_N": float(np.linalg.norm(force_imbalance)),
        "moment_imbalance_Nmm": float(np.linalg.norm(moment_imbalance)),
        "retained_residual_max": float(abs(residual).max()),
        "cell_backward_error": float(np.linalg.norm(matrix @ x-rhs, np.inf) / (
            np.max(np.asarray(abs(matrix).sum(axis=1)))*np.linalg.norm(x, np.inf)+np.linalg.norm(rhs, np.inf))),
    }
    if case.kind == "pure_bending":
        reference_u, _ = pure_bending_fields(grid.cell_centers, case.curvature, case.young, case.poisson)
        metrics["displacement_relative_l2"] = relative_l2(x[ops.u_slice].reshape((3,grid.num_cells),order="F"), reference_u, grid.cell_volumes)
    return ElasticBendingResult(case, x, faces, strain, stress, traction, reference, metrics, fs, cs)
