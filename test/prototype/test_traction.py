"""One-sided retained-face laws: physical patches, weights, signs, and TPSA closure."""

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.structured import CartGrid, TensorGrid
from porepy.params.bc import BoundaryConditionVectorial
from porepy.params.tensor import FourthOrderTensor
from porepy.utils.common_constants import PARAMETERS

from alternatives.coupled.residual import CoupledOperators
from utils.reconstruction import mandel
from alternatives.plane_strain import KEYWORD, PlaneStrainTpsa, _assemble_matrices, _discretize_get_matrices
from prototype.operators import RetainedFaceOperators
from prototype.residual import evaluate_one_sided_traction


def make_operators(
    *, mu: float | NDArray[np.float64] | None = 2.0,
    angle: float = 0.0, single_cell: bool = False,
) -> RetainedFaceOperators:
    grid = (CartGrid(np.array([1, 1]), physdims=np.array([2.0, 0.8])) if single_cell else
            TensorGrid(np.array([0.0, 0.5, 2.0]), np.array([0.0, 0.8])))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    grid.nodes[:2] = rotation @ grid.nodes[:2]
    grid.compute_geometry()
    boundary = grid.get_all_boundary_faces()
    bc = BoundaryConditionVectorial(grid, boundary, ["dir"] * len(boundary))
    return RetainedFaceOperators(grid, bc, mu=mu)


def test_side_geometry_and_material_coefficients() -> None:
    mu = np.array([2.0, 9.0])
    ops = make_operators(mu=mu)
    np.testing.assert_array_equal(ops.side_faces, [0, 1, 1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(ops.side_cells, [0, 0, 1, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(ops.side_normals, [
        [-1, 1, -1, 1, 0, 0, 0, 0], [0, 0, 0, 0, -1, -1, 1, 1],
    ])
    np.testing.assert_allclose(ops.side_areas, [0.8, 0.8, 0.8, 0.8, 0.5, 1.5, 0.5, 1.5])
    np.testing.assert_allclose(ops.side_distances, [0.25, 0.25, 0.75, 0.75, 0.4, 0.4, 0.4, 0.4])
    assert ops.side_stiffness is not None
    np.testing.assert_allclose(ops.side_stiffness, [16, 16, 24, 24, 10, 45, 10, 45])
    assert ops.num_sides == 8 and ops.rotation_transfer is not None
    np.testing.assert_allclose(ops.rotation_transfer.weights.toarray()[1], [0.6, 0.4])
    mu[:] = 100
    np.testing.assert_array_equal(ops.mu, [2, 9])
    np.testing.assert_allclose(ops.side_stiffness, [16, 16, 24, 24, 10, 45, 10, 45])


@pytest.mark.parametrize("field", ["zero", "translation", "rotation", "affine"])
@pytest.mark.parametrize("single_cell", [False, True])
@pytest.mark.parametrize("angle", [0.0, 0.37])
def test_elastic_affine_patch_matches_physical_stress(
    field: str, single_cell: bool, angle: float,
) -> None:
    mu, lam = 2.0, 3.0
    ops = make_operators(mu=mu, angle=angle, single_cell=single_cell)
    g, nc = ops.grid, ops.num_cells
    gradient = {
        "zero": np.zeros((2, 2)), "translation": np.zeros((2, 2)),
        "rotation": np.array([[0, -0.003], [0.003, 0]]),
        "affine": np.array([[0.001, 0.002], [-0.0003, -0.0004]]),
    }[field]
    shift = np.zeros((2, 1)) if field == "zero" else np.array([[0.005], [-0.003]])
    displacement = gradient @ g.cell_centers[:2] + shift
    face = gradient @ g.face_centers[:2] + shift
    cell = np.r_[displacement.ravel("F"), np.full(nc, mu * (gradient[0, 1] - gradient[1, 0])),
                 np.full(nc, lam * np.trace(gradient))]
    z = ops.pack(cell, face[:, ops.free_faces])
    old = np.full((2, ops.num_faces), 0.0007)
    q = np.zeros((3, 3, nc))
    actual = evaluate_one_sided_traction(ops, z, old + face, old, q)

    stress = mu * (gradient + gradient.T) + lam * np.trace(gradient) * np.eye(2)
    # Independently recover each side's area normal from PorePy's incidence.
    expected = np.column_stack([
        stress @ (g.cell_faces[f, c] * g.face_normals[:2, f])
        for f, c in zip(ops.side_faces, ops.side_cells, strict=True)
    ])
    assert actual.shape == (2, ops.num_sides)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-16)
    if not single_cell:
        np.testing.assert_allclose(actual[:, ops.side_faces == 1].sum(axis=1), 0, atol=1e-16)


def test_heterogeneous_shared_rotation_and_cell_pressure_use_their_own_values() -> None:
    ops = make_operators(mu=np.array([2.0, 9.0]))
    z = np.zeros(ops.num_dofs)
    z[ops.r_slice] = [3, 8]
    z[ops.p_slice] = [7, 11]
    boundary = np.zeros((2, ops.num_faces))
    traction = evaluate_one_sided_traction(ops, z, boundary, boundary, np.zeros((3, 3, 2)))
    # a=(16,24) gives r_face=0.6*3+0.4*8=5; pressures stay one-sided.
    np.testing.assert_allclose(traction[:, ops.side_faces == 1], [[5.6, -8.8], [4.0, -4.0]])
    # Left and right Dirichlet faces use r_i=3 and r_j=8, respectively.
    np.testing.assert_allclose(traction[:, ops.side_faces == 0], [[-5.6], [-2.4]])
    np.testing.assert_allclose(traction[:, ops.side_faces == 2], [[8.8], [6.4]])


def test_plastic_corrections_remain_one_sided_and_cancel_in_each_cell_sum() -> None:
    ops = make_operators(mu=np.array([2.0, 9.0]))
    q = np.stack((np.array([[2., 3., 10.], [3., 4., 20.], [10., 20., 50.]]),
                  np.array([[7., -2., 30.], [-2., -5., 40.], [30., 40., 60.]])), axis=2)
    boundary = np.zeros((2, ops.num_faces))
    traction = evaluate_one_sided_traction(ops, np.zeros(ops.num_dofs), boundary, boundary, q)
    # Interface normals are +x and -x. Neither Q is averaged with its neighbor.
    np.testing.assert_allclose(traction[:, ops.side_faces == 1], [[-1.6, 5.6], [-2.4, -1.6]])
    assert np.linalg.norm(traction[:, ops.side_faces == 1].sum(axis=1)) > 1
    for cell in range(ops.num_cells):
        np.testing.assert_allclose(traction[:, ops.side_cells == cell].sum(axis=1), 0, atol=1e-14)
    in_plane = q.copy()
    in_plane[2] = in_plane[:, 2] = 0
    np.testing.assert_array_equal(
        evaluate_one_sided_traction(ops, np.zeros(ops.num_dofs), boundary, boundary, in_plane), traction,
    )


def test_free_displacement_is_not_eliminated_to_enforce_interface_balance() -> None:
    ops = make_operators(mu=np.array([2.0, 9.0]))
    z = ops.pack(np.zeros(8), np.array([[0.01], [-0.02]]))
    boundary = np.zeros((2, ops.num_faces))
    traction = evaluate_one_sided_traction(ops, z, boundary, boundary, np.zeros((3, 3, 2)))
    np.testing.assert_allclose(traction[:, ops.side_faces == 1], [[0.128, 0.192], [-0.256, -0.384]])
    np.testing.assert_array_equal(traction[:, ops.side_faces != 1], 0)


def test_dirichlet_reaction_increments_ignore_unused_boundary_entries() -> None:
    ops = make_operators(single_cell=True)
    old = np.full((2, ops.num_faces), 0.003)
    increment = np.array([[0.001], [-0.002]])
    result = evaluate_one_sided_traction(
        ops, np.zeros(ops.num_dofs), old + increment, old, np.zeros((3, 3, 1)),
    )
    # For the 2 by 0.8 rectangle: A*a is 3.2 on vertical and 20 on horizontal faces.
    np.testing.assert_allclose(result, increment * np.array([3.2, 3.2, 20, 20]))

    ops = make_operators()
    old = np.full((2, ops.num_faces), np.nan)
    target = np.full_like(old, np.inf)
    old[:, ops.boundary_faces] = 0.003
    target[:, ops.boundary_faces] = 0.003
    np.testing.assert_array_equal(evaluate_one_sided_traction(
        ops, np.zeros(ops.num_dofs), target, old, np.zeros((3, 3, 2)),
    ), 0)


@pytest.mark.parametrize("plastic", [False, True], ids=["elastic", "plastic"])
def test_eliminated_faces_recover_existing_tpsa_tractions(plastic: bool) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.prepare_simulation()
    g = case.grid
    g.nodes[0] = 1.6 * g.nodes[0] ** 1.4
    g.nodes[1] *= 0.7
    g.compute_geometry()
    mu = np.array([2., 7., 11., 3.])
    case.d[PARAMETERS][KEYWORD]["fourth_order_tensor"] = FourthOrderTensor(mu=mu, lmbda=mu * 1.5)
    case.matrices = _discretize_get_matrices(g, case.d)
    case.face_discretization, case.rhs_matrix, case.div, case.accum = _assemble_matrices(case.matrices, g, case.d)
    condensed = CoupledOperators(case)
    ops = RetainedFaceOperators(g, case.bc, mu=mu)
    rng = np.random.default_rng(32)
    x, boundary = rng.normal(size=4 * g.num_cells), rng.normal(size=(2, g.num_faces))
    q = rng.normal(size=(3, 3, g.num_cells)) if plastic else np.zeros((3, 3, g.num_cells))
    q = 0.5 * (q + q.swapaxes(0, 1))
    face = (condensed.U @ x + condensed.U_g @ boundary.ravel("F") + condensed.H_f @ mandel(q)).reshape((2, -1), order="F")
    expected = (condensed.T @ x + condensed.T_g @ boundary.ravel("F") + condensed.H_t @ mandel(q)).reshape((2, -1), order="F")
    actual = evaluate_one_sided_traction(
        ops, ops.pack(x, face[:, ops.free_faces]), boundary, np.zeros_like(boundary), q,
    )
    np.testing.assert_allclose(actual, expected[:, ops.side_faces] * ops.side_signs, rtol=1e-12, atol=1e-13)
    for face_id in ops.free_faces:
        np.testing.assert_allclose(actual[:, ops.side_faces == face_id].sum(axis=1), 0, atol=1e-13)


def test_normal_reversal_and_repeated_evaluation_preserve_inputs() -> None:
    ops = make_operators(mu=np.array([2.0, 9.0]), angle=0.37)
    rng = np.random.default_rng(5)
    z = rng.normal(size=ops.num_dofs)
    old, target = rng.normal(size=(2, 2, ops.num_faces))
    q = rng.normal(size=(3, 3, ops.num_cells))
    q = q + q.swapaxes(0, 1)
    inputs = (z, target, old, q)
    saved = [value.copy() for value in inputs]
    for value in inputs:
        value.flags.writeable = False
    expected = evaluate_one_sided_traction(ops, *inputs)
    first = evaluate_one_sided_traction(ops, *inputs)
    first.fill(0)
    np.testing.assert_array_equal(evaluate_one_sided_traction(ops, *inputs), expected)
    for actual, snapshot in zip(inputs, saved, strict=True):
        np.testing.assert_array_equal(actual, snapshot)
    signs = np.ones(ops.num_faces)
    signs[::2] = -1
    g = ops.grid
    g.face_normals *= signs
    g.cell_faces = sps.csc_matrix(sps.diags(signs) @ g.cell_faces)
    bc = BoundaryConditionVectorial(g, ops.boundary_faces, ["dir"] * len(ops.boundary_faces))
    reversed_ops = RetainedFaceOperators(g, bc, mu=np.array([2., 9.]))
    np.testing.assert_array_equal(evaluate_one_sided_traction(reversed_ops, *inputs), expected)


@pytest.mark.parametrize("mu", [0.0, -1.0, np.inf, np.nan, np.ones(3), np.array([1., 0.])])
def test_rejects_invalid_shear_modulus(mu: float | NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="mu must"):
        make_operators(mu=mu)


def test_traction_requires_material_coefficients() -> None:
    ops = make_operators(mu=None)
    boundary = np.zeros((2, ops.num_faces))
    with pytest.raises(ValueError, match="Supply mu"):
        evaluate_one_sided_traction(ops, np.zeros(ops.num_dofs), boundary, boundary, np.zeros((3, 3, 2)))


@pytest.mark.parametrize("defect", ["shape", "nan", "nonsymmetric"])
def test_rejects_invalid_stress_correction(defect: str) -> None:
    ops = make_operators()
    q = np.zeros((3, 3, ops.num_cells))
    if defect == "shape":
        q = q[:2, :2]
    elif defect == "nan":
        q[2, 2, 0] = np.nan
    else:
        q[0, 1, 0] = 1
    boundary = np.zeros((2, ops.num_faces))
    with pytest.raises(ValueError, match="stress_correction"):
        evaluate_one_sided_traction(ops, np.zeros(ops.num_dofs), boundary, boundary, q)


@pytest.mark.parametrize("defect", ["skew", "normal_magnitude", "negative_distance"])
def test_rejects_geometry_incompatible_with_traction_law(defect: str) -> None:
    ops = make_operators()
    g = ops.grid
    if defect == "skew":
        g.nodes[0] += 0.1 * g.nodes[1]
        g.compute_geometry()
    elif defect == "normal_magnitude":
        g.face_normals[:, 0] *= 2
    else:
        g.cell_centers[0, 0] = -1
    bc = BoundaryConditionVectorial(g, ops.boundary_faces, ["dir"] * len(ops.boundary_faces))
    with pytest.raises(ValueError, match={
        "skew": "orthogonal", "normal_magnitude": "magnitude", "negative_distance": "positive outward",
    }[defect]):
        RetainedFaceOperators(g, bc, mu=2.0)
