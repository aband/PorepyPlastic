"""Retained-face equilibrium: patch tests, interface balance, loads, and TPSA parity."""

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.structured import CartGrid, TensorGrid
from porepy.params.bc import BoundaryConditionVectorial
from porepy.params.tensor import FourthOrderTensor
from porepy.utils.common_constants import PARAMETERS

from alternatives.plane_strain import KEYWORD, PlaneStrainTpsa, _assemble_matrices, _discretize_get_matrices
from alternatives.postprocessing import reconstruct_face_displacement
from core.operators import RetainedFaceOperators
from core.residual import assemble_global_residual, evaluate_one_sided_traction


def make_operators(
    *, mu: float | NDArray[np.float64] | None = 2.0,
    lam: float | NDArray[np.float64] | None = 3.0,
    single_cell: bool = False, angle: float = 0.0,
) -> RetainedFaceOperators:
    grid = (CartGrid(np.array([1, 1]), physdims=np.array([2.0, 0.8])) if single_cell else
            TensorGrid(np.array([0.0, 0.5, 2.0]), np.array([0.0, 0.2, 0.8])))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    grid.nodes[:2] = rotation @ grid.nodes[:2]
    grid.compute_geometry()
    bf = grid.get_all_boundary_faces()
    return RetainedFaceOperators(grid, BoundaryConditionVectorial(grid, bf, ["dir"] * len(bf)), mu=mu, lam=lam)


def test_heterogeneous_accumulation_coefficients_are_owned_snapshots() -> None:
    mu, lam = np.array([2., 5., 7., 11.]), np.array([3., 9., 12., 15.])
    ops = make_operators(mu=mu, lam=lam)
    assert ops.rotation_accumulation is not None and ops.pressure_accumulation is not None
    expected_r = np.array([0.1 / 2, 0.3 / 5, 0.3 / 7, 0.9 / 11])
    expected_p = np.array([0.1 / 3, 0.3 / 9, 0.3 / 12, 0.9 / 15])
    mu[:] = lam[:] = 100
    np.testing.assert_allclose(ops.rotation_accumulation, expected_r)
    np.testing.assert_allclose(ops.pressure_accumulation, expected_p)
    np.testing.assert_array_equal(ops.mu, [2, 5, 7, 11])
    np.testing.assert_array_equal(ops.lam, [3, 9, 12, 15])


@pytest.mark.parametrize("single_cell", [False, True])
def test_zero_increment_is_equilibrated_with_nonzero_boundary_history(single_cell: bool) -> None:
    ops = make_operators(single_cell=single_cell)
    z = np.zeros(ops.num_dofs)
    old = np.full((2, ops.num_faces), np.nan)
    old[:, ops.boundary_faces] = 0.003
    target = old.copy()
    target[:, ops.free_faces] = np.inf
    traction = evaluate_one_sided_traction(ops, z, target, old, np.zeros((3, 3, ops.num_cells)))
    residual = assemble_global_residual(ops, z, target, old, traction)
    assert residual.shape == (4 * ops.num_cells + 2 * ops.num_free_faces,)
    np.testing.assert_array_equal(residual, 0)


@pytest.mark.parametrize("field", ["translation", "rotation", "affine"])
@pytest.mark.parametrize("single_cell", [False, True])
@pytest.mark.parametrize("angle", [0.0, 0.37])
def test_elastic_affine_patch_satisfies_all_four_blocks(field: str, single_cell: bool, angle: float) -> None:
    ops = make_operators(single_cell=single_cell, angle=angle)
    g, nc = ops.grid, ops.num_cells
    gradient = {
        "translation": np.zeros((2, 2)),
        "rotation": np.array([[0., -0.003], [0.003, 0.]]),
        "affine": np.array([[0.001, 0.002], [-0.0003, -0.0004]]),
    }[field]
    shift = np.array([[0.005], [-0.003]])
    u = gradient @ g.cell_centers[:2] + shift
    d = gradient @ g.face_centers[:2] + shift
    cell = np.r_[u.ravel("F"), np.full(nc, 2 * (gradient[0, 1] - gradient[1, 0])),
                 np.full(nc, 3 * np.trace(gradient))]
    z = ops.pack(cell, d[:, ops.free_faces])
    old = np.random.default_rng(8).normal(scale=0.001, size=d.shape)
    target = old + d
    traction = evaluate_one_sided_traction(ops, z, target, old, np.zeros((3, 3, nc)))
    result = assemble_global_residual(ops, z, target, old, traction)
    np.testing.assert_allclose(result, 0, atol=1e-16)


@pytest.mark.parametrize("location", ["interior", "boundary"])
def test_supplied_tractions_are_summed_without_extra_area_or_orientation(location: str) -> None:
    ops = make_operators()
    face = ops.free_faces[-1] if location == "interior" else ops.boundary_faces[0]
    sides = np.flatnonzero(ops.side_faces == face)
    traction = np.zeros((2, ops.num_sides))
    traction[:, sides] = np.array([[3., 5.], [-2., 7.]])[:, :sides.size]
    boundary = np.zeros((2, ops.num_faces))
    actual = assemble_global_residual(ops, np.zeros(ops.num_dofs), boundary, boundary, traction)
    expected = np.zeros(ops.num_dofs)
    for side in sides:
        cell = ops.side_cells[side]
        expected[2 * cell:2 * cell + 2] = traction[:, side]
    if location == "interior":
        expected[-2:] = [8., 5.]
    np.testing.assert_array_equal(actual, expected)


def test_rotation_and_pressure_increments_are_not_solved_or_overwritten() -> None:
    ops = make_operators()
    z = np.zeros(ops.num_dofs)
    z[ops.r_slice], z[ops.p_slice] = [1, -2, 3, -4], [2, -3, 4, -5]
    original = z.copy()
    boundary = np.zeros((2, ops.num_faces))
    result = assemble_global_residual(ops, z, boundary, boundary, np.zeros((2, ops.num_sides)))
    np.testing.assert_allclose(result[ops.r_slice], [-0.05, 0.3, -0.45, 1.8])
    np.testing.assert_allclose(result[ops.p_slice], [-0.2 / 3, 0.3, -0.4, 1.5])
    np.testing.assert_array_equal(result[ops.u_slice], 0)
    np.testing.assert_array_equal(result[ops.face_slice], 0)
    np.testing.assert_array_equal(z, original)


def test_integrated_load_increment_and_committed_imbalance_are_applied_once() -> None:
    ops = make_operators()
    z = np.zeros(ops.num_dofs)
    boundary = np.zeros((2, ops.num_faces))
    traction = np.zeros((2, ops.num_sides))
    body = np.array([[2., -3., 4., -5.], [-6., 7., -8., 9.]])
    old_residual = np.arange(ops.num_dofs, dtype=float) + 0.5
    expected = old_residual.copy()
    expected[:2 * ops.num_cells] -= body.ravel("F")
    inputs = (z, boundary, traction, body, old_residual)
    snapshots = [value.copy() for value in inputs]
    for value in inputs:
        value.flags.writeable = False
    first = assemble_global_residual(ops, z, boundary, boundary, traction,
                                    body_force_increment=body, committed_residual=old_residual)
    np.testing.assert_array_equal(first, expected)
    first.fill(0)
    np.testing.assert_array_equal(assemble_global_residual(
        ops, z, boundary, boundary, traction, body_force_increment=body, committed_residual=old_residual,
    ), expected)
    for value, snapshot in zip(inputs, snapshots, strict=True):
        np.testing.assert_array_equal(value, snapshot)


def test_global_force_sum_accounts_for_interface_imbalance_and_boundary_reactions() -> None:
    ops = make_operators()
    rng = np.random.default_rng(13)
    traction, body = rng.normal(size=(2, ops.num_sides)), rng.normal(size=(2, ops.num_cells))
    boundary = np.zeros((2, ops.num_faces))
    result = assemble_global_residual(ops, np.zeros(ops.num_dofs), boundary, boundary, traction,
                                      body_force_increment=body)
    boundary_sides = np.isin(ops.side_faces, ops.boundary_faces)
    momentum = result[ops.u_slice].reshape((2, -1), order="F").sum(axis=1)
    interface = result[ops.face_slice].reshape((2, -1), order="F").sum(axis=1)
    np.testing.assert_allclose(momentum, interface + traction[:, boundary_sides].sum(axis=1) - body.sum(axis=1))


def test_cell_plastic_correction_enters_face_balance_without_double_counting_pressure() -> None:
    ops = make_operators()
    z, boundary = np.zeros(ops.num_dofs), np.zeros((2, ops.num_faces))
    q = np.zeros((3, 3, ops.num_cells))
    q[:, :, 0] = [[2, 3, 0], [3, 4, 0], [0, 0, 5]]  # Nonzero trace, as in volumetric plasticity.
    traction = evaluate_one_sided_traction(ops, z, boundary, boundary, q)
    result = assemble_global_residual(ops, z, boundary, boundary, traction)
    np.testing.assert_allclose(result[ops.u_slice], 0, atol=1e-14)
    np.testing.assert_array_equal(result[ops.r_slice], 0)
    np.testing.assert_array_equal(result[ops.p_slice], 0)
    assert np.linalg.norm(result[ops.face_slice]) > 0


@pytest.mark.parametrize("solve", [False, True], ids=["arbitrary_candidate", "equilibrated_solution"])
def test_eliminated_elastic_faces_match_original_tpsa_system(solve: bool) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.prepare_simulation()
    g, nc = case.grid, case.grid.num_cells
    g.nodes[0] = 1.6 * g.nodes[0] ** 1.4
    g.nodes[1] *= 0.7
    g.compute_geometry()
    mu, lam = np.array([2., 7., 11., 3.]), np.array([3., 5., 8., 4.])
    case.d[PARAMETERS][KEYWORD]["fourth_order_tensor"] = FourthOrderTensor(mu=mu, lmbda=lam)
    case.matrices = _discretize_get_matrices(g, case.d)
    case.face_discretization, case.rhs_matrix, case.div, case.accum = _assemble_matrices(case.matrices, g, case.d)
    ops = RetainedFaceOperators(g, case.bc, mu=mu, lam=lam)
    rng = np.random.default_rng(12)
    delta_bc = np.zeros((2, g.num_faces))
    delta_bc[:, ops.boundary_faces] = rng.normal(scale=0.01, size=(2, ops.boundary_faces.size))
    old = rng.normal(scale=0.01, size=delta_bc.shape)
    target = old + delta_bc
    body = rng.normal(size=(2, nc))
    sources = np.r_[body.ravel("F"), np.zeros(2 * nc)]
    matrix = case.div @ case.face_discretization - case.accum
    rhs = sources - case.div @ case.rhs_matrix @ delta_bc.ravel("F")
    x = np.asarray(sps.linalg.spsolve(matrix, rhs)) if solve else rng.normal(size=4 * nc)
    d = reconstruct_face_displacement(g, case.face_discretization, case.rhs_matrix, x, delta_bc)
    z = ops.pack(x, d[:, ops.free_faces])
    traction = evaluate_one_sided_traction(ops, z, target, old, np.zeros((3, 3, nc)))
    result = assemble_global_residual(ops, z, target, old, traction, body_force_increment=body)
    expected = np.asarray(matrix @ x - rhs)
    np.testing.assert_allclose(result[ops.cell_slice], expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result[ops.face_slice], 0, atol=1e-13)
    if solve:
        np.testing.assert_allclose(result, 0, atol=1e-13)


def test_residual_is_invariant_under_normal_and_incidence_reversal() -> None:
    ops = make_operators(angle=0.37)
    rng = np.random.default_rng(7)
    z, traction = rng.normal(size=ops.num_dofs), rng.normal(size=(2, ops.num_sides))
    target, old = rng.normal(size=(2, 2, ops.num_faces))
    expected = assemble_global_residual(ops, z, target, old, traction)
    g = ops.grid
    signs = np.ones(ops.num_faces)
    signs[::2] = -1
    g.face_normals *= signs
    g.cell_faces = sps.csc_matrix(sps.diags(signs) @ g.cell_faces)
    bc = BoundaryConditionVectorial(g, ops.boundary_faces, ["dir"] * len(ops.boundary_faces))
    reversed_ops = RetainedFaceOperators(g, bc, mu=2.0, lam=3.0)
    np.testing.assert_array_equal(assemble_global_residual(reversed_ops, z, target, old, traction), expected)


@pytest.mark.parametrize("lam", [0.0, -1.0, np.inf, np.nan, np.ones(3), np.array([1., 2., 3., 0.])])
def test_rejects_invalid_lame_parameter(lam: float | NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="lam must"):
        make_operators(lam=lam)


def test_lame_parameter_requires_shear_modulus() -> None:
    with pytest.raises(ValueError, match="Supply mu together with lam"):
        make_operators(mu=None)


@pytest.mark.parametrize("mu", [None, 2.0])
def test_residual_requires_both_elastic_coefficients(mu: float | None) -> None:
    ops = make_operators(mu=mu, lam=None)
    boundary = np.zeros((2, ops.num_faces))
    with pytest.raises(ValueError, match="Supply mu and lam"):
        assemble_global_residual(ops, np.zeros(ops.num_dofs), boundary, boundary, np.zeros((2, ops.num_sides)))


@pytest.mark.parametrize("field", ["z", "target", "old", "traction", "body", "committed"])
@pytest.mark.parametrize("defect", ["shape", "nonfinite"])
def test_rejects_invalid_residual_inputs(field: str, defect: str) -> None:
    ops = make_operators()
    values = {
        "z": np.zeros(ops.num_dofs), "target": np.zeros((2, ops.num_faces)),
        "old": np.zeros((2, ops.num_faces)), "traction": np.zeros((2, ops.num_sides)),
        "body": np.zeros((2, ops.num_cells)), "committed": np.zeros(ops.num_dofs),
    }
    if defect == "shape":
        values[field] = values[field][:-1]
    else:
        values[field].flat[0] = np.inf
    with pytest.raises(ValueError):
        assemble_global_residual(ops, values["z"], values["target"], values["old"], values["traction"],
                                 body_force_increment=values["body"], committed_residual=values["committed"])
