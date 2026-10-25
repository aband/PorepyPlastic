"""Retained-face kinematics: affine patches, incremental history, and face ordering."""

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from porepy.grids.structured import CartGrid, TensorGrid
from porepy.params.bc import BoundaryConditionVectorial

from alternatives.plane_strain import PlaneStrainTpsa
from prototype.operators import RetainedFaceOperators


def make_operators(grid: TensorGrid) -> RetainedFaceOperators:
    grid.compute_geometry()
    faces = grid.get_all_boundary_faces()
    bc = BoundaryConditionVectorial(grid, faces, ["dir"] * len(faces))
    return RetainedFaceOperators(grid, bc)


@pytest.mark.parametrize("single_cell", [False, True])
def test_pack_preserves_component_order_and_boundary_has_no_dofs(single_cell: bool) -> None:
    grid = CartGrid(np.array([1, 1] if single_cell else [2, 3]))
    ops = make_operators(grid)
    cell = np.arange(4 * ops.num_cells, dtype=float)
    face = np.vstack((100 + ops.free_faces, 200 + ops.free_faces)).astype(float)
    z = ops.pack(cell, face)
    np.testing.assert_array_equal(z[ops.u_slice], cell[:2 * ops.num_cells])
    np.testing.assert_array_equal(z[ops.r_slice], cell[2 * ops.num_cells:3 * ops.num_cells])
    np.testing.assert_array_equal(z[ops.p_slice], cell[3 * ops.num_cells:])
    np.testing.assert_array_equal(z[ops.face_slice], face.T.ravel())
    np.testing.assert_array_equal(ops.face_dofs[:, ops.boundary_faces], -1)
    np.testing.assert_array_equal(z[ops.face_dofs[:, ops.free_faces]], face)
    cell_copy, face_copy = ops.unpack(z)
    np.testing.assert_array_equal(cell_copy, cell)
    np.testing.assert_array_equal(face_copy, face)
    cell_copy.fill(-1)
    face_copy.fill(-1)
    np.testing.assert_array_equal(ops.pack(cell, face), z)
    if single_cell:
        assert ops.num_free_faces == 0 and ops.num_dofs == 4


@pytest.mark.parametrize("field", ["translation", "rotation", "affine"])
@pytest.mark.parametrize("single_cell", [False, True])
def test_affine_increment_reproduces_total_strain_with_nonzero_history(field: str, single_cell: bool) -> None:
    # Nonuniform coordinate spacing also checks face areas and cell-volume factors.
    grid = (CartGrid(np.array([1, 1]), physdims=np.array([2.0, 0.8])) if single_cell else
            TensorGrid(np.array([0.0, 0.2, 0.9, 2.0]), np.array([0.0, 0.1, 0.8])))
    ops = make_operators(grid)
    old_gradient = np.array([[0.003, 0.0004], [-0.0002, -0.0005]])
    increment_gradient = {
        "translation": np.zeros((2, 2)),
        "rotation": np.array([[0, -0.002], [0.002, 0]]),
        "affine": np.array([[0.001, 0.0006], [-0.0002, -0.0003]]),
    }[field]
    old_face = old_gradient @ grid.face_centers[:2] + np.array([[0.005], [-0.002]])
    increment = increment_gradient @ grid.face_centers[:2] + np.array([[0.001], [-0.0003]])
    target = old_face + increment
    old_epsilon = np.zeros((3, 3, grid.num_cells))
    old_epsilon[:2, :2] = (0.5 * (old_gradient + old_gradient.T))[:, :, None]
    z = ops.pack(np.zeros(4 * grid.num_cells), increment[:, ops.free_faces])
    before = [v.copy() for v in (z, target, old_face, old_epsilon)]
    delta = ops.strain_increment(z, target, old_face)
    expected = np.zeros_like(delta)
    expected[:2, :2] = (0.5 * (increment_gradient + increment_gradient.T))[:, :, None]
    np.testing.assert_allclose(delta, expected, rtol=0, atol=1e-16)
    total = ops.reconstruct_strain(z, target, old_face, old_epsilon)
    np.testing.assert_allclose(total, old_epsilon + expected, rtol=0, atol=1e-16)
    for value, saved in zip((z, target, old_face, old_epsilon), before, strict=True):
        np.testing.assert_array_equal(value, saved)
    # Strain depends only on independent face increments, not on cell u/r/p.
    z[ops.cell_slice] = np.arange(4 * grid.num_cells) * 1e7
    np.testing.assert_array_equal(ops.reconstruct_strain(z, target, old_face, old_epsilon), total)


def test_boundary_values_are_incremental_and_unused_interior_entries_are_ignored() -> None:
    ops = make_operators(CartGrid(np.array([2, 2])))
    old = np.full((2, ops.num_faces), np.nan)
    target = np.full_like(old, np.inf)
    old[:, ops.boundary_faces] = 0.5
    target[:, ops.boundary_faces] = -0.25
    free = np.vstack((0.01 * ops.free_faces, -0.02 * ops.free_faces))
    z = ops.pack(np.zeros(4 * ops.num_cells), free)
    actual = ops.face_displacement_increment(z, target, old)
    np.testing.assert_array_equal(actual[:, ops.boundary_faces], -0.75)
    np.testing.assert_array_equal(actual[:, ops.free_faces], free)
    np.testing.assert_array_equal(ops.face_displacement_increment(z, old, old)[:, ops.boundary_faces], 0)


def test_face_normal_reversal_does_not_change_reconstructed_strain() -> None:
    ops = make_operators(CartGrid(np.array([2, 3]), physdims=np.array([1.5, 0.8])))
    rng = np.random.default_rng(17)
    z = rng.normal(size=ops.num_dofs)
    old, target = rng.normal(size=(2, 2, ops.num_faces))
    expected = ops.strain_increment(z, target, old)
    signs = np.ones(ops.num_faces)
    signs[::2] = -1
    ops.grid.face_normals *= signs
    ops.grid.cell_faces = sps.csc_matrix(sps.diags(signs) @ ops.grid.cell_faces)
    bc = BoundaryConditionVectorial(ops.grid, ops.boundary_faces, ["dir"] * len(ops.boundary_faces))
    reversed_ops = RetainedFaceOperators(ops.grid, bc)
    np.testing.assert_array_equal(reversed_ops.strain_increment(z, target, old), expected)


@pytest.mark.parametrize("normal_axis", [0, 1], ids=["vertical_face", "horizontal_face"])
@pytest.mark.parametrize("component", [0, 1], ids=["x_displacement", "y_displacement"])
def test_single_free_face_increment_is_local_and_conservative(
    normal_axis: int, component: int,
) -> None:
    grid = TensorGrid(np.array([0.0, 0.2, 1.1]), np.array([0.0, 0.7, 1.3]))
    ops = make_operators(grid)
    # Select one interior face by position, independently of the DOF map.
    center = np.array([0.2, 0.35] if normal_axis == 0 else [0.1, 0.7])
    faces = np.flatnonzero(np.all(np.isclose(grid.face_centers[:2], center[:, None]), axis=0))
    assert faces.size == 1
    free = np.zeros((2, ops.num_free_faces))
    free[component, ops.free_faces == faces[0]] = 0.003
    z = ops.pack(np.zeros(4 * ops.num_cells), free)
    boundary = np.zeros((2, ops.num_faces))

    actual = ops.strain_increment(z, boundary, boundary)

    # Area/volume = 1/width. Neighboring cells see opposite outward normals;
    # tangential displacement contributes half that gradient to tensor shear.
    expected = np.zeros((3, 3, ops.num_cells))
    neighbors = (0, 1) if normal_axis == 0 else (0, 2)
    widths = (0.2, 0.9) if normal_axis == 0 else (0.7, 0.6)
    for cell, width, sign in zip(neighbors, widths, (1, -1), strict=True):
        value = sign * 0.003 / width
        expected[component, normal_axis, cell] += 0.5 * value
        expected[normal_axis, component, cell] += 0.5 * value
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-17)
    np.testing.assert_allclose(np.sum(actual * grid.cell_volumes, axis=2), 0, atol=1e-17)


def test_loading_subdivision_and_unloading_preserve_committed_history() -> None:
    ops = make_operators(TensorGrid(np.array([0.0, 0.3, 1.0]), np.array([0.0, 0.4, 1.2])))
    rng = np.random.default_rng(21)
    old_face, first_increment, second_increment = rng.normal(scale=0.001, size=(3, 2, ops.num_faces))
    old_strain = np.zeros((3, 3, ops.num_cells))
    initial = rng.normal(scale=0.001, size=(2, 2, ops.num_cells))
    old_strain[:2, :2] = 0.5 * (initial + initial.swapaxes(0, 1))
    saved_face, saved_strain = old_face.copy(), old_strain.copy()
    cell = np.zeros(4 * ops.num_cells)
    first_z = ops.pack(cell, first_increment[:, ops.free_faces])
    second_z = ops.pack(cell, second_increment[:, ops.free_faces])
    first_target = old_face + first_increment
    second_target = first_target + second_increment

    first_strain = ops.reconstruct_strain(first_z, first_target, old_face, old_strain)
    saved_first = first_strain.copy()
    split = ops.reconstruct_strain(second_z, second_target, first_target, first_strain)
    combined = ops.reconstruct_strain(first_z + second_z, second_target, old_face, old_strain)
    np.testing.assert_allclose(split, combined, rtol=1e-14, atol=1e-17)

    unloaded = ops.reconstruct_strain(-first_z - second_z, old_face, second_target, split)
    np.testing.assert_allclose(unloaded, old_strain, rtol=1e-14, atol=1e-17)
    # Rejected Newton trials and returned arrays must not alter fixed history.
    split.fill(0)
    repeated = ops.reconstruct_strain(first_z, first_target, old_face, old_strain)
    np.testing.assert_array_equal(repeated, saved_first)
    np.testing.assert_array_equal(first_strain, saved_first)
    np.testing.assert_array_equal(old_face, saved_face)
    np.testing.assert_array_equal(old_strain, saved_strain)


def test_operators_use_the_existing_case_without_a_solution_or_material_updates() -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.prepare_simulation()
    ops = RetainedFaceOperators(case.grid, case.bc)
    assert ops.grid is case.grid and case.x is None
    assert all(point.trial is None and point.committed.alpha == 0 for point in case.material_points)


@pytest.mark.parametrize("defect", ["uncomputed", "three_dimensional", "tilted", "neumann"])
def test_rejects_unsupported_setup(defect: str) -> None:
    grid = CartGrid(np.array([2, 2, 2] if defect == "three_dimensional" else [2, 2]))
    if defect != "uncomputed":
        grid.compute_geometry()
    if defect == "tilted":
        grid.nodes[2] = 0.1 * grid.nodes[0]
        grid.compute_geometry()
    faces = grid.get_all_boundary_faces()
    bc = BoundaryConditionVectorial(grid, faces, ["neu" if defect == "neumann" else "dir"] * len(faces))
    with pytest.raises(ValueError, match={
        "uncomputed": "compute_geometry", "three_dimensional": "2D", "tilted": "xy plane", "neumann": "Dirichlet",
    }[defect]):
        RetainedFaceOperators(grid, bc)


@pytest.mark.parametrize("field", ["cell", "free_face"])
@pytest.mark.parametrize("defect", ["shape", "nan", "inf", "negative_inf"])
def test_pack_rejects_invalid_increments(field: str, defect: str) -> None:
    ops = make_operators(CartGrid(np.array([2, 2])))
    values = [np.zeros(4 * ops.num_cells), np.zeros((2, ops.num_free_faces))]
    index = 0 if field == "cell" else 1
    if defect == "shape":
        values[index] = values[index][:-1]
    else:
        values[index].flat[0] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[defect]
    with pytest.raises(ValueError, match=f"{field}_increment"):
        ops.pack(values[0], values[1])


@pytest.mark.parametrize(
    "defect", ["missing_boundary_neighbor", "missing_interior_neighbor", "same_orientation", "nonunit_sign"],
)
def test_rejects_inconsistent_face_cell_incidence(defect: str) -> None:
    ops = make_operators(CartGrid(np.array([2, 2])))
    grid = ops.grid
    bc = BoundaryConditionVectorial(grid, ops.boundary_faces, ["dir"] * len(ops.boundary_faces))
    incidence = grid.cell_faces.tocsr(copy=True)
    face = ops.boundary_faces[0] if defect == "missing_boundary_neighbor" else ops.free_faces[0]
    start = incidence.indptr[face]
    if defect.startswith("missing"):
        incidence.data[start] = 0
        incidence.eliminate_zeros()
    elif defect == "same_orientation":
        incidence.data[start] = incidence.data[start + 1]
    else:
        incidence.data[start] = 2
    grid.cell_faces = incidence.tocsc()
    with pytest.raises(ValueError, match="oppositely oriented interior neighbors"):
        RetainedFaceOperators(grid, bc)


@pytest.mark.parametrize("defect", ["z_shape", "z_nan", "bc_shape", "bc_nan", "old_bc_nan", "strain_shape", "strain_nan", "strain_nonsymmetric", "strain_zz"])
def test_invalid_increment_or_history_is_rejected(defect: str) -> None:
    ops = make_operators(CartGrid(np.array([2, 2])))
    z, target = np.zeros(ops.num_dofs), np.zeros((2, ops.num_faces))
    old, epsilon = target.copy(), np.zeros((3, 3, ops.num_cells))
    if defect == "z_shape":
        z = z[:-1]
    elif defect == "z_nan":
        z[-1] = np.nan
    elif defect == "bc_shape":
        target = target[:, :-1]
    elif defect == "bc_nan":
        target[0, ops.boundary_faces[0]] = np.nan
    elif defect == "old_bc_nan":
        old[0, ops.boundary_faces[0]] = np.nan
    elif defect == "strain_shape":
        epsilon = epsilon[:, :, :-1]
    elif defect == "strain_nan":
        epsilon[0, 0, 0] = np.nan
    elif defect == "strain_nonsymmetric":
        epsilon[0, 1, 0] = 0.001
    else:
        epsilon[2, 2, 0] = 0.001
    with pytest.raises(ValueError):
        ops.reconstruct_strain(z, target, old, epsilon)
