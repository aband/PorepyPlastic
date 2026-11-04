"""Circumcenter preparation and triangular strain/traction patch checks."""

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

from utils.geometry import preprocess_simplex_grid
from core.layout import DofLayout
from core.operators import RetainedFaceOperators
from core.tpsa_wrapper import ElasticTpsa


def _triangles(acute: bool) -> pp.StructuredTriangleGrid:
    grid = pp.StructuredTriangleGrid(np.array([2, 2]), np.ones(2))
    if acute:
        # Turn the right triangles into acute, non-equilateral triangles.
        grid.nodes[0] -= 0.35 * grid.nodes[1]
    grid.compute_geometry()
    return grid


def _patch_errors(
    grid: pp.Grid, gradient: np.ndarray, boundary_type: str = "dirichlet",
) -> tuple[float, float]:
    mu, lam = 2.0, 3.0
    boundary = grid.get_all_boundary_faces()
    prescribed = boundary if boundary_type == "dirichlet" else boundary[::2]
    bc = pp.BoundaryConditionVectorial(grid, prescribed, "dir")
    layout = DofLayout.from_grid(grid, bc)
    operators = RetainedFaceOperators(grid, layout, mu=np.full(grid.num_cells, mu))
    elastic = ElasticTpsa(grid, bc, mu=mu, lam=lam)
    assert np.all(np.isfinite(elastic.matrices.cell_matrix.data))

    translation = np.array([[0.1], [-0.2]])
    u = gradient @ grid.cell_centers[:2] + translation
    faces = gradient @ grid.face_centers[:2] + translation
    rotation = np.full(grid.num_cells, mu * (gradient[0, 1] - gradient[1, 0]))
    pressure = np.full(grid.num_cells, lam * np.trace(gradient))
    cell_values = np.concatenate((u.ravel(order="F"), rotation, pressure))
    z = layout.pack(cell_values, faces[:, layout.free_faces])
    traction = operators.elastic_one_sided_traction(
        z, faces[:, layout.dirichlet_faces]
    )
    sigma = mu * (gradient + gradient.T) + lam * np.trace(gradient) * np.eye(2)
    normals = operators.side_normals * operators.side_areas[None, :]
    traction_error = float(np.max(np.abs(
        (traction - sigma @ normals) / operators.side_areas[None, :]
    )))
    strain = operators.green_gauss_reconstruction(faces)
    expected = np.zeros((3, 3, grid.num_cells))
    expected[:2, :2] = (0.5 * (gradient + gradient.T))[:, :, None]
    strain_error = float(np.max(np.abs(strain - expected)))
    return strain_error, traction_error


@pytest.mark.parametrize("boundary_type", ["dirichlet", "mixed"])
@pytest.mark.parametrize("gradient", [
    np.zeros((2, 2)),
    np.array([[0.0, -0.1], [0.1, 0.0]]),
    np.array([[0.1, 0.2], [-0.4, 0.5]]),
], ids=["translation", "rigid_rotation", "affine"])
def test_acute_triangular_patch_after_preprocessing(
    gradient: np.ndarray, boundary_type: str,
) -> None:
    grid = _triangles(acute=True)
    _, before = _patch_errors(grid, gradient, boundary_type)
    _, shift, changed = preprocess_simplex_grid(grid)
    np.testing.assert_array_equal(shift, np.ones(grid.num_cells))
    assert np.all(changed)
    strain_error, traction_error = _patch_errors(grid, gradient, boundary_type)
    assert strain_error < 1e-12
    assert traction_error < 1e-12
    if np.any(gradient):
        assert before > 1e-2


def test_right_triangles_improve_but_remain_nonorthogonal() -> None:
    grid = _triangles(acute=False)
    gradients = [
        np.array([[0.1, 0.2], [-0.4, 0.5]]),
        np.array([[0.0, -0.1], [0.1, 0.0]]),
    ]
    before = [_patch_errors(grid, gradient)[1] for gradient in gradients]
    _, shift, changed = preprocess_simplex_grid(grid)
    np.testing.assert_allclose(shift, 0.95)
    assert np.all(changed)
    for gradient, initial_error in zip(gradients, before, strict=True):
        strain_error, traction_error = _patch_errors(grid, gradient)
        assert strain_error < 1e-12
        # A capped center must not be mistaken for an orthogonal geometry.
        assert 1e-3 < traction_error < initial_error / 10


def test_preprocessing_preserves_geometry_and_is_idempotent() -> None:
    grid = _triangles(acute=True)
    names = ("nodes", "face_centers", "face_normals", "face_areas", "cell_volumes")
    saved = {name: getattr(grid, name).copy() for name in names}
    cell_faces, face_nodes = grid.cell_faces.copy(), grid.face_nodes.copy()
    centers, _, changed = preprocess_simplex_grid(grid)
    assert grid.cell_centers is centers
    assert np.all(changed)
    repeated, _, changed_again = preprocess_simplex_grid(grid)
    np.testing.assert_allclose(repeated, centers, atol=1e-14)
    assert not np.any(changed_again)
    for name, values in saved.items():
        np.testing.assert_array_equal(getattr(grid, name), values)
    for actual, reference in ((grid.cell_faces, cell_faces), (grid.face_nodes, face_nodes)):
        # PorePy can sort sparse indices while reading the cell-node map.
        assert (sps.csc_array(actual) - sps.csc_array(reference)).nnz == 0


@pytest.mark.parametrize("dim", [2, 3])
def test_native_simplex_matches_porepy_preprocessing(dim: int) -> None:
    grid: pp.Grid
    if dim == 2:
        grid = pp.TriangleGrid(
            np.array([[0.0, 1.0, 0.3], [0.0, 0.0, 0.8]]),
            np.array([[0], [1], [2]]),
        )
    else:
        grid = pp.TetrahedralGrid(
            np.array([[0.0, 1.0, 0.5, 0.5],
                      [0.0, 0.0, np.sqrt(3) / 2, np.sqrt(3) / 6],
                      [0.0, 0.0, 0.0, 0.9]]),
            np.arange(4).reshape((4, 1)),
        )
    grid.compute_geometry()
    expected = pp.grid_utils.compute_circumcenters(grid)
    actual = preprocess_simplex_grid(grid)
    for returned, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(returned, reference)


def test_obtuse_triangle_keeps_center_inside_without_claiming_exact_traction() -> None:
    grid = pp.TriangleGrid(
        np.array([[0.0, 1.0, 0.2], [0.0, 0.0, 0.2]]),
        np.array([[0], [1], [2]]),
    )
    grid.compute_geometry()
    _, shift, changed = preprocess_simplex_grid(grid)
    assert np.all((shift > 0) & (shift < 1))
    assert np.all(changed)
    incidence = grid.cell_faces.tocoo()
    normals = grid.face_normals[:, incidence.row] * incidence.data
    offsets = grid.face_centers[:, incidence.row] - grid.cell_centers[:, incidence.col]
    assert np.all(np.sum(normals * offsets, axis=0) > 0)
    strain_error, traction_error = _patch_errors(
        grid, np.array([[0.1, 0.2], [-0.4, 0.5]])
    )
    assert strain_error < 1e-12
    assert traction_error > 1e-3


def test_structured_tetrahedra_preprocessing_keeps_centers_inside() -> None:
    grid = pp.StructuredTetrahedralGrid(np.array([1, 1, 1]), np.ones(3))
    grid.compute_geometry()
    _, shift, changed = preprocess_simplex_grid(grid)
    assert np.all((shift > 0) & (shift < 1))
    assert np.all(changed)
    incidence = grid.cell_faces.tocoo()
    normals = grid.face_normals[:, incidence.row] * incidence.data
    offsets = grid.face_centers[:, incidence.row] - grid.cell_centers[:, incidence.col]
    assert np.all(np.sum(normals * offsets, axis=0) > 0)


def test_equilateral_center_is_not_misclassified_as_capped() -> None:
    grid = pp.TriangleGrid(
        np.array([[0.0, 1.0, 0.5], [0.0, 0.0, np.sqrt(3) / 2]]),
        np.array([[0], [1], [2]]),
    )
    grid.compute_geometry()
    original = grid.cell_centers.copy()
    centers, shift, changed = preprocess_simplex_grid(grid)
    np.testing.assert_allclose(centers, original, atol=1e-14)
    np.testing.assert_array_equal(shift, 0.0)
    assert not np.any(changed)
    strain_error, traction_error = _patch_errors(
        grid, np.array([[0.1, 0.2], [-0.4, 0.5]])
    )
    assert max(strain_error, traction_error) < 1e-12


def test_invalid_threshold_does_not_change_centers() -> None:
    grid = _triangles(acute=True)
    original = grid.cell_centers.copy()
    with pytest.raises(ValueError, match="Threshold"):
        preprocess_simplex_grid(grid, threshold=1.0)
    np.testing.assert_array_equal(grid.cell_centers, original)


def test_nonsimplex_grid_is_rejected_without_mutation() -> None:
    grid = pp.CartGrid(np.array([2, 2]))
    grid.compute_geometry()
    original = grid.cell_centers.copy()
    with pytest.raises(ValueError, match="simplex"):
        preprocess_simplex_grid(grid)
    np.testing.assert_array_equal(grid.cell_centers, original)


@pytest.mark.parametrize(("scale", "offset"), [(1e-6, 0.0), (1.0, 1e3), (10.0, 1e4)])
def test_normalized_preprocessing_handles_any_length_scale(scale: float, offset: float) -> None:
    """PorePy's absolute tol fails tiny cells (degeneracy check) and large
    coordinates (equidistance check); normalize=True handles both."""
    reference = _triangles(acute=True)
    centers, shift, _ = preprocess_simplex_grid(reference)
    grid = _triangles(acute=True)
    grid.nodes[:2] = grid.nodes[:2] * scale + offset
    grid.compute_geometry()
    with pytest.raises((AssertionError, ValueError)):
        preprocess_simplex_grid(grid)
    grid.compute_geometry()
    scaled, scaled_shift, changed = preprocess_simplex_grid(grid, normalize=True)
    assert grid.cell_centers is scaled and np.all(changed)
    np.testing.assert_allclose(scaled_shift, shift)
    np.testing.assert_allclose(scaled[:2], centers[:2] * scale + offset, rtol=0, atol=1e-13 * (scale + offset))


def test_normalized_preprocessing_matches_default_on_unit_grids() -> None:
    for acute in (True, False):
        grid, other = _triangles(acute=acute), _triangles(acute=acute)
        expected = preprocess_simplex_grid(grid)
        actual = preprocess_simplex_grid(other, normalize=True)
        np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=1e-14)
        np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(actual[2], expected[2])
    repeated = preprocess_simplex_grid(other, normalize=True)
    assert not np.any(repeated[2])
