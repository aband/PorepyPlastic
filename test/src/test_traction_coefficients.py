"""Fixed side stiffness and complementary rotation transfer in 2D and 3D."""

from typing import cast

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]

from core.layout import DofLayout, FloatArray
from core.operators import RetainedFaceOperators


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def grid(request: pytest.FixtureRequest) -> pp.Grid:
    # Two boxes of widths 1 and 2 along x; the interface distances are 1/2 and 1.
    # Face measures are deliberately nonunit to detect accidental area factors.
    result = pp.TensorGrid(
        np.array([0.0, 1.0, 3.0]),
        np.array([0.0, 2.0]),
        np.array([0.0, 4.0]) if request.param == 3 else None,
    )
    result.compute_geometry()
    return result


def _layout(grid: pp.Grid, boundary_type: str = "mixed") -> DofLayout:
    boundary = grid.get_all_boundary_faces()
    prescribed = {
        "dirichlet": boundary,
        "neumann": boundary[:0],
        "mixed": boundary[::2][::-1],
    }[boundary_type]
    return DofLayout.from_grid(grid, pp.BoundaryConditionVectorial(grid, prescribed, "dir"))


def _interior_face(grid: pp.Grid) -> int:
    faces = np.setdiff1d(np.arange(grid.num_faces), grid.get_all_boundary_faces())
    assert faces.size == 1
    return int(faces[0])


def test_coefficients_are_optional(grid: pp.Grid) -> None:
    operators = RetainedFaceOperators(grid, _layout(grid))
    assert operators.mu is None
    assert operators.side_stiffness is None
    assert operators.rotation_transfer is None
    np.testing.assert_array_equal(
        operators.green_gauss_reconstruction(np.zeros((grid.dim, grid.num_faces))),
        np.zeros((3, 3, grid.num_cells)),
    )


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed"])
@pytest.mark.parametrize(
    ("moduli", "interface_weights"),
    [
        ((2.0, 2.0), (1.0 / 3.0, 2.0 / 3.0)),
        ((2.0, 3.0), (3.0 / 7.0, 4.0 / 7.0)),
        ((2.0, 4.0), (0.5, 0.5)),
    ],
    ids=["uniform_mu", "heterogeneous_mu", "equal_side_stiffness"],
)
def test_side_stiffness_and_complementary_rotation_weights(
    grid: pp.Grid, boundary_type: str,
    moduli: tuple[float, float], interface_weights: tuple[float, float],
) -> None:
    mu = np.array(moduli)
    layout = _layout(grid, boundary_type)
    operators = RetainedFaceOperators(grid, layout, mu=mu)
    assert operators.mu is not None
    assert operators.side_stiffness is not None
    assert operators.rotation_transfer is not None
    np.testing.assert_array_equal(operators.mu, mu)
    assert operators.side_stiffness.shape == (operators.num_sides,)
    assert operators.side_stiffness.dtype == np.float64

    # For each box side: delta = width / 2, so a = 4 * mu / width.
    widths = np.array([[1.0, 2.0, 4.0], [2.0, 2.0, 4.0]])
    for side, (face, cell) in enumerate(zip(operators.side_faces, operators.side_cells, strict=True)):
        axis = int(np.argmax(np.abs(grid.face_normals[:grid.dim, face])))
        np.testing.assert_allclose(operators.side_stiffness[side], 4.0 * mu[cell] / widths[cell, axis])

    expected = np.zeros((grid.num_faces, grid.num_cells))
    # Boundary rows use their own cell irrespective of boundary-condition type.
    incidence = grid.cell_faces.tocoo()
    boundary = grid.get_all_boundary_faces()
    for face, cell in zip(incidence.row, incidence.col, strict=True):
        if face in boundary:
            expected[face, cell] = 1.0
    expected[_interior_face(grid)] = interface_weights
    matrix = operators.rotation_transfer.weights
    assert isinstance(matrix, sps.csr_array)
    assert matrix.shape == expected.shape
    np.testing.assert_allclose(matrix.toarray(), expected, rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1.0, atol=1e-14)
    assert np.all(matrix.data >= 0)

    # The same scalar weights apply independently to the 1 or 3 rotation components.
    rotation = np.arange(layout.rotation_dim * grid.num_cells, dtype=float).reshape(
        (layout.rotation_dim, grid.num_cells), order="F",
    ) + 0.5
    saved_rotation = rotation.copy()
    face_rotation = operators.rotation_transfer.apply(rotation)
    assert face_rotation.shape == (layout.rotation_dim, grid.num_faces)
    np.testing.assert_allclose(face_rotation, rotation @ expected.T, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(rotation, saved_rotation)
    assert not np.shares_memory(face_rotation, rotation)
    np.testing.assert_allclose(
        operators.rotation_transfer.apply(np.ones_like(rotation)), 1.0, atol=1e-14,
    )
    if grid.dim == 2:
        np.testing.assert_allclose(operators.rotation_transfer.apply(rotation[0]), face_rotation[0])


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann"])
def test_single_cell_has_only_unit_boundary_weights(grid: pp.Grid, boundary_type: str) -> None:
    single = pp.CartGrid(np.ones(grid.dim, dtype=int))
    single.compute_geometry()
    operators = RetainedFaceOperators(single, _layout(single, boundary_type), mu=np.array([3.0]))
    assert operators.side_stiffness is not None
    assert operators.rotation_transfer is not None
    np.testing.assert_allclose(operators.side_stiffness, 12.0)
    np.testing.assert_array_equal(
        operators.rotation_transfer.weights.toarray(), np.ones((single.num_faces, 1)),
    )


def test_coefficients_preserve_and_copy_strided_input(grid: pp.Grid) -> None:
    storage = np.array([2.0, -99.0, 3.0, -99.0])
    mu = storage[::2]
    mu.setflags(write=False)
    original = storage.copy()
    operators = RetainedFaceOperators(grid, _layout(grid), mu=mu)
    assert operators.mu is not None
    assert operators.side_stiffness is not None
    assert operators.rotation_transfer is not None
    saved_stiffness = operators.side_stiffness.copy()
    saved_weights = operators.rotation_transfer.weights.toarray()
    np.testing.assert_array_equal(storage, original)
    assert not np.shares_memory(operators.mu, storage)
    assert not np.shares_memory(operators.side_stiffness, storage)
    storage[::2] = 100.0
    np.testing.assert_array_equal(operators.mu, original[::2])
    np.testing.assert_array_equal(operators.side_stiffness, saved_stiffness)
    np.testing.assert_array_equal(operators.rotation_transfer.weights.toarray(), saved_weights)


def test_grid_scaling_changes_stiffness_but_preserves_weights(grid: pp.Grid) -> None:
    layout = _layout(grid)
    mu = np.array([2.0, 3.0])
    original = RetainedFaceOperators(grid, layout, mu=mu)
    grid.nodes *= 3.0
    grid.compute_geometry()
    scaled = RetainedFaceOperators(grid, layout, mu=mu)
    assert original.side_stiffness is not None and scaled.side_stiffness is not None
    assert original.rotation_transfer is not None and scaled.rotation_transfer is not None
    np.testing.assert_allclose(scaled.side_stiffness, original.side_stiffness / 3.0)
    np.testing.assert_allclose(
        scaled.rotation_transfer.weights.toarray(), original.rotation_transfer.weights.toarray(),
    )


def test_weight_normalization_avoids_overflow(grid: pp.Grid) -> None:
    # Each interior stiffness is 1e308: finite individually, but their sum overflows.
    mu = np.array([2.5e307, 5.0e307])
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        operators = RetainedFaceOperators(grid, _layout(grid), mu=mu)
    assert operators.side_stiffness is not None
    assert operators.rotation_transfer is not None
    face = _interior_face(grid)
    interior_stiffness = operators.side_stiffness[operators.side_faces == face]
    assert np.all(np.isfinite(operators.side_stiffness))
    assert np.all(interior_stiffness > np.finfo(np.float64).max / 2)
    np.testing.assert_allclose(
        operators.rotation_transfer.weights.toarray()[face], [0.5, 0.5], rtol=1e-14,
    )


def test_large_contrast_retains_small_complementary_weight(grid: pp.Grid) -> None:
    operators = RetainedFaceOperators(grid, _layout(grid), mu=np.array([2.0, 1e200]))
    assert operators.rotation_transfer is not None
    # Interior stiffnesses are 8 and 2e200. A subtraction from one would lose
    # the small weight; compare with zero absolute tolerance to detect that.
    np.testing.assert_allclose(
        operators.rotation_transfer.weights.toarray()[_interior_face(grid)],
        [1.0, 4e-200], rtol=1e-14, atol=0,
    )


@pytest.mark.parametrize("invalid", ["scalar", "short", "long", "column"])
def test_mu_must_be_a_cellwise_vector(grid: pp.Grid, invalid: str) -> None:
    nc = grid.num_cells
    mu = {
        "scalar": np.array(2.0), "short": np.ones(nc - 1),
        "long": np.ones(nc + 1), "column": np.ones((nc, 1)),
    }[invalid]
    with pytest.raises(ValueError, match="Shear modulus must have"):
        RetainedFaceOperators(grid, _layout(grid), mu=mu)


@pytest.mark.parametrize("invalid", ["zero", "negative", "nan", "inf", "negative_inf", "complex"])
def test_mu_must_be_real_finite_and_positive(grid: pp.Grid, invalid: str) -> None:
    if invalid == "complex":
        mu = cast(FloatArray, np.full(grid.num_cells, 1.0 + 1j))
        message = "Shear modulus must be real"
    else:
        mu = np.ones(grid.num_cells)
        mu[0] = {"zero": 0.0, "negative": -1.0, "nan": np.nan,
                 "inf": np.inf, "negative_inf": -np.inf}[invalid]
        message = "finite and positive"
    with pytest.raises(ValueError, match=message):
        RetainedFaceOperators(grid, _layout(grid), mu=mu)


def test_nonfinite_derived_stiffness_is_rejected(grid: pp.Grid) -> None:
    mu = np.full(grid.num_cells, np.finfo(np.float64).max)
    with pytest.raises(ValueError, match="Stiffness must be finite and positive"):
        RetainedFaceOperators(grid, _layout(grid), mu=mu)
