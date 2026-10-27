"""Dimension-aware Green–Gauss gradients, symmetric strains, and geometry."""

from typing import cast

import numpy as np
from numpy.typing import NDArray
import porepy as pp
import pytest

from utils.reconstruction import (
    gradient_green_gauss,
    plane_strain_from_gradient,
    strain_from_gradient,
    validate_reconstruction_geometry,
)


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def grid(request: pytest.FixtureRequest) -> pp.Grid:
    coordinates = [
        np.array([0.0, 0.3, 1.4]),
        np.array([0.0, 0.2, 0.9]),
        np.array([0.0, 0.4, 1.2]),
    ]
    result = pp.TensorGrid(
        coordinates[0], coordinates[1], coordinates[2] if request.param == 3 else None,
    )
    result.compute_geometry()
    return result


@pytest.mark.parametrize("field", ["translation", "rotation", "affine"])
@pytest.mark.parametrize("rotate_grid", [False, True], ids=["aligned", "rotated"])
def test_affine_gradient_and_strain(
    grid: pp.Grid, field: str, rotate_grid: bool,
) -> None:
    if rotate_grid:
        c, s = np.cos(0.4), np.sin(0.4)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        if grid.dim == 3:
            rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ rotation
        grid.nodes[:] = rotation @ grid.nodes
        grid.compute_geometry()

    nd, nc = grid.dim, grid.num_cells
    matrix = np.array([[0.1, 0.2, -0.3], [-0.4, 0.5, 0.6], [0.7, -0.8, 0.9]])[:nd, :nd]
    if field == "translation":
        matrix = np.zeros((nd, nd))
    elif field == "rotation":
        matrix = 0.5 * (matrix - matrix.T)
    face_values = matrix @ grid.face_centers[:nd] + np.arange(1, nd + 1)[:, None] / 7
    saved_values = face_values.copy()

    gradient = gradient_green_gauss(grid, face_values)
    expected_gradient = np.repeat(matrix[:, :, None], nc, axis=2)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-13)
    assert gradient.shape == (nd, nd, nc)
    expected_strain = np.zeros((3, 3, nc))
    expected_strain[:nd, :nd] = 0.5 * (matrix + matrix.T)[:, :, None]
    epsilon = strain_from_gradient(gradient)
    np.testing.assert_allclose(epsilon, expected_strain, rtol=1e-12, atol=1e-13)
    assert not np.shares_memory(epsilon, gradient)
    assert not np.shares_memory(gradient, face_values)
    np.testing.assert_array_equal(face_values, saved_values)
    if nd == 2:
        np.testing.assert_array_equal(epsilon, plane_strain_from_gradient(gradient))
        np.testing.assert_array_equal(epsilon[2], 0.0)
        np.testing.assert_array_equal(epsilon[:, 2], 0.0)


def test_arbitrary_face_values_match_surface_sum_and_conserve_gradient(grid: pp.Grid) -> None:
    rng = np.random.default_rng(19)
    storage = rng.normal(size=(grid.dim, 2 * grid.num_faces))
    values = storage[:, ::2]
    assert not values.flags.c_contiguous
    original = storage.copy()
    actual = gradient_green_gauss(grid, values)
    expected = np.zeros_like(actual)
    incidence = grid.cell_faces.tocoo()
    for face, cell, sign in zip(incidence.row, incidence.col, incidence.data, strict=True):
        expected[:, :, cell] += np.outer(
            values[:, face], sign * grid.face_normals[:grid.dim, face]
        ) / grid.cell_volumes[cell]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    # Internal faces cancel from the domain integral, leaving the boundary only.
    boundary = grid.get_all_boundary_faces()
    signs = np.asarray(grid.cell_faces[boundary].sum(axis=1)).ravel()
    surface_sum = np.einsum(
        "if,jf,f->ij", values[:, boundary], grid.face_normals[:grid.dim, boundary], signs,
    )
    np.testing.assert_allclose(
        np.sum(actual * grid.cell_volumes, axis=2), surface_sum, rtol=1e-13, atol=1e-13,
    )
    np.testing.assert_array_equal(storage, original)


def test_reconstruction_is_invariant_under_face_orientation_reversal(grid: pp.Grid) -> None:
    values = np.random.default_rng(23).normal(size=(grid.dim, grid.num_faces))
    expected = gradient_green_gauss(grid, values)
    signs = np.ones(grid.num_faces)
    signs[::2] = -1
    grid.face_normals *= signs
    grid.cell_faces = grid.cell_faces.multiply(signs[:, None]).tocsc()
    np.testing.assert_allclose(gradient_green_gauss(grid, values), expected, atol=1e-14)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("num_cells", [0, 1, 4])
def test_strain_symmetrizes_gradients_and_preserves_inputs(dim: int, num_cells: int) -> None:
    storage = np.random.default_rng(27).normal(size=(dim, dim, 2 * num_cells))
    gradient = storage[:, :, ::2]
    saved = storage.copy()
    expected = np.zeros((3, 3, num_cells))
    for i in range(dim):
        for j in range(dim):
            expected[i, j] = (gradient[i, j] + gradient[j, i]) / 2
    epsilon = strain_from_gradient(gradient)
    np.testing.assert_array_equal(epsilon, expected)
    assert epsilon.dtype == np.float64
    assert not np.shares_memory(epsilon, gradient)
    epsilon.fill(999.0)
    np.testing.assert_array_equal(storage, saved)


def test_plane_strain_wrapper_keeps_2d_only_contract() -> None:
    with pytest.raises(ValueError, match=r"\(2, 2, nc\)"):
        plane_strain_from_gradient(np.zeros((3, 3, 2)))


@pytest.mark.parametrize("invalid", ["flat", "rectangular", "four_dimensional", "nan", "inf", "complex"])
def test_strain_rejects_invalid_gradients(invalid: str) -> None:
    gradient = {
        "flat": np.zeros(4),
        "rectangular": np.zeros((2, 3, 1)),
        "four_dimensional": np.zeros((4, 4, 1)),
        "nan": np.full((2, 2, 1), np.nan),
        "inf": np.full((2, 2, 1), np.inf),
        "complex": np.full((2, 2, 1), 1j),
    }[invalid]
    for reconstruct in (strain_from_gradient, plane_strain_from_gradient):
        with pytest.raises(ValueError, match="gradient"):
            reconstruct(cast(NDArray[np.float64], gradient))


@pytest.mark.parametrize("invalid", ["flat", "components", "short", "nan", "inf", "complex"])
def test_gradient_requires_finite_displacement_on_every_face(grid: pp.Grid, invalid: str) -> None:
    nd, nf = grid.dim, grid.num_faces
    values = {
        "flat": np.zeros(nd * nf),
        "components": np.zeros((nd + 1, nf)),
        "short": np.zeros((nd, nf - 1)),
        "nan": np.zeros((nd, nf)),
        "inf": np.zeros((nd, nf)),
        "complex": np.full((nd, nf), 1j),
    }[invalid]
    if invalid in ("nan", "inf"):
        interior = np.setdiff1d(np.arange(nf), grid.get_all_boundary_faces())
        values[0, interior[0]] = np.nan if invalid == "nan" else np.inf
    with pytest.raises(ValueError, match="[Ff]ace displacement"):
        gradient_green_gauss(grid, cast(NDArray[np.float64], values))


def test_geometry_validation_preserves_data(grid: pp.Grid) -> None:
    arrays = [grid.cell_volumes, grid.cell_centers, grid.face_areas,
              grid.face_centers, grid.face_normals, grid.cell_faces.data]
    originals = [array.copy() for array in arrays]
    validate_reconstruction_geometry(grid)
    for array, original in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("dim", [2, 3])
def test_geometry_must_be_computed(dim: int) -> None:
    grid = pp.CartGrid(np.full(dim, 2, dtype=int))
    with pytest.raises(ValueError, match="compute_geometry"):
        validate_reconstruction_geometry(grid)
    with pytest.raises(ValueError, match="compute_geometry"):
        gradient_green_gauss(grid, np.zeros((dim, grid.num_faces)))


@pytest.mark.parametrize("name", ["cell_volumes", "cell_centers", "face_areas", "face_centers", "face_normals"])
@pytest.mark.parametrize("defect", ["missing", "shape", "nan", "complex"])
def test_geometry_validation_rejects_invalid_arrays(
    grid: pp.Grid, name: str, defect: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if defect == "missing":
        monkeypatch.delattr(grid, name)
    else:
        values = getattr(grid, name).copy()
        if defect == "shape":
            values = values[:-1]
        elif defect == "nan":
            values.flat[0] = np.nan
        else:
            values = values.astype(complex)
        monkeypatch.setattr(grid, name, values)
    with pytest.raises(ValueError, match=name):
        validate_reconstruction_geometry(grid)


@pytest.mark.parametrize("name", ["cell_volumes", "face_areas"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_geometry_measures_must_be_positive(grid: pp.Grid, name: str, value: float) -> None:
    getattr(grid, name)[0] = value
    with pytest.raises(ValueError, match="positive"):
        validate_reconstruction_geometry(grid)


@pytest.mark.parametrize("name", ["num_cells", "num_faces"])
def test_geometry_must_be_nonempty(
    grid: pp.Grid, name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grid, name, 0)
    with pytest.raises(ValueError, match="nonempty"):
        validate_reconstruction_geometry(grid)


@pytest.mark.parametrize("defect", ["missing", "shape", "nan"])
def test_geometry_connectivity_must_match_grid(
    grid: pp.Grid, defect: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if defect == "missing":
        monkeypatch.delattr(grid, "cell_faces")
    elif defect == "shape":
        grid.cell_faces = grid.cell_faces[:, :-1]
    else:
        grid.cell_faces = grid.cell_faces.astype(float)
        grid.cell_faces.data[0] = np.nan
    with pytest.raises(ValueError, match="cell_faces"):
        validate_reconstruction_geometry(grid)


def test_geometry_rejects_unsupported_dimension() -> None:
    grid = pp.CartGrid(np.array([2]))
    grid.compute_geometry()
    with pytest.raises(ValueError, match="2D or 3D"):
        validate_reconstruction_geometry(grid)


def test_2d_geometry_requires_xy_coordinate_convention() -> None:
    grid = pp.CartGrid(np.array([2, 2]))
    # A tilted surface requires an explicit local-coordinate transformation.
    grid.nodes[2] = 0.5 * grid.nodes[0]
    grid.compute_geometry()
    with pytest.raises(ValueError, match="parallel to the xy plane"):
        gradient_green_gauss(grid, np.zeros((2, grid.num_faces)))
