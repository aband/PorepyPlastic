"""Configurable cell-to-face transfer, tensor layout, and shared-face checks."""

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.grid import Grid
from porepy.grids.structured import CartGrid

from coupling.transfer import CellToFaceTransfer


@pytest.fixture
def grid() -> Grid:
    grid = CartGrid(np.array([2, 1]), physdims=np.array([2.0, 0.5]))
    grid.compute_geometry()
    return grid


def interior_face(grid: Grid) -> int:
    return int(np.flatnonzero(np.diff(grid.cell_faces.tocsr().indptr) == 2)[0])


def test_default_averaging_and_one_sided_boundaries(grid: Grid) -> None:
    transfer = CellToFaceTransfer(grid)
    result = transfer.apply(np.array([2.0, 8.0]))
    assert transfer.weights.shape == (grid.num_faces, grid.num_cells)
    face = interior_face(grid)
    assert result[face] == 5.0
    np.testing.assert_array_equal(transfer.weights[[face], :].toarray(), [[0.5, 0.5]])
    for boundary in grid.get_all_boundary_faces():
        cells = grid.cell_faces.tocsr().indices[
            grid.cell_faces.tocsr().indptr[boundary]:grid.cell_faces.tocsr().indptr[boundary + 1]
        ]
        assert cells.size == 1
        assert result[boundary] == [2.0, 8.0][cells[0]]
        assert transfer.weights[boundary, cells[0]] == 1.0


def test_custom_rule_receives_cell_order_and_runs_only_once(grid: Grid) -> None:
    calls: list[int] = []

    def custom(g: Grid, face: int, cells: NDArray[np.int64]) -> NDArray[np.float64]:
        assert g is grid
        np.testing.assert_array_equal(cells, [0, 1])
        calls.append(face)
        return np.array([0.25, 0.75])

    transfer = CellToFaceTransfer.from_rule(grid, custom)
    face = interior_face(grid)
    assert calls == [face]  # The rule is not called on boundary faces.
    for _ in range(3):
        assert transfer.apply(np.array([2.0, 8.0]))[face] == 6.5
    assert calls == [face]


def test_explicit_matrix_is_copied_and_editing_requires_a_new_transfer(grid: Grid) -> None:
    original = CellToFaceTransfer(grid)
    weights = original.weights.tolil()
    face = interior_face(grid)
    weights[face, :] = [0.2, 0.8]
    supplied = weights.tocsr()
    modified = CellToFaceTransfer(grid, weights=supplied)
    supplied.data[:] = 0.0
    exported = modified.weights
    exported.data[:] = 0.0
    assert original.apply(np.array([2.0, 8.0]))[face] == 5.0
    assert modified.apply(np.array([2.0, 8.0]))[face] == pytest.approx(6.8)


@pytest.mark.parametrize("component_shape", [(), (2,), (3, 3)])
def test_constant_fields_and_component_order(
    grid: Grid, component_shape: tuple[int, ...],
) -> None:
    transfer = CellToFaceTransfer(grid)
    components = np.arange(np.prod(component_shape, dtype=int), dtype=float).reshape(component_shape) + 2
    constant = np.repeat(components[..., None], grid.num_cells, axis=-1)
    expected = np.repeat(components[..., None], grid.num_faces, axis=-1)
    np.testing.assert_allclose(transfer.apply(constant), expected)
    varying = constant.copy()
    varying[..., 1] *= 3.0
    result = transfer.apply(varying)
    assert result.shape == (*component_shape, grid.num_faces)
    np.testing.assert_allclose(result[..., interior_face(grid)], 2.0 * components)


def test_transfer_preserves_symmetric_full_3d_tensor_and_input(grid: Grid) -> None:
    transfer = CellToFaceTransfer(grid)
    tensor = np.array([[2.0, 3.0, 4.0], [3.0, 5.0, 6.0], [4.0, 6.0, 7.0]])
    values = np.stack((tensor, 3 * tensor), axis=-1)
    saved = values.copy()
    values.flags.writeable = False
    result = transfer.apply(values)
    np.testing.assert_array_equal(result, result.swapaxes(0, 1))
    np.testing.assert_allclose(result[:, :, interior_face(grid)], 2 * tensor)
    assert not np.shares_memory(result, values)
    result[:] = 0.0
    np.testing.assert_array_equal(values, saved)


def test_transfer_ignores_face_orientation(grid: Grid) -> None:
    expected = CellToFaceTransfer(grid).weights.toarray()
    grid.cell_faces.data *= -1
    np.testing.assert_array_equal(CellToFaceTransfer(grid).weights.toarray(), expected)


def test_single_cell_uses_one_sided_transfer_without_calling_rule() -> None:
    grid = CartGrid(np.array([1, 1]))

    def forbidden(g: Grid, face: int, cells: NDArray[np.int64]) -> NDArray[np.float64]:
        raise AssertionError("No interior face exists.")

    transfer = CellToFaceTransfer.from_rule(grid, forbidden)
    np.testing.assert_array_equal(transfer.apply(np.array([9.0])), np.full(grid.num_faces, 9.0))


def test_shared_interior_correction_cancels_between_cells(grid: Grid) -> None:
    transfer = CellToFaceTransfer(grid)
    corrections = np.stack((np.diag([2.0, 3.0, 4.0]), np.diag([6.0, 7.0, 8.0])), axis=-1)
    face_values = transfer.apply(corrections)
    force = np.einsum("ijf,jf->if", face_values[:2, :2], grid.face_normals[:2])
    face = interior_face(grid)
    expected = np.diag([4.0, 5.0]) @ grid.face_normals[:2, face]
    np.testing.assert_allclose(force[:, face], expected)  # W includes no area factor.
    force[:, grid.get_all_boundary_faces()] = 0.0
    balance = np.asarray(grid.divergence(dim=2) @ force.ravel(order="F")).reshape((2, -1), order="F")
    assert np.linalg.norm(balance[:, 0]) > 0.0
    np.testing.assert_allclose(balance[:, 0], -balance[:, 1])


@pytest.mark.parametrize("defect", ["shape", "nan", "inf", "negative", "sum", "nonadjacent"])
def test_invalid_explicit_weights_are_rejected(grid: Grid, defect: str) -> None:
    matrix = CellToFaceTransfer(grid).weights.tolil()
    face = interior_face(grid)
    match = ""
    if defect == "shape":
        matrix = matrix[:-1]
        match = "shape"
    elif defect in ("nan", "inf"):
        matrix[face, 0] = np.nan if defect == "nan" else np.inf
        match = "finite"
    elif defect == "negative":
        matrix[face, :] = [-0.1, 1.1]
        match = "nonnegative"
    elif defect == "sum":
        matrix[face, :] = [0.1, 0.2]
        match = "sum to one"
    else:
        boundary = grid.get_all_boundary_faces()[0]
        cell = matrix.tocsr().indices[matrix.tocsr().indptr[boundary]]
        matrix[boundary, :] = 0.0
        matrix[boundary, 1 - cell] = 1.0
        match = "adjacent"
    with pytest.raises(ValueError, match=match):
        CellToFaceTransfer(grid, matrix.tocsr())


@pytest.mark.parametrize("value", [np.array([0.2]), np.array([[0.5, 0.5]])])
def test_rule_must_return_one_weight_per_adjacent_cell(grid: Grid, value: NDArray[np.float64]) -> None:
    def malformed(g: Grid, face: int, cells: NDArray[np.int64]) -> NDArray[np.float64]:
        return value

    with pytest.raises(ValueError, match="Weight rule.*shape"):
        CellToFaceTransfer.from_rule(grid, malformed)


@pytest.mark.parametrize("count", [0, 3])
def test_unsupported_face_connectivity_is_rejected(count: int) -> None:
    grid = CartGrid(np.array([3, 1]))
    incidence = grid.cell_faces.tolil()
    incidence[0, :] = 0.0 if count == 0 else 1.0
    grid.cell_faces = incidence.tocsc()
    with pytest.raises(ValueError, match="one or two"):
        CellToFaceTransfer(grid)


@pytest.mark.parametrize("values", [np.array(1.0), np.zeros((3, 3, 3)), np.array([np.nan, 0.0]), np.array([0.0, np.inf])])
def test_invalid_cell_values_are_rejected(grid: Grid, values: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="finite cell values.*last axis"):
        CellToFaceTransfer(grid).apply(values)
