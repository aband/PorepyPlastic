"""Retained-face assembly, strain reconstruction, and side geometry in 2D/3D."""

from dataclasses import replace
from typing import cast

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]

from core.layout import DofLayout, FloatArray
from core.operators import RetainedFaceOperators
from utils.reconstruction import gradient_green_gauss, strain_from_gradient


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def grid(request: pytest.FixtureRequest) -> pp.Grid:
    result = pp.TensorGrid(
        np.array([0.0, 0.3, 1.4]),
        np.array([0.0, 0.2, 0.9]),
        np.array([0.0, 0.4, 1.2]) if request.param == 3 else None,
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
    bc = pp.BoundaryConditionVectorial(grid, prescribed, "dir")
    return DofLayout.from_grid(grid, bc)


@pytest.fixture
def operators(grid: pp.Grid) -> RetainedFaceOperators:
    return RetainedFaceOperators(grid, _layout(grid))


def _mandel_reference(strain: FloatArray, dim: int) -> FloatArray:
    # State the component convention independently of the production helpers.
    components = [strain[0, 0], strain[1, 1]]
    if dim == 3:
        components += [strain[2, 2], np.sqrt(2.0) * strain[1, 2],
                       np.sqrt(2.0) * strain[0, 2]]
    components.append(np.sqrt(2.0) * strain[0, 1])
    return np.stack(components).ravel(order="F")


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed", "single_cell"])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_face_increment_ordering_and_input_preservation(
    grid: pp.Grid, boundary_type: str, strided: bool,
) -> None:
    if boundary_type == "single_cell":
        grid = pp.CartGrid(np.ones(grid.dim, dtype=int))
        grid.compute_geometry()
        boundary_type = "dirichlet"
    layout = _layout(grid, boundary_type)
    operators = RetainedFaceOperators(grid, layout)
    nd, nf = grid.dim, grid.num_faces
    expected = (100.0 + 10 * np.arange(nf)[None, :]
                + np.arange(nd)[:, None]).astype(np.float64)
    # Construct the packed vector without relying on layout.pack().
    face_block = np.array([
        expected[component, face]
        for face in layout.free_faces for component in range(nd)
    ])
    z = np.concatenate((np.full(layout.cell_size, -999.0), face_block))
    prescribed = expected[:, layout.dirichlet_faces].copy()
    if strided:
        z_storage = np.zeros(2 * z.size)
        z_storage[::2] = z
        z = z_storage[::2]
        prescribed_storage = np.zeros((nd, 2 * prescribed.shape[1]))
        prescribed_storage[:, ::2] = prescribed
        prescribed = prescribed_storage[:, ::2]
    saved_z, saved_prescribed = z.copy(), prescribed.copy()

    actual = operators.face_displacement_increment(z, prescribed)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (nd, nf)
    assert actual.dtype == np.float64
    assert actual.flags.owndata
    assert not np.shares_memory(actual, z)
    assert not np.shares_memory(actual, prescribed)
    actual.fill(-1.0)
    np.testing.assert_array_equal(z, saved_z)
    np.testing.assert_array_equal(prescribed, saved_prescribed)


@pytest.mark.parametrize("field", ["translation", "rotation", "affine"])
@pytest.mark.parametrize("rotate_grid", [False, True], ids=["aligned", "rotated"])
def test_affine_reconstruction_is_exact(
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
    gradient = np.array([
        [0.1, 0.2, -0.3], [-0.4, 0.5, 0.6], [0.7, -0.8, 0.9],
    ])[:nd, :nd]
    if field == "translation":
        gradient = np.zeros((nd, nd))
    elif field == "rotation":
        gradient = 0.5 * (gradient - gradient.T)
    expected = np.zeros((3, 3, nc))
    expected[:nd, :nd] = (0.5 * (gradient + gradient.T))[:, :, None]
    values = gradient @ grid.face_centers[:nd] + np.arange(1, nd + 1)[:, None] / 7
    operators = RetainedFaceOperators(grid, _layout(grid))
    layout = operators.layout
    z = layout.pack(np.zeros(layout.cell_size), values[:, layout.free_faces])
    assembled = operators.face_displacement_increment(z, values[:, layout.dirichlet_faces])

    actual = operators.green_gauss_reconstruction(assembled)
    assert actual.shape == (3, 3, nc)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        operators.B @ assembled.ravel(order="F"), _mandel_reference(expected, nd),
        rtol=1e-12, atol=1e-13,
    )
    if nd == 2:
        np.testing.assert_array_equal(actual[2], 0.0)
        np.testing.assert_array_equal(actual[:, 2], 0.0)


def test_random_reconstruction_matches_surface_sum_and_shared_utilities(
    grid: pp.Grid, operators: RetainedFaceOperators,
) -> None:
    storage = np.random.default_rng(12).normal(size=(grid.dim, 2 * grid.num_faces))
    values = storage[:, ::2]
    saved = storage.copy()
    expected_gradient = np.zeros((grid.dim, grid.dim, grid.num_cells))
    incidence = grid.cell_faces.tocoo()
    for face, cell, sign in zip(incidence.row, incidence.col, incidence.data, strict=True):
        expected_gradient[:, :, cell] += np.outer(
            values[:, face], sign * grid.face_normals[:grid.dim, face],
        ) / grid.cell_volumes[cell]
    expected = np.zeros((3, 3, grid.num_cells))
    expected[:grid.dim, :grid.dim] = 0.5 * (
        expected_gradient + expected_gradient.swapaxes(0, 1)
    )

    actual = operators.green_gauss_reconstruction(values)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        actual, strain_from_gradient(gradient_green_gauss(grid, values)),
        rtol=1e-13, atol=1e-13,
    )
    np.testing.assert_allclose(
        operators.B @ values.ravel(order="F"), _mandel_reference(expected, grid.dim),
        rtol=1e-13, atol=1e-13,
    )
    assert actual.flags.owndata
    assert not np.shares_memory(actual, values)
    actual.fill(999.0)
    np.testing.assert_array_equal(storage, saved)


def test_strain_matrix_is_sparse_local_and_cached(
    grid: pp.Grid, operators: RetainedFaceOperators, monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = operators.B
    num_components = 3 if grid.dim == 2 else 6
    assert isinstance(matrix, sps.csr_array)
    assert matrix.shape == (num_components * grid.num_cells, grid.dim * grid.num_faces)
    assert np.all(np.isfinite(matrix.data))
    assert np.all(matrix.data != 0)
    # A cell strain may only depend on displacement at that cell's own faces.
    rows, cols = matrix.nonzero()
    for row, col in zip(rows, cols, strict=True):
        assert grid.cell_faces[col // grid.dim, row // num_components] != 0

    def unexpected_rebuild() -> None:
        pytest.fail("Reconstruction must reuse the cached strain matrix.")

    monkeypatch.setattr(operators, "_build_strain_matrix", unexpected_rebuild)
    for amplitude in (0.0, 1.0):
        values = np.full((grid.dim, grid.num_faces), amplitude)
        np.testing.assert_allclose(operators.green_gauss_reconstruction(values), 0, atol=1e-13)
        assert operators.B is matrix


def test_reversing_face_orientations_preserves_reconstruction(grid: pp.Grid) -> None:
    layout = _layout(grid)
    original = RetainedFaceOperators(grid, layout)
    values = np.random.default_rng(21).normal(size=(grid.dim, grid.num_faces))
    expected = original.green_gauss_reconstruction(values)
    signs = np.ones(grid.num_faces)
    signs[::2] = -1
    grid.face_normals *= signs
    grid.cell_faces = grid.cell_faces.multiply(signs[:, None]).tocsc()
    reversed_operators = RetainedFaceOperators(grid, layout)
    np.testing.assert_allclose(
        reversed_operators.green_gauss_reconstruction(values), expected, atol=1e-13,
    )
    np.testing.assert_array_equal(reversed_operators.side_faces, original.side_faces)
    np.testing.assert_array_equal(reversed_operators.side_cells, original.side_cells)
    np.testing.assert_array_equal(
        reversed_operators.side_signs, original.side_signs * signs[original.side_faces],
    )
    for attribute in ("side_areas", "side_normals", "side_distances"):
        np.testing.assert_allclose(
            getattr(reversed_operators, attribute), getattr(original, attribute), atol=1e-13,
        )


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed", "single_cell"])
def test_free_and_prescribed_column_split_and_strain_derivative(
    grid: pp.Grid, boundary_type: str,
) -> None:
    if boundary_type == "single_cell":
        grid = pp.CartGrid(np.ones(grid.dim, dtype=int))
        grid.compute_geometry()
        boundary_type = "dirichlet"
    layout = _layout(grid, boundary_type)
    operators = RetainedFaceOperators(grid, layout)
    rng = np.random.default_rng(32)
    z = rng.normal(size=layout.size)
    prescribed = rng.normal(size=(grid.dim, layout.dirichlet_faces.size))
    b_free, b_dirichlet = operators.B_free, operators.B_dirichlet
    num_rows = (3 if grid.dim == 2 else 6) * grid.num_cells
    assert isinstance(b_free, sps.csr_array)
    assert isinstance(b_dirichlet, sps.csr_array)
    assert b_free.shape == (num_rows, grid.dim * layout.free_faces.size)
    assert b_dirichlet.shape == (num_rows, grid.dim * layout.dirichlet_faces.size)

    def reconstructed_mandel(increments: FloatArray) -> FloatArray:
        values = operators.face_displacement_increment(increments, prescribed)
        return _mandel_reference(operators.green_gauss_reconstruction(values), grid.dim)

    expected = b_free @ z[layout.face_slice] + b_dirichlet @ prescribed.ravel(order="F")
    np.testing.assert_allclose(reconstructed_mandel(z), expected, rtol=1e-13, atol=1e-13)
    changed_cells = z.copy()
    changed_cells[layout.cell_slice] += 100.0
    np.testing.assert_array_equal(reconstructed_mandel(changed_cells), reconstructed_mandel(z))

    direction = rng.normal(size=layout.size)
    step = 1e-5
    difference = (
        reconstructed_mandel(z + step * direction)
        - reconstructed_mandel(z - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(
        difference, b_free @ direction[layout.face_slice], rtol=1e-8, atol=1e-8,
    )


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed", "single_cell"])
def test_side_ordering_normals_and_known_tensor_grid_distances(
    grid: pp.Grid, boundary_type: str,
) -> None:
    if boundary_type == "single_cell":
        grid = pp.CartGrid(np.ones(grid.dim, dtype=int))
        grid.compute_geometry()
        boundary_type = "dirichlet"
    operators = RetainedFaceOperators(grid, _layout(grid, boundary_type))
    boundary = grid.get_all_boundary_faces()
    expected_counts = np.full(grid.num_faces, 2)
    expected_counts[boundary] = 1
    np.testing.assert_array_equal(
        np.bincount(operators.side_faces, minlength=grid.num_faces), expected_counts,
    )
    assert operators.num_sides == 2 * grid.num_faces - boundary.size
    assert operators.side_faces.dtype == operators.side_cells.dtype == np.int64
    assert operators.side_signs.dtype == np.float64
    assert operators.side_normals.shape == (grid.dim, operators.num_sides)

    # Independently enumerate connections in face/cell order from COO entries.
    entries = grid.cell_faces.tocoo()
    expected_sides = sorted(zip(entries.row, entries.col, entries.data, strict=True))
    np.testing.assert_array_equal(operators.side_faces, [side[0] for side in expected_sides])
    np.testing.assert_array_equal(operators.side_cells, [side[1] for side in expected_sides])
    np.testing.assert_array_equal(operators.side_signs, [side[2] for side in expected_sides])

    # Analytic box geometry: distance = half the cell width along the normal;
    # area = product of cell widths in the other coordinate directions.
    cell_nodes = grid.cell_nodes().tocsc()
    for side, (face, cell, _) in enumerate(expected_sides):
        nodes = cell_nodes.indices[cell_nodes.indptr[cell]:cell_nodes.indptr[cell + 1]]
        coordinates = grid.nodes[:grid.dim, nodes]
        widths = np.ptp(coordinates, axis=1)
        offset = grid.face_centers[:grid.dim, face] - grid.cell_centers[:grid.dim, cell]
        axis = int(np.argmax(np.abs(offset)))
        expected_normal = np.zeros(grid.dim)
        expected_normal[axis] = np.sign(offset[axis])
        np.testing.assert_allclose(operators.side_normals[:, side], expected_normal, atol=1e-13)
        np.testing.assert_allclose(operators.side_distances[side], widths[axis] / 2, atol=1e-13)
        np.testing.assert_allclose(operators.side_areas[side], np.prod(np.delete(widths, axis)))

    # Two outward normals on an interior face cancel. Each cell is also closed.
    for face in np.flatnonzero(expected_counts == 2):
        sides = np.flatnonzero(operators.side_faces == face)
        np.testing.assert_allclose(operators.side_normals[:, sides].sum(axis=1), 0, atol=1e-13)
    for cell in range(grid.num_cells):
        sides = operators.side_cells == cell
        np.testing.assert_allclose(
            (operators.side_normals[:, sides] * operators.side_areas[sides]).sum(axis=1),
            0, atol=1e-13,
        )


@pytest.mark.parametrize("transformation", ["rotation", "shear"])
def test_side_geometry_under_affine_grid_transformations(
    grid: pp.Grid, transformation: str,
) -> None:
    layout = _layout(grid)
    original = RetainedFaceOperators(grid, layout)
    transform = np.eye(3)
    if transformation == "rotation":
        c, s = np.cos(0.4), np.sin(0.4)
        transform = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        if grid.dim == 3:
            transform = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ transform
    else:
        transform[0, 1] = 0.35
        if grid.dim == 3:
            transform[1, 2] = -0.2
    grid.nodes[:] = transform @ grid.nodes
    grid.compute_geometry()
    actual = RetainedFaceOperators(grid, layout)

    # Normals transform by the inverse transpose of the coordinate map.
    active_transform = transform[:grid.dim, :grid.dim]
    normal_map = np.linalg.inv(active_transform).T @ original.side_normals
    normal_scale = np.linalg.norm(normal_map, axis=0)
    np.testing.assert_array_equal(actual.side_faces, original.side_faces)
    np.testing.assert_array_equal(actual.side_cells, original.side_cells)
    np.testing.assert_allclose(actual.side_normals, normal_map / normal_scale, atol=1e-13)
    np.testing.assert_allclose(
        actual.side_areas,
        abs(np.linalg.det(active_transform)) * normal_scale * original.side_areas,
        rtol=1e-12, atol=1e-13,
    )
    np.testing.assert_allclose(
        actual.side_distances, original.side_distances / normal_scale,
        rtol=1e-12, atol=1e-13,
    )
    if transformation == "shear":
        offsets = (grid.face_centers[:grid.dim, actual.side_faces]
                   - grid.cell_centers[:grid.dim, actual.side_cells])
        # A nonorthogonal offset distinguishes projection from Euclidean length.
        assert np.any(np.linalg.norm(offsets, axis=0) - actual.side_distances > 1e-4)


@pytest.mark.parametrize("sparse_format", ["csr", "csc", "coo"])
def test_side_geometry_canonicalizes_a_copy_of_connectivity(
    grid: pp.Grid, sparse_format: str,
) -> None:
    layout = _layout(grid)
    expected = RetainedFaceOperators(grid, layout)
    canonical = grid.cell_faces.tocsr()
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for face in range(grid.num_faces):
        start, stop = canonical.indptr[face:face + 2]
        # Store duplicate entries and an explicit zero, with unsorted indices.
        zero_cell = next(cell for cell in range(grid.num_cells)
                         if cell not in canonical.indices[start:stop])
        row = [(zero_cell, 0.0)]
        for position in range(start, stop):
            cell, sign = int(canonical.indices[position]), float(canonical.data[position])
            row.extend([(cell, 0.25 * sign), (cell, 0.75 * sign)])
        for cell, value in sorted(row, reverse=True):
            indices.append(cell)
            data.append(value)
        indptr.append(len(data))
    connectivity = sps.csr_array(
        (np.array(data), np.array(indices), np.array(indptr)), shape=canonical.shape,
    ).asformat(sparse_format)
    grid.cell_faces = connectivity
    storage_names = ("data", "row", "col") if sparse_format == "coo" else ("data", "indices", "indptr")
    saved_storage = {name: getattr(connectivity, name).copy() for name in storage_names}
    saved_geometry = {name: getattr(grid, name).copy() for name in (
        "face_normals", "face_areas", "face_centers", "cell_centers", "cell_volumes",
    )}

    actual = RetainedFaceOperators(grid, layout)
    assert grid.cell_faces is connectivity
    for name, saved in saved_storage.items():
        np.testing.assert_array_equal(getattr(connectivity, name), saved)
    for name, saved in saved_geometry.items():
        np.testing.assert_array_equal(getattr(grid, name), saved)
    for name in ("side_faces", "side_cells", "side_signs", "side_areas", "side_normals", "side_distances"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-13)
    assert not np.shares_memory(actual.side_signs, connectivity.data)
    assert not np.shares_memory(actual.side_areas, grid.face_areas)
    assert not np.shares_memory(actual.side_normals, grid.face_normals)


@pytest.mark.parametrize("defect", [
    "missing_boundary", "extra_boundary", "missing_interior", "extra_interior",
    "same_signs", "nonunit_sign",
])
def test_side_geometry_rejects_invalid_connectivity(grid: pp.Grid, defect: str) -> None:
    layout = _layout(grid)
    boundary = grid.get_all_boundary_faces()
    interior = np.setdiff1d(np.arange(grid.num_faces), boundary)
    face = int(boundary[0] if "boundary" in defect or defect == "nonunit_sign" else interior[0])
    connectivity = grid.cell_faces.astype(float).tolil()
    cells = list(connectivity.rows[face])
    if defect.startswith("missing"):
        connectivity[face, cells[0]] = 0.0
    elif defect.startswith("extra"):
        extra_cell = next(cell for cell in range(grid.num_cells) if cell not in cells)
        connectivity[face, extra_cell] = -float(connectivity[face, cells[0]])
    elif defect == "same_signs":
        connectivity[face, cells[1]] = connectivity[face, cells[0]]
    else:
        connectivity[face, cells[0]] = 0.5
    grid.cell_faces = connectivity.tocsc()
    with pytest.raises(ValueError, match="Expected one side"):
        RetainedFaceOperators(grid, layout)


@pytest.mark.parametrize("scale", [0.0, 2.0])
def test_side_geometry_rejects_inconsistent_normal_magnitudes(grid: pp.Grid, scale: float) -> None:
    layout = _layout(grid)
    grid.face_normals[:, 0] *= scale
    with pytest.raises(ValueError, match="magnitudes must equal face measures"):
        RetainedFaceOperators(grid, layout)


def test_side_geometry_rejects_zero_projected_distance(grid: pp.Grid) -> None:
    layout = _layout(grid)
    face = int(grid.get_all_boundary_faces()[0])
    incidence = grid.cell_faces.tocsr()
    cell = incidence.indices[incidence.indptr[face]]
    grid.cell_centers[:, cell] = grid.face_centers[:, face]
    with pytest.raises(ValueError, match="distances must be finite and positive"):
        RetainedFaceOperators(grid, layout)


@pytest.mark.parametrize("attribute", ["dim", "num_cells", "num_faces"])
def test_constructor_rejects_incompatible_layout(grid: pp.Grid, attribute: str) -> None:
    layout = _layout(grid)
    incompatible = replace(layout, **{attribute: getattr(layout, attribute) + 1})
    with pytest.raises(ValueError, match="incompatible"):
        RetainedFaceOperators(grid, incompatible)


def test_constructor_requires_valid_geometry(grid: pp.Grid) -> None:
    layout = _layout(grid)
    grid.cell_volumes[0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        RetainedFaceOperators(grid, layout)


@pytest.mark.parametrize("invalid", ["scalar", "short", "long", "column"])
def test_face_assembly_rejects_invalid_solution_shape(
    operators: RetainedFaceOperators, invalid: str,
) -> None:
    layout = operators.layout
    z = {
        "scalar": np.array(0.0), "short": np.zeros(layout.size - 1),
        "long": np.zeros(layout.size + 1), "column": np.zeros((layout.size, 1)),
    }[invalid]
    with pytest.raises(ValueError, match="Solution vector must have shape"):
        operators.face_displacement_increment(z, np.zeros((layout.dim, layout.dirichlet_faces.size)))


@pytest.mark.parametrize("invalid", ["flat", "transposed", "components", "all_faces"])
def test_face_assembly_rejects_invalid_prescribed_shape(
    operators: RetainedFaceOperators, invalid: str,
) -> None:
    layout = operators.layout
    nd, nf = layout.dim, layout.dirichlet_faces.size
    prescribed = {
        "flat": np.zeros(nd * nf), "transposed": np.zeros((nf, nd)),
        "components": np.zeros((nd + 1, nf)), "all_faces": np.zeros((nd, layout.num_faces)),
    }[invalid]
    with pytest.raises(ValueError, match="Dirichlet increments must have shape"):
        operators.face_displacement_increment(np.zeros(layout.size), prescribed)


@pytest.mark.parametrize("target", ["cell", "free_face", "dirichlet"])
@pytest.mark.parametrize("invalid", ["nan", "inf", "complex"])
def test_face_assembly_rejects_nonfinite_or_complex_data(
    operators: RetainedFaceOperators, target: str, invalid: str,
) -> None:
    layout = operators.layout
    dtype = np.complex128 if invalid == "complex" else np.float64
    z = np.zeros(layout.size, dtype=dtype)
    prescribed = np.zeros((layout.dim, layout.dirichlet_faces.size), dtype=dtype)
    value = {"nan": np.nan, "inf": np.inf, "complex": 1j}[invalid]
    if target == "dirichlet":
        prescribed[0, 0] = value
    else:
        z[0 if target == "cell" else layout.cell_size] = value
    with pytest.raises(ValueError, match="real|finite"):
        operators.face_displacement_increment(cast(FloatArray, z), cast(FloatArray, prescribed))


def test_reconstruction_rejects_wrong_number_of_values(operators: RetainedFaceOperators) -> None:
    with pytest.raises(ValueError):
        operators.green_gauss_reconstruction(
            np.zeros((operators.layout.dim, operators.layout.num_faces - 1))
        )


def test_reconstruction_rejects_complex_data(operators: RetainedFaceOperators) -> None:
    values = np.full((operators.layout.dim, operators.layout.num_faces), 1j)
    with pytest.raises(ValueError, match="real"):
        operators.green_gauss_reconstruction(cast(FloatArray, values))


@pytest.mark.parametrize("invalid", ["flat", "transposed"])
def test_reconstruction_requires_face_array_shape(
    operators: RetainedFaceOperators, invalid: str,
) -> None:
    values: FloatArray = np.zeros((operators.layout.dim, operators.layout.num_faces))
    values = values.ravel() if invalid == "flat" else values.T
    with pytest.raises(ValueError):
        operators.green_gauss_reconstruction(values)


@pytest.mark.parametrize("value", [np.nan, np.inf], ids=["nan", "inf"])
def test_reconstruction_requires_finite_values(
    operators: RetainedFaceOperators, value: float,
) -> None:
    values = np.zeros((operators.layout.dim, operators.layout.num_faces))
    values[0, operators.layout.free_faces[0]] = value
    with pytest.raises(ValueError):
        operators.green_gauss_reconstruction(values)
