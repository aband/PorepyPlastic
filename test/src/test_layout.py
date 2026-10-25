"""DOF numbering and packing for the dimension-aware retained formulation."""

import numpy as np
import porepy as pp
import pytest

from core.layout import DofLayout
from core.tpsa_wrapper import ElasticTpsa


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def grid(request: pytest.FixtureRequest) -> pp.Grid:
    cells = [2, 3] if request.param == 2 else [2, 3, 2]
    result = pp.CartGrid(cells)
    result.compute_geometry()
    return result


@pytest.fixture
def bc(grid: pp.Grid) -> pp.BoundaryConditionVectorial:
    # Interleave Dirichlet and Neumann faces and supply unsorted Dirichlet IDs.
    faces = grid.get_all_boundary_faces()[::2][::-1]
    return pp.BoundaryConditionVectorial(grid, faces, "dir")


@pytest.fixture
def layout(grid: pp.Grid, bc: pp.BoundaryConditionVectorial) -> DofLayout:
    return DofLayout.from_grid(grid, bc)


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed"])
def test_face_partition_and_global_numbering(grid: pp.Grid, boundary_type: str) -> None:
    boundary = grid.get_all_boundary_faces()
    prescribed = {
        "dirichlet": boundary,
        "neumann": boundary[:0],
        "mixed": boundary[::2][::-1],
    }[boundary_type]
    bc = pp.BoundaryConditionVectorial(grid, prescribed, "dir")
    saved_flags = [flags.copy() for flags in (bc.is_dir, bc.is_neu, bc.is_rob)]
    layout = DofLayout.from_grid(grid, bc)

    expected_free = np.array([
        face for face in range(grid.num_faces) if face not in prescribed
    ], dtype=np.int64)
    np.testing.assert_array_equal(layout.dirichlet_faces, np.sort(prescribed))
    np.testing.assert_array_equal(layout.free_faces, expected_free)
    assert layout.dirichlet_faces.dtype == layout.free_faces.dtype == np.int64

    assert layout.dim == grid.dim
    assert layout.num_cells == grid.num_cells
    assert layout.num_faces == grid.num_faces
    assert layout.rotation_dim == (1 if grid.dim == 2 else 3)
    expected_cell_size = (4 if grid.dim == 2 else 7) * grid.num_cells
    assert layout.cell_size == expected_cell_size
    assert layout.size == expected_cell_size + grid.dim * expected_free.size

    assert layout.face_dofs.shape == (grid.dim, grid.num_faces)
    assert layout.face_dofs.dtype == np.int64
    np.testing.assert_array_equal(layout.face_dofs[:, prescribed], -1)
    # Number by ascending global face ID, with components adjacent on each face.
    next_dof = expected_cell_size
    for face in expected_free:
        for component in range(grid.dim):
            assert layout.face_dofs[component, face] == next_dof
            next_dof += 1
    assert next_dof == layout.size
    for flags, saved in zip((bc.is_dir, bc.is_neu, bc.is_rob), saved_flags, strict=True):
        np.testing.assert_array_equal(flags, saved)


@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_pack_preserves_cell_blocks_and_face_component_order(
    layout: DofLayout, strided: bool,
) -> None:
    nd, nr, nc = layout.dim, layout.rotation_dim, layout.num_cells
    u = 100 + 10 * np.arange(nc)[None, :] + np.arange(nd)[:, None]
    r = 200 + 10 * np.arange(nc)[None, :] + np.arange(nr)[:, None]
    p = 300 + np.arange(nc)
    # Construct the reference cell vector explicitly in cell/component order.
    cell_values = np.array(
        [u[component, cell] for cell in range(nc) for component in range(nd)]
        + [r[component, cell] for cell in range(nc) for component in range(nr)]
        + list(p), dtype=float,
    )
    face_values = (1000 + 10 * layout.free_faces[None, :]
                   + np.arange(nd)[:, None]).astype(float)
    if strided:
        cell_storage = np.zeros(2 * cell_values.size)
        cell_storage[::2] = cell_values
        cell_values = cell_storage[::2]
        face_storage = np.zeros((nd, 2 * face_values.shape[1]))
        face_storage[:, ::2] = face_values
        face_values = face_storage[:, ::2]
        assert not cell_values.flags.c_contiguous
        assert not face_values.flags.c_contiguous

    saved_cell, saved_face = cell_values.copy(), face_values.copy()
    z = layout.pack(cell_values, face_values)
    expected_face = np.array([
        face_values[component, position]
        for position in range(layout.free_faces.size) for component in range(nd)
    ])
    np.testing.assert_array_equal(z, np.concatenate((saved_cell, expected_face)))
    np.testing.assert_array_equal(z[layout.cell_slice], saved_cell)
    np.testing.assert_array_equal(z[layout.u_slice].reshape(nc, nd).T, u)
    np.testing.assert_array_equal(z[layout.r_slice].reshape(nc, nr).T, r)
    np.testing.assert_array_equal(z[layout.p_slice], p)
    np.testing.assert_array_equal(z[layout.face_slice], expected_face)
    np.testing.assert_array_equal(z[layout.face_dofs[:, layout.free_faces]], saved_face)

    cell_copy, face_copy = layout.unpack(z)
    np.testing.assert_array_equal(cell_copy, saved_cell)
    np.testing.assert_array_equal(face_copy, saved_face)
    np.testing.assert_array_equal(layout.pack(cell_copy, face_copy), z)
    assert not np.shares_memory(z, cell_values)
    assert not np.shares_memory(z, face_values)
    assert not np.shares_memory(z, cell_copy)
    assert not np.shares_memory(z, face_copy)
    saved_z = z.copy()
    cell_copy.fill(-1.0)
    face_copy.fill(-2.0)
    np.testing.assert_array_equal(z, saved_z)
    z.fill(-3.0)
    np.testing.assert_array_equal(cell_values, saved_cell)
    np.testing.assert_array_equal(face_values, saved_face)


def test_unpack_accepts_strided_solution(layout: DofLayout) -> None:
    storage = np.arange(2 * layout.size, dtype=float)
    z = storage[::2]
    cell, face = layout.unpack(z)
    np.testing.assert_array_equal(layout.pack(cell, face), z)
    assert not np.shares_memory(cell, storage)
    assert not np.shares_memory(face, storage)


def test_single_cell_dirichlet_has_no_retained_face_dofs(grid: pp.Grid) -> None:
    single = pp.CartGrid([1] * grid.dim)
    single.compute_geometry()
    bc = pp.BoundaryConditionVectorial(single, single.get_all_boundary_faces(), "dir")
    layout = DofLayout.from_grid(single, bc)
    assert layout.free_faces.size == 0
    assert layout.size == layout.cell_size == (4 if single.dim == 2 else 7)
    np.testing.assert_array_equal(layout.face_dofs, -1)
    cell = np.arange(layout.cell_size, dtype=float)
    z = layout.pack(cell, np.empty((single.dim, 0)))
    np.testing.assert_array_equal(z, cell)
    cell_copy, face_copy = layout.unpack(z)
    np.testing.assert_array_equal(cell_copy, cell)
    assert face_copy.shape == (single.dim, 0)
    assert z[layout.face_slice].size == 0


def test_cell_layout_matches_elastic_tpsa(
    grid: pp.Grid, bc: pp.BoundaryConditionVectorial, layout: DofLayout,
) -> None:
    elastic = ElasticTpsa(grid, bc, mu=2.0, lam=3.0)
    assert elastic.matrices.cell_matrix.shape == (layout.cell_size, layout.cell_size)
    assert elastic.matrices.face_discretization.shape[1] == layout.cell_size
    assert elastic.boundary_rhs(np.zeros(grid.dim * grid.num_faces)).shape == (
        layout.cell_size,
    )


def test_numbering_is_independent_of_boundary_enumeration(
    grid: pp.Grid, bc: pp.BoundaryConditionVectorial, layout: DofLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reversed_boundary = grid.get_all_boundary_faces()[::-1]
    monkeypatch.setattr(grid, "get_all_boundary_faces", lambda: reversed_boundary.copy())
    reordered = DofLayout.from_grid(grid, bc)
    np.testing.assert_array_equal(reordered.dirichlet_faces, layout.dirichlet_faces)
    np.testing.assert_array_equal(reordered.free_faces, layout.free_faces)
    np.testing.assert_array_equal(reordered.face_dofs, layout.face_dofs)


@pytest.mark.parametrize("invalid", ["scalar", "short", "long", "column"])
def test_pack_rejects_invalid_cell_shape(layout: DofLayout, invalid: str) -> None:
    values = {
        "scalar": np.array(0.0),
        "short": np.zeros(layout.cell_size - 1),
        "long": np.zeros(layout.cell_size + 1),
        "column": np.zeros((layout.cell_size, 1)),
    }[invalid]
    with pytest.raises(ValueError, match="Cell values must have shape"):
        layout.pack(values, np.zeros((layout.dim, layout.free_faces.size)))


@pytest.mark.parametrize("invalid", ["flat", "components", "short", "transposed", "all_faces"])
def test_pack_rejects_invalid_face_shape(layout: DofLayout, invalid: str) -> None:
    nd, nf = layout.dim, layout.free_faces.size
    values = {
        "flat": np.zeros(nd * nf),
        "components": np.zeros((nd + 1, nf)),
        "short": np.zeros((nd, nf - 1)),
        "transposed": np.zeros((nf, nd)),
        "all_faces": np.zeros((nd, layout.num_faces)),
    }[invalid]
    with pytest.raises(ValueError, match="Free-face values must have shape"):
        layout.pack(np.zeros(layout.cell_size), values)


@pytest.mark.parametrize("invalid", ["scalar", "short", "long", "column"])
def test_unpack_rejects_invalid_solution_shape(layout: DofLayout, invalid: str) -> None:
    values = {
        "scalar": np.array(0.0),
        "short": np.zeros(layout.size - 1),
        "long": np.zeros(layout.size + 1),
        "column": np.zeros((layout.size, 1)),
    }[invalid]
    with pytest.raises(ValueError, match="Solution vector must have shape"):
        layout.unpack(values)


@pytest.mark.parametrize("invalid", ["partial", "robin", "overlap", "unset"])
def test_unsupported_boundary_conditions(grid: pp.Grid, invalid: str) -> None:
    boundary = grid.get_all_boundary_faces()
    bc = pp.BoundaryConditionVectorial(grid, boundary, "dir")
    face = boundary[0]
    if invalid == "partial":
        bc.is_dir[0, face] = False
        bc.is_neu[0, face] = True
    elif invalid == "robin":
        bc.is_dir[:, face] = False
        bc.is_rob[:, face] = True
    elif invalid == "overlap":
        bc.is_neu[:, face] = True
    else:
        bc.is_dir[:, face] = False
    with pytest.raises(ValueError, match="Only full-vector Dirichlet and Neumann"):
        DofLayout.from_grid(grid, bc)


@pytest.mark.parametrize("attribute", ["dim", "num_faces"])
def test_boundary_grid_mismatch(
    grid: pp.Grid, bc: pp.BoundaryConditionVectorial, attribute: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bc, attribute, getattr(bc, attribute) + 1)
    with pytest.raises(ValueError, match="Boundary conditions do not match"):
        DofLayout.from_grid(grid, bc)


def test_unsupported_grid_dimension() -> None:
    grid = pp.CartGrid([2])
    grid.compute_geometry()
    bc = pp.BoundaryConditionVectorial(grid, grid.get_all_boundary_faces(), "dir")
    with pytest.raises(ValueError, match="Only 2D and 3D grids"):
        DofLayout.from_grid(grid, bc)
