"""Outward integrated elastic tractions and their relation to PorePy TPSA."""

from typing import cast

import numpy as np
import porepy as pp
import pytest

from core.layout import DofLayout, FloatArray
from core.operators import RetainedFaceOperators
from core.tpsa_wrapper import ElasticTpsa


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def grid(request: pytest.FixtureRequest) -> pp.Grid:
    result = pp.TensorGrid(
        np.array([0.0, 0.3, 1.4]),
        np.array([0.0, 0.4, 1.0]),
        np.array([0.0, 0.2, 0.9]) if request.param == 3 else None,
    )
    result.compute_geometry()
    return result


def _bc(grid: pp.Grid, boundary_type: str = "mixed") -> pp.BoundaryConditionVectorial:
    boundary = grid.get_all_boundary_faces()
    prescribed = {
        "dirichlet": boundary,
        "neumann": boundary[:0],
        "mixed": boundary[::2][::-1],
    }[boundary_type]
    return pp.BoundaryConditionVectorial(grid, prescribed, "dir")


@pytest.fixture
def operators(grid: pp.Grid) -> RetainedFaceOperators:
    return RetainedFaceOperators(
        grid, DofLayout.from_grid(grid, _bc(grid)), mu=np.linspace(1.5, 3.0, grid.num_cells),
    )


def _rotation_tensor(rotation: FloatArray, dim: int) -> FloatArray:
    """Independent skew-matrix reference for the rotation stress."""
    if dim == 2:
        return np.array([[0.0, -rotation[0]], [rotation[0], 0.0]])
    rx, ry, rz = rotation
    return np.array([[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]])


def _affine_state(
    operators: RetainedFaceOperators, gradient: FloatArray, mu: float, lam: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    grid, layout = operators.grid, operators.layout
    nd, nc = grid.dim, grid.num_cells
    translation = np.arange(1, nd + 1, dtype=float)[:, None] / 7
    u = gradient @ grid.cell_centers[:nd] + translation
    faces = gradient @ grid.face_centers[:nd] + translation
    # TPSA rotation is minus mu times curl(u).
    rotation = (np.array([gradient[0, 1] - gradient[1, 0]]) if nd == 2 else np.array([
        gradient[1, 2] - gradient[2, 1],
        gradient[2, 0] - gradient[0, 2],
        gradient[0, 1] - gradient[1, 0],
    ]))
    r = np.repeat((mu * rotation)[:, None], nc, axis=1)
    p = np.full(nc, lam * np.trace(gradient))
    cell_values = np.concatenate((u.ravel(order="F"), r.ravel(order="F"), p))
    z = layout.pack(cell_values, faces[:, layout.free_faces])
    stress = mu * (gradient + gradient.T) + lam * np.trace(gradient) * np.eye(nd)
    return z, faces[:, layout.dirichlet_faces], stress


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed"])
@pytest.mark.parametrize("field", ["translation", "rotation", "affine"])
@pytest.mark.parametrize("rotate_grid", [False, True], ids=["aligned", "rotated"])
def test_affine_traction_matches_continuum_and_balances(
    grid: pp.Grid, boundary_type: str, field: str, rotate_grid: bool,
) -> None:
    if rotate_grid:
        c, s = np.cos(0.4), np.sin(0.4)
        transform = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        if grid.dim == 3:
            transform = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ transform
        grid.nodes[:] = transform @ grid.nodes
        grid.compute_geometry()
    mu, lam = 2.0, 3.0
    layout = DofLayout.from_grid(grid, _bc(grid, boundary_type))
    operators = RetainedFaceOperators(grid, layout, mu=np.full(grid.num_cells, mu))
    gradient = np.array([
        [0.1, 0.2, -0.3], [-0.4, 0.5, 0.6], [0.7, -0.8, 0.9],
    ])[:grid.dim, :grid.dim]
    if field == "translation":
        gradient = np.zeros_like(gradient)
    elif field == "rotation":
        gradient = 0.5 * (gradient - gradient.T)
    z, prescribed, stress = _affine_state(operators, gradient, mu, lam)
    actual = operators.elastic_one_sided_traction(z, prescribed)
    outward_measure = grid.face_normals[:grid.dim, operators.side_faces] * operators.side_signs
    np.testing.assert_allclose(actual, stress @ outward_measure, rtol=1e-12, atol=1e-12)
    assert actual.shape == (grid.dim, operators.num_sides)
    assert actual.dtype == np.float64
    assert actual.flags.owndata
    if field in ("translation", "rotation"):
        np.testing.assert_allclose(actual, 0.0, atol=1e-12)
    for cell in range(grid.num_cells):
        np.testing.assert_allclose(actual[:, operators.side_cells == cell].sum(axis=1), 0, atol=1e-12)
    interior = np.setdiff1d(np.arange(grid.num_faces), grid.get_all_boundary_faces())
    for face in interior:
        np.testing.assert_allclose(actual[:, operators.side_faces == face].sum(axis=1), 0, atol=1e-12)


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann"])
def test_single_cell_traction_with_empty_free_or_prescribed_block(
    grid: pp.Grid, boundary_type: str,
) -> None:
    single = pp.CartGrid(np.ones(grid.dim, dtype=int))
    single.compute_geometry()
    layout = DofLayout.from_grid(single, _bc(single, boundary_type))
    operators = RetainedFaceOperators(single, layout, mu=np.array([2.0]))
    gradient = np.array([[0.2, 0.3, 0.4], [-0.1, 0.5, 0.6], [0.7, -0.8, 0.9]])[:grid.dim, :grid.dim]
    z, prescribed, stress = _affine_state(operators, gradient, 2.0, 3.0)
    if boundary_type == "dirichlet":
        assert z[layout.face_slice].size == 0
    else:
        assert prescribed.shape == (grid.dim, 0)
    expected = stress @ (single.face_normals[:grid.dim, operators.side_faces] * operators.side_signs)
    np.testing.assert_allclose(operators.elastic_one_sided_traction(z, prescribed), expected, atol=1e-12)


@pytest.mark.parametrize("term", ["displacement", "rotation", "pressure", "combined"])
def test_traction_terms_match_independent_side_reference(
    operators: RetainedFaceOperators, term: str,
) -> None:
    grid, layout = operators.grid, operators.layout
    assert operators.mu is not None
    rng = np.random.default_rng(41)
    u = rng.normal(size=(grid.dim, grid.num_cells))
    r = rng.normal(size=(layout.rotation_dim, grid.num_cells))
    p = rng.normal(size=grid.num_cells)
    faces = rng.normal(size=(grid.dim, grid.num_faces))
    if term not in ("displacement", "combined"):
        u.fill(0.0)
        faces.fill(0.0)
    if term not in ("rotation", "combined"):
        r.fill(0.0)
    if term not in ("pressure", "combined"):
        p.fill(0.0)
    z = layout.pack(np.concatenate((u.ravel(order="F"), r.ravel(order="F"), p)),
                    faces[:, layout.free_faces])
    actual = operators.elastic_one_sided_traction(z, faces[:, layout.dirichlet_faces])
    expected = np.empty_like(actual)
    incidence = grid.cell_faces.tocsr()
    for side, (face, cell) in enumerate(zip(operators.side_faces, operators.side_cells, strict=True)):
        start, stop = incidence.indptr[face:face + 2]
        adjacent = incidence.indices[start:stop]
        area = grid.face_areas[face]
        n = grid.face_normals[:grid.dim, face] * float(incidence[face, cell]) / area
        distances = np.abs(n @ (grid.face_centers[:grid.dim, face, None]
                               - grid.cell_centers[:grid.dim, adjacent]))
        a = 2.0 * operators.mu[adjacent] / distances
        face_r = r[:, adjacent[0]] if adjacent.size == 1 else (
            a[1] * r[:, adjacent[0]] + a[0] * r[:, adjacent[1]]
        ) / (a[0] + a[1])
        cell_a = a[np.flatnonzero(adjacent == cell)[0]]
        expected[:, side] = area * (
            cell_a * (faces[:, face] - u[:, cell])
            + _rotation_tensor(face_r, grid.dim) @ n + p[cell] * n
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed"])
def test_face_equilibrated_traction_matches_porepy_tpsa(
    grid: pp.Grid, boundary_type: str,
) -> None:
    """Compare force fluxes for arbitrary cell values after local face elimination."""
    bc = _bc(grid, boundary_type)
    layout = DofLayout.from_grid(grid, bc)
    mu = np.linspace(1.5, 3.0, grid.num_cells)
    operators = RetainedFaceOperators(grid, layout, mu=mu)
    tpsa = ElasticTpsa(grid, bc, mu=mu, lam=3.0)
    rng = np.random.default_rng(51)
    cell_values = rng.normal(size=layout.cell_size)
    u = cell_values[layout.u_slice].reshape((grid.dim, grid.num_cells), order="F")
    r = cell_values[layout.r_slice].reshape((layout.rotation_dim, grid.num_cells), order="F")
    p = cell_values[layout.p_slice]
    boundary_values = np.zeros((grid.dim, grid.num_faces))
    boundary = grid.get_all_boundary_faces()
    boundary_values[:, boundary] = rng.normal(size=(grid.dim, boundary.size))
    face_values = np.empty((grid.dim, grid.num_faces))
    incidence = grid.cell_faces.tocsr()
    for face in range(grid.num_faces):
        if face in layout.dirichlet_faces:
            face_values[:, face] = boundary_values[:, face]
            continue
        start, stop = incidence.indptr[face:face + 2]
        adjacent = incidence.indices[start:stop]
        area = grid.face_areas[face]
        n = grid.face_normals[:grid.dim, face] * incidence.data[start] / area
        distances = np.abs(n @ (grid.face_centers[:grid.dim, face, None]
                               - grid.cell_centers[:grid.dim, adjacent]))
        a = 2.0 * mu[adjacent] / distances
        if adjacent.size == 2:
            i, j = adjacent
            # Shared rotation cancels in the sum of the two outward tractions.
            face_values[:, face] = (
                a[0] * u[:, i] + a[1] * u[:, j] - (p[i] - p[j]) * n
            ) / (a[0] + a[1])
        else:
            i = adjacent[0]
            # Neumann input is outward integrated force, not traction density.
            face_values[:, face] = u[:, i] + (
                boundary_values[:, face] / area - _rotation_tensor(r[:, i], grid.dim) @ n
                - p[i] * n
            ) / a[0]
    z = layout.pack(cell_values, face_values[:, layout.free_faces])
    actual = operators.elastic_one_sided_traction(z, face_values[:, layout.dirichlet_faces])
    flux = (tpsa.matrices.face_discretization @ cell_values
            + tpsa.matrices.rhs_matrix @ boundary_values.ravel(order="F"))
    fixed_normal_traction = flux[:grid.dim * grid.num_faces].reshape((grid.dim, grid.num_faces), order="F")
    expected = fixed_normal_traction[:, operators.side_faces] * operators.side_signs
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    interior = np.setdiff1d(np.arange(grid.num_faces), boundary)
    neumann = np.setdiff1d(boundary, layout.dirichlet_faces)
    for face in interior:
        np.testing.assert_allclose(actual[:, operators.side_faces == face].sum(axis=1), 0, atol=1e-11)
    for face in neumann:
        np.testing.assert_allclose(
            actual[:, operators.side_faces == face].ravel(), boundary_values[:, face], atol=1e-11,
        )


def test_outward_traction_is_invariant_under_face_orientation_reversal(
    operators: RetainedFaceOperators,
) -> None:
    grid, layout = operators.grid, operators.layout
    assert operators.mu is not None
    rng = np.random.default_rng(61)
    z = rng.normal(size=layout.size)
    prescribed = rng.normal(size=(grid.dim, layout.dirichlet_faces.size))
    expected = operators.elastic_one_sided_traction(z, prescribed)
    signs = np.ones(grid.num_faces)
    signs[::2] = -1
    grid.face_normals *= signs
    grid.cell_faces = grid.cell_faces.multiply(signs[:, None]).tocsc()
    reversed_operators = RetainedFaceOperators(grid, layout, mu=operators.mu)
    np.testing.assert_array_equal(reversed_operators.side_faces, operators.side_faces)
    np.testing.assert_array_equal(reversed_operators.side_cells, operators.side_cells)
    np.testing.assert_allclose(
        reversed_operators.elastic_one_sided_traction(z, prescribed), expected, atol=1e-12,
    )


def test_traction_is_linear_preserves_inputs_and_reuses_coefficients(
    operators: RetainedFaceOperators, monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = operators.layout
    assert operators.side_stiffness is not None and operators.rotation_transfer is not None
    stiffness = operators.side_stiffness.copy()
    weights = operators.rotation_transfer.weights.toarray()
    rng = np.random.default_rng(71)
    z_storage = rng.normal(size=2 * layout.size)
    d_storage = rng.normal(size=(layout.dim, 2 * layout.dirichlet_faces.size))
    z, prescribed = z_storage[::2], d_storage[:, ::2]
    z.setflags(write=False)
    prescribed.setflags(write=False)
    saved_z, saved_d = z_storage.copy(), d_storage.copy()
    other_z = rng.normal(size=layout.size)
    other_d = rng.normal(size=prescribed.shape)

    def unexpected_rebuild(*args: object) -> None:
        pytest.fail("Traction evaluation must reuse cached operators.")

    for name in ("_build_strain_matrix", "_build_side_geometry", "_build_traction_coefficients"):
        monkeypatch.setattr(operators, name, unexpected_rebuild)
    first = operators.elastic_one_sided_traction(z, prescribed)
    second = operators.elastic_one_sided_traction(other_z, other_d)
    combined = operators.elastic_one_sided_traction(2.0 * z - 0.5 * other_z, 2.0 * prescribed - 0.5 * other_d)
    np.testing.assert_allclose(combined, 2.0 * first - 0.5 * second, rtol=1e-12, atol=1e-12)
    assert first.flags.owndata
    assert not np.shares_memory(first, z)
    assert not np.shares_memory(first, prescribed)
    first.fill(999.0)
    np.testing.assert_array_equal(z_storage, saved_z)
    np.testing.assert_array_equal(d_storage, saved_d)
    np.testing.assert_array_equal(operators.side_stiffness, stiffness)
    np.testing.assert_array_equal(operators.rotation_transfer.weights.toarray(), weights)


@pytest.mark.parametrize("missing", ["both", "side_stiffness", "rotation_transfer"])
def test_traction_requires_initialized_coefficients(
    grid: pp.Grid, missing: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = DofLayout.from_grid(grid, _bc(grid))
    mu = None if missing == "both" else np.full(grid.num_cells, 2.0)
    operators = RetainedFaceOperators(grid, layout, mu=mu)
    if missing != "both":
        monkeypatch.setattr(operators, missing, None)
    with pytest.raises(ValueError, match="Supply"):
        operators.elastic_one_sided_traction(np.zeros(layout.size), np.zeros((grid.dim, layout.dirichlet_faces.size)))


@pytest.mark.parametrize("invalid", ["short_z", "column_z", "flat_dirichlet", "transposed_dirichlet"])
def test_traction_validates_input_shapes(operators: RetainedFaceOperators, invalid: str) -> None:
    layout = operators.layout
    z: FloatArray = np.zeros(layout.size)
    prescribed: FloatArray = np.zeros((layout.dim, layout.dirichlet_faces.size))
    if invalid == "short_z":
        z = z[:-1]
    elif invalid == "column_z":
        z = z[:, None]
    elif invalid == "flat_dirichlet":
        prescribed = prescribed.ravel()
    else:
        prescribed = prescribed.T
    with pytest.raises(ValueError, match="must have shape"):
        operators.elastic_one_sided_traction(z, prescribed)


@pytest.mark.parametrize("target", ["u", "r", "p", "free_face", "dirichlet"])
@pytest.mark.parametrize("invalid", ["nan", "inf", "complex"])
def test_traction_rejects_nonfinite_or_complex_inputs(
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
        block = {"u": layout.u_slice, "r": layout.r_slice, "p": layout.p_slice,
                 "free_face": layout.face_slice}[target]
        z[block.start] = value
    with pytest.raises(ValueError, match="real|finite"):
        operators.elastic_one_sided_traction(cast(FloatArray, z), cast(FloatArray, prescribed))


def test_traction_rejects_overflow_from_finite_inputs(operators: RetainedFaceOperators) -> None:
    layout = operators.layout
    z = np.zeros(layout.size)
    z[layout.u_slice.start] = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="Elastic traction increments must be finite"):
        operators.elastic_one_sided_traction(z, np.zeros((layout.dim, layout.dirichlet_faces.size)))
