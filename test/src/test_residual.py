"""Retained elastic balances, load validation, and reduction to PorePy TPSA."""

from dataclasses import replace
from typing import cast

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]

from core.layout import DofLayout, FloatArray
from core.operators import RetainedFaceOperators
from core.residual import LoadIncrement, ResidualAssembler
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


def _assembler(
    grid: pp.Grid, boundary_type: str = "mixed", *, uniform: bool = False,
) -> ResidualAssembler:
    boundary = grid.get_all_boundary_faces()
    prescribed = {
        "dirichlet": boundary,
        "neumann": boundary[:0],
        "mixed": boundary[::2][::-1],
    }[boundary_type]
    bc = pp.BoundaryConditionVectorial(grid, prescribed, "dir")
    mu = np.full(grid.num_cells, 2.0) if uniform else np.linspace(1.5, 3.0, grid.num_cells)
    lam = np.full(grid.num_cells, 3.0) if uniform else np.linspace(2.0, 4.0, grid.num_cells)
    operators = RetainedFaceOperators(grid, DofLayout.from_grid(grid, bc), mu=mu)
    return ResidualAssembler(operators, ElasticTpsa(grid, bc, mu, lam))


@pytest.fixture
def assembler(grid: pp.Grid) -> ResidualAssembler:
    return _assembler(grid)


def _loads(assembler: ResidualAssembler, seed: int = 11) -> LoadIncrement:
    layout = assembler.layout
    rng = np.random.default_rng(seed)
    return LoadIncrement(
        rng.normal(size=(layout.dim, layout.dirichlet_faces.size)),
        rng.normal(size=(layout.dim, assembler.neumann_faces.size)),
        rng.normal(size=layout.cell_size),
    )


def _rotation_stress(rotation: FloatArray, dim: int) -> FloatArray:
    if dim == 2:
        return np.array([[0.0, -rotation[0]], [rotation[0], 0.0]])
    rx, ry, rz = rotation
    return np.array([[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]])


def _equilibrated_faces(
    assembler: ResidualAssembler, x: FloatArray, boundary: FloatArray,
) -> FloatArray:
    """Solve the local traction equations independently of retained operators."""
    grid, layout = assembler.operators.grid, assembler.layout
    tensor = assembler.elastic.data[pp.PARAMETERS][assembler.elastic.keyword]["fourth_order_tensor"]
    u = x[layout.u_slice].reshape((grid.dim, grid.num_cells), order="F")
    r = x[layout.r_slice].reshape((layout.rotation_dim, grid.num_cells), order="F")
    p = x[layout.p_slice]
    faces = boundary.copy()
    incidence = grid.cell_faces.tocsr()
    for face in layout.free_faces:
        start, stop = incidence.indptr[face:face + 2]
        cells = incidence.indices[start:stop]
        area = grid.face_areas[face]
        n = grid.face_normals[:grid.dim, face] * incidence.data[start] / area
        distances = np.abs(n @ (grid.face_centers[:grid.dim, face, None]
                               - grid.cell_centers[:grid.dim, cells]))
        stiffness = 2 * tensor.mu[cells] / distances
        if cells.size == 2:
            i, j = cells
            faces[:, face] = (
                stiffness[0] * u[:, i] + stiffness[1] * u[:, j] - (p[i] - p[j]) * n
            ) / stiffness.sum()
        else:
            i = cells[0]
            faces[:, face] = u[:, i] + (
                boundary[:, face] / area - _rotation_stress(r[:, i], grid.dim) @ n - p[i] * n
            ) / stiffness[0]
    return faces


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed"])
def test_balances_match_independent_surface_sums(grid: pp.Grid, boundary_type: str) -> None:
    assembler = _assembler(grid, boundary_type)
    operators, layout = assembler.operators, assembler.layout
    rng = np.random.default_rng(12)
    z = rng.normal(size=layout.size)
    displacement = rng.normal(size=(grid.dim, grid.num_faces))
    traction = rng.normal(size=(grid.dim, operators.num_sides))
    loads = _loads(assembler)
    momentum = np.zeros((grid.dim, grid.num_cells))
    rotation = np.zeros((layout.rotation_dim, grid.num_cells))
    dilation = np.zeros(grid.num_cells)
    face_sum = np.zeros((grid.dim, grid.num_faces))
    for side, (face, cell) in enumerate(zip(operators.side_faces, operators.side_cells, strict=True)):
        # PorePy normals already include area; side tractions are also integrated.
        normal = grid.face_normals[:grid.dim, face] * float(grid.cell_faces[face, cell])
        d = displacement[:, face]
        momentum[:, cell] += traction[:, side]
        face_sum[:, face] += traction[:, side]
        if grid.dim == 2:
            rotation[0, cell] += d[0] * normal[1] - d[1] * normal[0]
        else:
            rotation[:, cell] += np.array([
                d[1] * normal[2] - d[2] * normal[1],
                d[2] * normal[0] - d[0] * normal[2],
                d[0] * normal[1] - d[1] * normal[0],
            ])
        dilation[cell] += d @ normal
    flux = np.concatenate((momentum.ravel(order="F"), rotation.ravel(order="F"), dilation))
    np.testing.assert_allclose(assembler._cell_flux_balance(displacement, traction), flux, atol=1e-13)

    # Neumann load columns follow sorted boundary IDs, which differ from positions
    # in the free-face block. Dirichlet reactions enter cells but have no face row.
    neumann = np.setdiff1d(grid.get_all_boundary_faces(), layout.dirichlet_faces)
    np.testing.assert_array_equal(assembler.neumann_faces, neumann)
    np.testing.assert_array_equal(layout.free_faces[assembler.neumann_positions], neumann)
    face_sum[:, neumann] -= loads.neumann_increment
    expected_faces = face_sum[:, layout.free_faces]
    np.testing.assert_allclose(
        assembler._free_face_balance(traction, loads.neumann_increment), expected_faces, atol=1e-13,
    )

    # Check accumulation from material coefficients, independently of the cached matrix.
    tensor = assembler.elastic.data[pp.PARAMETERS][assembler.elastic.keyword]["fourth_order_tensor"]
    flux[layout.r_slice] -= np.repeat(grid.cell_volumes / tensor.mu, layout.rotation_dim) * z[layout.r_slice]
    flux[layout.p_slice] -= grid.cell_volumes / tensor.lmbda * z[layout.p_slice]
    expected = np.concatenate((flux - loads.cell_source_increment, expected_faces.ravel(order="F")))
    actual = assembler.assemble(z, displacement, traction, loads)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert actual.shape == (layout.size,)
    assert actual.dtype == np.float64
    assert assembler.accumulation is assembler.elastic.matrices.accum


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed"])
@pytest.mark.parametrize("rotate_grid", [False, True], ids=["aligned", "rotated"])
def test_eliminating_faces_recovers_all_tpsa_cell_equations(
    grid: pp.Grid, boundary_type: str, rotate_grid: bool,
) -> None:
    if rotate_grid:
        c, s = np.cos(0.4), np.sin(0.4)
        transform = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        if grid.dim == 3:
            transform = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ transform
        grid.nodes[:] = transform @ grid.nodes
        grid.compute_geometry()
    assembler = _assembler(grid, boundary_type)
    layout, elastic = assembler.layout, assembler.elastic
    rng = np.random.default_rng(13)
    x = rng.normal(size=layout.cell_size)
    boundary = np.zeros((grid.dim, grid.num_faces))
    boundary_faces = grid.get_all_boundary_faces()
    boundary[:, boundary_faces] = rng.normal(size=(grid.dim, boundary_faces.size))
    faces = _equilibrated_faces(assembler, x, boundary)
    source = rng.normal(size=layout.cell_size)
    loads = LoadIncrement(boundary[:, layout.dirichlet_faces], boundary[:, assembler.neumann_faces], source)
    z = layout.pack(x, faces[:, layout.free_faces])
    evaluation = assembler.evaluate_elastic(z, loads)
    expected = elastic.matrices.cell_matrix @ x - elastic.boundary_rhs(boundary.ravel(order="F")) - source
    for block in (layout.u_slice, layout.r_slice, layout.p_slice):
        np.testing.assert_allclose(evaluation.residual[block], expected[block], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(evaluation.residual[layout.face_slice], 0.0, atol=1e-12)
    np.testing.assert_allclose(evaluation.face_increment, faces, atol=1e-14)


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed"])
@pytest.mark.parametrize("field", ["translation", "rotation", "affine"])
def test_affine_equilibrium(grid: pp.Grid, boundary_type: str, field: str) -> None:
    assembler = _assembler(grid, boundary_type, uniform=True)
    layout, nd, nc = assembler.layout, grid.dim, grid.num_cells
    gradient = np.array([[0.1, 0.2, -0.3], [-0.4, 0.5, 0.6], [0.7, -0.8, 0.9]])[:nd, :nd]
    if field == "translation":
        gradient = np.zeros_like(gradient)
    elif field == "rotation":
        gradient = 0.5 * (gradient - gradient.T)
    translation = np.arange(1, nd + 1)[:, None] / 7.0
    u = gradient @ grid.cell_centers[:nd] + translation
    faces = gradient @ grid.face_centers[:nd] + translation
    rotation = np.array([gradient[0, 1] - gradient[1, 0]]) if nd == 2 else np.array([
        gradient[1, 2] - gradient[2, 1], gradient[2, 0] - gradient[0, 2],
        gradient[0, 1] - gradient[1, 0],
    ])
    r = np.repeat((2.0 * rotation)[:, None], nc, axis=1)
    p = np.full(nc, 3.0 * np.trace(gradient))
    x = np.concatenate((u.ravel(order="F"), r.ravel(order="F"), p))
    stress = 2.0 * (gradient + gradient.T) + 3.0 * np.trace(gradient) * np.eye(nd)
    neumann = assembler.neumann_faces
    outward_normals = grid.face_normals[:nd, neumann] * np.asarray(grid.cell_faces[neumann].sum(axis=1)).ravel()
    loads = LoadIncrement(faces[:, layout.dirichlet_faces], stress @ outward_normals, np.zeros(layout.cell_size))
    evaluation = assembler.evaluate_elastic(layout.pack(x, faces[:, layout.free_faces]), loads)
    np.testing.assert_allclose(evaluation.residual, 0.0, atol=1e-12)
    if field != "affine":
        np.testing.assert_allclose(evaluation.traction_increment, 0.0, atol=1e-12)


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann"])
def test_single_cell_empty_boundary_or_free_block(grid: pp.Grid, boundary_type: str) -> None:
    single = pp.CartGrid(np.ones(grid.dim, dtype=int))
    single.compute_geometry()
    assembler = _assembler(single, boundary_type)
    layout = assembler.layout
    loads = _loads(assembler)
    zero_loads = LoadIncrement(*(np.zeros_like(value) for value in (
        loads.dirichlet_increment, loads.neumann_increment, loads.cell_source_increment,
    )))
    evaluation = assembler.evaluate_elastic(np.zeros(layout.size), zero_loads)
    np.testing.assert_array_equal(evaluation.residual, np.zeros(layout.size))
    if boundary_type == "dirichlet":
        assert evaluation.residual[layout.face_slice].shape == (0,)
        assert loads.neumann_increment.shape == (grid.dim, 0)
    else:
        assert loads.dirichlet_increment.shape == (grid.dim, 0)


def test_prescribed_neumann_load_changes_only_free_face_balance(assembler: ResidualAssembler) -> None:
    layout = assembler.layout
    z = np.random.default_rng(14).normal(size=layout.size)
    loads = _loads(assembler)
    first = assembler.evaluate_elastic(z, loads)
    increment = np.arange(loads.neumann_increment.size, dtype=float).reshape(loads.neumann_increment.shape) + 1
    second = assembler.evaluate_elastic(z, replace(loads, neumann_increment=loads.neumann_increment + increment))
    np.testing.assert_array_equal(first.face_increment, second.face_increment)
    np.testing.assert_array_equal(first.traction_increment, second.traction_increment)
    expected = np.zeros((layout.dim, layout.free_faces.size))
    expected[:, assembler.neumann_positions] = -increment
    difference = second.residual - first.residual
    np.testing.assert_array_equal(difference[layout.cell_slice], np.zeros(layout.cell_size))
    np.testing.assert_allclose(difference[layout.face_slice], expected.ravel(order="F"), atol=1e-13)


def test_evaluation_preserves_readonly_strided_inputs_and_is_linear(
    assembler: ResidualAssembler, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(15)
    layout = assembler.layout
    loads = _loads(assembler)

    def strided_readonly(value: FloatArray) -> FloatArray:
        storage = np.empty(value.shape[:-1] + (2 * value.shape[-1],))
        storage[..., ::2] = value
        result = storage[..., ::2]
        result.setflags(write=False)
        return result

    z = strided_readonly(rng.normal(size=layout.size))
    loads = LoadIncrement(*(strided_readonly(value) for value in (
        loads.dirichlet_increment, loads.neumann_increment, loads.cell_source_increment,
    )))
    inputs = (z, loads.dirichlet_increment, loads.neumann_increment, loads.cell_source_increment)
    saved = [value.copy() for value in inputs]

    def unexpected_rebuild(*args: object) -> None:
        pytest.fail("Residual evaluation must reuse cached discretization and geometry.")

    for name in ("_build_side_geometry", "_build_strain_matrix", "_build_traction_coefficients"):
        monkeypatch.setattr(assembler.operators, name, unexpected_rebuild)
    monkeypatch.setattr(assembler.elastic, "_assemble_matrices", unexpected_rebuild)
    first = assembler.evaluate_elastic(z, loads)
    other_z, other_loads = rng.normal(size=layout.size), _loads(assembler, 16)
    second = assembler.evaluate_elastic(other_z, other_loads)
    combined_loads = LoadIncrement(*(
        2.0 * getattr(loads, name) - 0.5 * getattr(other_loads, name)
        for name in ("dirichlet_increment", "neumann_increment", "cell_source_increment")
    ))
    combined = assembler.evaluate_elastic(2.0 * z - 0.5 * other_z, combined_loads)
    for name in ("residual", "face_increment", "traction_increment"):
        output = getattr(first, name)
        np.testing.assert_allclose(getattr(combined, name), 2.0 * output - 0.5 * getattr(second, name), atol=1e-12)
        assert output.dtype == np.float64
        assert output.flags.owndata
        assert all(not np.shares_memory(output, value) for value in inputs)
        output.fill(-999.0)
    for value, original in zip(inputs, saved, strict=True):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("field", ["z", "dirichlet_increment", "neumann_increment", "cell_source_increment"])
@pytest.mark.parametrize("invalid", ["shape", "short", "nan", "inf", "negative_inf", "complex", "string", "object", "bool"])
def test_invalid_candidate_and_loads_are_rejected(
    assembler: ResidualAssembler, field: str, invalid: str,
) -> None:
    z = np.zeros(assembler.layout.size)
    loads = _loads(assembler)
    value = (z if field == "z" else getattr(loads, field)).copy()
    if invalid == "shape":
        value = value[..., None]
    elif invalid == "short":
        value = value[..., :-1]
    elif invalid in ("nan", "inf", "negative_inf"):
        value.flat[-1] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[invalid]
    else:
        value = value.astype({"complex": "complex128", "string": "U32", "object": "O", "bool": "bool"}[invalid])
    if field == "z":
        z = value
    else:
        loads = replace(loads, **{field: value})
    with pytest.raises(ValueError, match=field):
        assembler.evaluate_elastic(z, loads)


@pytest.mark.parametrize("dtype", ["float32", "int64", "uint64"])
def test_real_numeric_dtypes_are_accepted(assembler: ResidualAssembler, dtype: str) -> None:
    layout = assembler.layout
    z = np.arange(layout.size, dtype=float) % 5
    loads = _loads(assembler)
    loads = LoadIncrement(*(cast(FloatArray, np.ones_like(value, dtype=dtype)) for value in (
        loads.dirichlet_increment, loads.neumann_increment, loads.cell_source_increment,
    )))
    evaluation = assembler.evaluate_elastic(cast(FloatArray, z.astype(dtype)), loads)
    assert evaluation.residual.dtype == np.float64
    assert np.all(np.isfinite(evaluation.residual))


@pytest.mark.parametrize("missing", ["mu", "side_stiffness", "rotation_transfer"])
def test_uninitialized_traction_is_rejected(assembler: ResidualAssembler, missing: str) -> None:
    setattr(assembler.operators, missing, None)
    with pytest.raises(ValueError, match="Initialize"):
        ResidualAssembler(assembler.operators, assembler.elastic)


@pytest.mark.parametrize("mismatch", ["grid", "mu", "boundary", "num_faces", "face_dofs", "tensor"])
def test_inconsistent_setup_is_rejected(assembler: ResidualAssembler, mismatch: str) -> None:
    operators, elastic = assembler.operators, assembler.elastic
    if mismatch == "grid":
        elastic.grid = operators.grid.copy()
    elif mismatch == "mu":
        assert operators.mu is not None
        operators.mu = operators.mu + 1.0
    elif mismatch == "boundary":
        elastic.bc = pp.BoundaryConditionVectorial(operators.grid, operators.grid.get_all_boundary_faces(), "dir")
    elif mismatch == "num_faces":
        operators.layout = replace(operators.layout, num_faces=operators.layout.num_faces + 1)
    elif mismatch == "face_dofs":
        operators.layout = replace(operators.layout, face_dofs=operators.layout.face_dofs + 1)
    else:
        elastic.data[pp.PARAMETERS][elastic.keyword]["fourth_order_tensor"] = object()
    with pytest.raises(ValueError):
        ResidualAssembler(operators, elastic)


@pytest.mark.parametrize("invalid", ["shape", "nan", "inf", "complex"])
def test_invalid_accumulation_is_rejected(assembler: ResidualAssembler, invalid: str) -> None:
    matrix = assembler.elastic.matrices.accum.copy()
    if invalid == "shape":
        matrix = matrix[:-1, :-1]
    elif invalid == "complex":
        matrix = matrix.astype(complex)
    else:
        matrix.data[0] = np.nan if invalid == "nan" else np.inf
    assembler.elastic.matrices.accum = sps.csr_array(matrix)
    with pytest.raises(ValueError, match="Accumulation"):
        ResidualAssembler(assembler.operators, assembler.elastic)


@pytest.mark.parametrize("field", ["face_increment", "traction_increment", "neumann_increment"])
@pytest.mark.parametrize("invalid", ["shape", "nan", "complex"])
def test_flux_helpers_reject_invalid_fields(
    assembler: ResidualAssembler, field: str, invalid: str,
) -> None:
    layout = assembler.layout
    fields: dict[str, FloatArray] = {
        "face_increment": np.zeros((layout.dim, layout.num_faces)),
        "traction_increment": np.zeros((layout.dim, assembler.operators.num_sides)),
        "neumann_increment": np.zeros((layout.dim, assembler.neumann_faces.size)),
    }
    value = fields[field]
    if invalid == "shape":
        value = value.ravel()
    elif invalid == "nan":
        value.flat[0] = np.nan
    else:
        value = value.astype(complex)
    fields[field] = value
    if field != "neumann_increment":
        with pytest.raises(ValueError):
            assembler._cell_flux_balance(fields["face_increment"], fields["traction_increment"])
    if field != "face_increment":
        with pytest.raises(ValueError):
            assembler._free_face_balance(fields["traction_increment"], fields["neumann_increment"])


@pytest.mark.parametrize("stage", ["cell", "face", "assembled"])
def test_overflow_from_finite_inputs_is_rejected(assembler: ResidualAssembler, stage: str) -> None:
    layout = assembler.layout
    displacement = np.zeros((layout.dim, layout.num_faces))
    traction = np.full((layout.dim, assembler.operators.num_sides), 1e308)
    if stage == "cell":
        with pytest.raises(ValueError, match="Cell flux"):
            assembler._cell_flux_balance(displacement, traction)
    elif stage == "face":
        with pytest.raises(ValueError, match="Free-face"):
            assembler._free_face_balance(traction, np.zeros((layout.dim, assembler.neumann_faces.size)))
    else:
        z = np.zeros(layout.size)
        z[layout.p_slice] = 1e308
        source = np.zeros(layout.cell_size)
        source[layout.p_slice] = 1e308
        assembler.accumulation = sps.eye_array(layout.cell_size, format="csr")
        loads = LoadIncrement(np.zeros((layout.dim, layout.dirichlet_faces.size)),
                              np.zeros((layout.dim, assembler.neumann_faces.size)), source)
        with np.errstate(over="ignore"), pytest.raises(ValueError, match="Residual must"):
            assembler.assemble(z, displacement, np.zeros_like(traction), loads)
