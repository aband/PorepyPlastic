"""Retained balances, material trials, and reduction to PorePy TPSA."""

from dataclasses import replace
from typing import cast

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

from core.layout import FloatArray
from core.residual import LoadIncrement, ResidualAssembler
from core.state import MaterialHistory
from core.tpsa_wrapper import ElasticTpsa, TpsaMatrices
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint, MaterialState
from material.scalar_hardening import (
    HardeningParameters, isotropic_hardening_K, kinematic_hardening_H,
)
from material.tensor import strain, stress


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def grid(request: pytest.FixtureRequest) -> pp.Grid:
    result = pp.TensorGrid(
        np.array([0.0, 0.3, 1.4]),
        np.array([0.0, 0.4, 1.0]),
        np.array([0.0, 0.2, 0.9]) if request.param == 3 else None,
    )
    result.compute_geometry()
    return result


def _boundary_conditions(grid: pp.Grid, boundary_type: str = "mixed") -> pp.BoundaryConditionVectorial:
    boundary = grid.get_all_boundary_faces()
    prescribed = {
        "dirichlet": boundary,
        "neumann": boundary[:0],
        "mixed": boundary[::2][::-1],
    }[boundary_type]
    return pp.BoundaryConditionVectorial(grid, prescribed, "dir")


def _material_points(grid: pp.Grid, *, uniform: bool = False) -> list[MaterialPoint]:
    mu = np.full(grid.num_cells, 2.0) if uniform else np.linspace(1.5, 3.0, grid.num_cells)
    lam = np.full(grid.num_cells, 3.0) if uniform else np.linspace(2.0, 4.0, grid.num_cells)
    result = []
    for shear, lame in zip(mu, lam, strict=True):
        # Preserve the original elastic fixtures through the material-point API.
        material = _materialProperty()
        material.young_modulus = float(shear * (3 * lame + 2 * shear) / (lame + shear))
        material.poisson_ratio = float(lame / (2 * (lame + shear)))
        result.append(MaterialPoint(
            material, HardeningParameters(0.05, 0.05, 0.2, 0.4, 0.0),
            vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H,
        ))
    return result


def _assembler(
    grid: pp.Grid, boundary_type: str = "mixed", *, uniform: bool = False,
) -> ResidualAssembler:
    return ResidualAssembler(
        grid, _boundary_conditions(grid, boundary_type), _material_points(grid, uniform=uniform),
    )


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
        normal = grid.face_normals[:grid.dim, face] * float(cast(float, grid.cell_faces[face, cell]))
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
    nonlinear = assembler.evaluate(np.zeros(layout.size), zero_loads, MaterialHistory.zeros(layout))
    assert evaluation.material_trial is None
    assert nonlinear.material_trial is not None
    for name in ("residual", "face_increment", "traction_increment"):
        np.testing.assert_array_equal(getattr(nonlinear, name), getattr(evaluation, name))
    np.testing.assert_array_equal(nonlinear.material_trial.epsilon, 0.0)
    np.testing.assert_array_equal(nonlinear.material_trial.stress_correction, 0.0)
    assert nonlinear.material_trial.material_states[0].alpha == 0.0
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


@pytest.mark.parametrize("uniform", [False, True], ids=["heterogeneous", "uniform"])
def test_constructor_derives_elastic_coefficients_from_material_points(grid: pp.Grid, uniform: bool) -> None:
    points = _material_points(grid, uniform=uniform)
    bc = _boundary_conditions(grid)
    assembler = ResidualAssembler(grid, bc, points, keyword="custom_mechanics")
    operators, elastic, layout = assembler.operators, assembler.elastic, assembler.layout
    mu = np.full(grid.num_cells, 2.0) if uniform else np.linspace(1.5, 3.0, grid.num_cells)
    lam = np.full(grid.num_cells, 3.0) if uniform else np.linspace(2.0, 4.0, grid.num_cells)
    tensor = elastic.data[pp.PARAMETERS]["custom_mechanics"]["fourth_order_tensor"]
    assert elastic.keyword == "custom_mechanics"
    assert elastic.grid is operators.grid is grid
    assert elastic.bc is bc
    assert operators.layout is layout
    assert isinstance(assembler.material_points, tuple)
    assert all(a is b for a, b in zip(assembler.material_points, points, strict=True))
    assert operators.mu is not None
    assert operators.side_stiffness is not None
    np.testing.assert_allclose(operators.mu, mu, rtol=1e-14)
    np.testing.assert_allclose(tensor.mu, mu, rtol=1e-14)
    np.testing.assert_allclose(tensor.lmbda, lam, rtol=1e-14)
    np.testing.assert_allclose(operators.side_stiffness, 2 * mu[operators.side_cells] / operators.side_distances)
    assert operators.rotation_transfer is not None
    expected = np.concatenate((
        np.zeros(layout.dim * grid.num_cells),
        np.repeat(grid.cell_volumes / mu, layout.rotation_dim), grid.cell_volumes / lam,
    ))
    np.testing.assert_allclose(assembler.accumulation.diagonal(), expected, rtol=1e-14)
    # Changing the caller's collection must not change the assembler's cell order.
    first = points[0]
    points.clear()
    assert len(assembler.material_points) == grid.num_cells
    assert assembler.material_points[0] is first


@pytest.mark.parametrize("count", ["empty", "short", "long"])
def test_constructor_requires_one_material_point_per_cell(grid: pp.Grid, count: str) -> None:
    points = _material_points(grid)
    points = {"empty": [], "short": points[:-1], "long": points + [points[0]]}[count]
    with pytest.raises(ValueError, match="one material point per cell"):
        ResidualAssembler(grid, _boundary_conditions(grid), points)


@pytest.mark.parametrize(("property_name", "value"), [
    ("young_modulus", 0.0), ("young_modulus", -1.0),
    ("young_modulus", np.nan), ("young_modulus", np.inf),
    ("poisson_ratio", 0.0), ("poisson_ratio", -0.1),
])
def test_constructor_rejects_invalid_derived_coefficients(
    grid: pp.Grid, property_name: str, value: float,
) -> None:
    points = _material_points(grid)
    setattr(points[0].material, property_name, value)
    with pytest.raises(ValueError, match="positive"):
        ResidualAssembler(grid, _boundary_conditions(grid), points)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_accumulation_is_rejected(
    grid: pp.Grid, monkeypatch: pytest.MonkeyPatch, value: float,
) -> None:
    build = ElasticTpsa._assemble_matrices

    def nonfinite_accumulation(elastic: ElasticTpsa) -> TpsaMatrices:
        matrices = build(elastic)
        matrices.accum.data[0] = value
        return matrices

    monkeypatch.setattr(ElasticTpsa, "_assemble_matrices", nonfinite_accumulation)
    with pytest.raises(ValueError, match="Accumulation"):
        _assembler(grid)


def _affine_material_increment(assembler: ResidualAssembler) -> tuple[FloatArray, FloatArray]:
    nd = assembler.layout.dim
    gradient = np.array([
        [0.04, 0.012, -0.004], [-0.004, -0.01, 0.009], [0.002, -0.003, -0.015],
    ])[:nd, :nd]
    faces = gradient @ assembler.operators.grid.face_centers[:nd]
    faces += np.arange(1, nd + 1)[:, None] / 10.0
    epsilon = np.zeros((3, 3, assembler.layout.num_cells))
    epsilon[:nd, :nd] = (0.5 * (gradient + gradient.T))[:, :, None]
    return faces, epsilon


def _hooke(point: MaterialPoint, epsilon: FloatArray) -> FloatArray:
    return np.asarray(
        2 * point.material.isotropic_shear_modulus * epsilon
        + point.material.lame_parameter * np.trace(epsilon) * np.eye(3), dtype=np.float64,
    )


def _reference_material_update(
    point: MaterialPoint, old: MaterialState, increment: FloatArray,
) -> MaterialState:
    """Closed-form J2 update for the fixtures' linear mixed hardening."""
    parameters = point.parameters
    assert parameters.sigma_y == parameters.sigma_u
    predictor = old.stress.to_numpy() + _hooke(point, increment)
    backstress = old.backstress.to_numpy()
    shifted = predictor - np.trace(predictor) / 3 * np.eye(3)
    shifted -= backstress - np.trace(backstress) / 3 * np.eye(3)
    norm = float(np.linalg.norm(shifted))
    c = np.sqrt(2.0 / 3.0)
    radius = c * (parameters.sigma_y + parameters.theta * parameters.H_bar * old.alpha)
    gamma = max(0.0, norm - radius) / (
        2 * point.material.isotropic_shear_modulus + 2 / 3 * parameters.H_bar
    )
    plastic_increment = np.zeros((3, 3)) if gamma == 0 else gamma * shifted / norm
    return MaterialState(
        stress(predictor - _hooke(point, plastic_increment)),
        strain(old.plastic_strain.to_numpy() + plastic_increment),
        stress(backstress + 2 / 3 * (1 - parameters.theta) * parameters.H_bar * plastic_increment),
        old.alpha + c * gamma,
    )


def _assert_material_state(actual: MaterialState, expected: MaterialState, *, exact: bool = False) -> None:
    if exact:
        assert actual.alpha == expected.alpha
    else:
        assert actual.alpha == pytest.approx(expected.alpha, rel=1e-11, abs=1e-13)
    for name in ("stress", "plastic_strain", "backstress"):
        a, b = getattr(actual, name).to_numpy(), getattr(expected, name).to_numpy()
        if exact:
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("scale", [0.01, 1.0], ids=["elastic", "plastic"])
def test_material_trial_reconstructs_affine_strain_and_updates_each_cell(
    assembler: ResidualAssembler, scale: float,
) -> None:
    faces, epsilon = _affine_material_increment(assembler)
    committed = MaterialHistory.zeros(assembler.layout)
    trial = assembler._evaluate_material_trial(scale * faces, committed)
    np.testing.assert_allclose(trial.epsilon, scale * epsilon, atol=1e-14)
    assert len(trial.material_states) == assembler.layout.num_cells
    assert trial.stress_correction.shape == epsilon.shape
    for cell, point in enumerate(assembler.material_points):
        expected = _reference_material_update(point, MaterialState(), scale * epsilon[:, :, cell])
        state = trial.material_states[cell]
        _assert_material_state(state, expected)
        assert (state.alpha > 0) == (scale == 1.0)
        correction = _hooke(point, expected.plastic_strain.to_numpy())
        np.testing.assert_allclose(trial.stress_correction[:, :, cell], correction, atol=1e-12)
    if assembler.layout.dim == 2:
        np.testing.assert_array_equal(trial.epsilon[2], 0.0)
        np.testing.assert_array_equal(trial.epsilon[:, 2], 0.0)
        assert trial.material_states[0].stress.to_numpy()[2, 2] != 0


@pytest.mark.parametrize("factor", [1.0, 0.99], ids=["hold", "unload"])
def test_material_trials_start_from_committed_history_and_do_not_commit(
    assembler: ResidualAssembler, factor: float,
) -> None:
    faces, epsilon = _affine_material_increment(assembler)
    initial = assembler._evaluate_material_trial(faces, MaterialHistory.zeros(assembler.layout))
    committed = MaterialHistory(assembler.layout.dim, initial.epsilon, initial.material_states)
    snapshot = committed.copy()
    # Material points supply configuration; their own state slots are not the
    # history used for this candidate and must remain untouched.
    slots = []
    for point in assembler.material_points:
        point.committed = MaterialState(alpha=8.0)
        point.trial = MaterialState(alpha=9.0)
        slots.append((point.committed, point.trial))
    increment = (factor - 1) * faces
    increment.setflags(write=False)
    committed.epsilon.setflags(write=False)
    first = assembler._evaluate_material_trial(increment, committed)
    second = assembler._evaluate_material_trial(increment, committed)
    for trial in (first, second):
        np.testing.assert_allclose(trial.epsilon, factor * epsilon, atol=1e-14)
        np.testing.assert_allclose(trial.stress_correction, 0.0, atol=1e-12)
        for cell, point in enumerate(assembler.material_points):
            old, state = snapshot.material_states[cell], trial.material_states[cell]
            expected_stress = old.stress.to_numpy() + _hooke(point, (factor - 1) * epsilon[:, :, cell])
            np.testing.assert_allclose(state.stress.to_numpy(), expected_stress, atol=1e-12)
            assert state.alpha == old.alpha
            for name in ("plastic_strain", "backstress"):
                np.testing.assert_array_equal(getattr(state, name).to_numpy(), getattr(old, name).to_numpy())
    first.epsilon.fill(-999.0)
    first.stress_correction.fill(-999.0)
    for state in first.material_states:
        state.alpha = -999.0
        state.stress = stress(np.full((3, 3), -999.0))
    np.testing.assert_array_equal(committed.epsilon, snapshot.epsilon)
    np.testing.assert_allclose(second.epsilon, factor * epsilon, atol=1e-14)
    np.testing.assert_allclose(second.stress_correction, 0.0, atol=1e-12)
    for cell, point in enumerate(assembler.material_points):
        old = committed.material_states[cell]
        assert old.alpha == snapshot.material_states[cell].alpha == second.material_states[cell].alpha
        for name in ("stress", "plastic_strain", "backstress"):
            np.testing.assert_array_equal(getattr(old, name).to_numpy(), getattr(snapshot.material_states[cell], name).to_numpy())
            assert not np.shares_memory(
                getattr(old, name).to_numpy(copy=False), getattr(second.material_states[cell], name).to_numpy(copy=False),
            )
        assert point.committed is slots[cell][0] and point.committed.alpha == 8.0
        assert point.trial is slots[cell][1] and point.trial.alpha == 9.0


def test_material_trial_forwards_return_map_tolerances(
    assembler: ResidualAssembler, monkeypatch: pytest.MonkeyPatch,
) -> None:
    update = MaterialPoint.update
    calls: list[tuple[float, float]] = []

    def recording_update(
        point: MaterialPoint, strain_increment: strain, tol: float = 1e-10, *, rtol: float = 1e-12,
    ) -> tuple[MaterialState, None]:
        calls.append((tol, rtol))
        return update(point, strain_increment, tol, rtol=rtol)

    monkeypatch.setattr(MaterialPoint, "update", recording_update)
    faces, _ = _affine_material_increment(assembler)
    assembler._evaluate_material_trial(faces, MaterialHistory.zeros(assembler.layout), tol=2e-11, rtol=3e-13)
    assert calls == [(2e-11, 3e-13)] * assembler.layout.num_cells


@pytest.mark.parametrize("invalid", ["dimension", "strain_shape", "state_count"])
def test_material_trial_rejects_incompatible_committed_history(
    assembler: ResidualAssembler, invalid: str,
) -> None:
    committed = MaterialHistory.zeros(assembler.layout)
    if invalid == "dimension":
        committed.dim = 5 - assembler.layout.dim
        message = "history.*layout"
    elif invalid == "strain_shape":
        committed.epsilon = committed.epsilon[:, :, :-1]
        message = "Committed strain.*shape"
    else:
        committed.material_states.pop()
        message = "one material point and history per cell"
    faces, _ = _affine_material_increment(assembler)
    with pytest.raises(ValueError, match=message):
        assembler._evaluate_material_trial(faces, committed)


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed"])
@pytest.mark.parametrize(("previous_factor", "increment_factor"), [
    (0.0, 0.01), (0.0, 1.0), (1.0, -0.01), (1.0, 0.3),
], ids=["elastic", "plastic", "unloading", "reloading"])
def test_nonlinear_evaluation_matches_material_and_balance_references(
    grid: pp.Grid, boundary_type: str, previous_factor: float, increment_factor: float,
) -> None:
    assembler = _assembler(grid, boundary_type)
    layout, operators = assembler.layout, assembler.operators
    faces, epsilon = _affine_material_increment(assembler)
    previous = previous_factor * epsilon
    committed = MaterialHistory(layout.dim, previous, [
        _reference_material_update(point, MaterialState(), previous[:, :, cell])
        for cell, point in enumerate(assembler.material_points)
    ])
    faces *= increment_factor
    z = layout.pack(0.01 * np.random.default_rng(37).normal(size=layout.cell_size), faces[:, layout.free_faces])
    loads = replace(_loads(assembler), dirichlet_increment=faces[:, layout.dirichlet_faces])
    elastic = assembler.evaluate_elastic(z, loads)
    actual = assembler.evaluate(z, loads, committed)
    trial = actual.material_trial
    assert elastic.material_trial is None
    assert trial is not None
    assert len(trial.material_states) == grid.num_cells
    np.testing.assert_allclose(actual.face_increment, faces, atol=1e-14)
    np.testing.assert_allclose(trial.epsilon, (previous_factor + increment_factor) * epsilon, atol=1e-14)

    correction = np.empty_like(epsilon)
    for cell, point in enumerate(assembler.material_points):
        old = committed.material_states[cell]
        increment = increment_factor * epsilon[:, :, cell]
        expected = _reference_material_update(point, old, increment)
        _assert_material_state(trial.material_states[cell], expected)
        correction[:, :, cell] = old.stress.to_numpy() + _hooke(point, increment) - expected.stress.to_numpy()
        if increment_factor in (0.01, -0.01):
            assert trial.material_states[cell].alpha == old.alpha
        else:
            assert trial.material_states[cell].alpha > old.alpha
    np.testing.assert_allclose(trial.stress_correction, correction, atol=1e-12)

    # Use grid normals and explicit sums, independently of the correction and
    # assembly helpers. Keep nonzero cell and boundary loads in every case.
    expected_traction = elastic.traction_increment.copy()
    delta_momentum = np.zeros((grid.dim, grid.num_cells))
    delta_face = np.zeros((grid.dim, grid.num_faces))
    for side, (face, side_cell) in enumerate(zip(operators.side_faces, operators.side_cells, strict=True)):
        normal = float(cast(float, grid.cell_faces[face, side_cell])) * grid.face_normals[:grid.dim, face]
        delta = -correction[:grid.dim, :grid.dim, side_cell] @ normal
        expected_traction[:, side] += delta
        delta_momentum[:, side_cell] += delta
        delta_face[:, face] += delta
    expected_residual = elastic.residual.copy()
    expected_residual[layout.u_slice] += delta_momentum.ravel(order="F")
    expected_residual[layout.face_slice] += delta_face[:, layout.free_faces].ravel(order="F")
    np.testing.assert_allclose(actual.traction_increment, expected_traction, atol=1e-12)
    for block in (layout.u_slice, layout.r_slice, layout.p_slice, layout.face_slice):
        np.testing.assert_allclose(actual.residual[block], expected_residual[block], atol=1e-12)
    if increment_factor in (0.01, -0.01):
        np.testing.assert_allclose(actual.residual, elastic.residual, atol=1e-12)
        np.testing.assert_allclose(trial.stress_correction, 0.0, atol=1e-12)
    else:
        assert np.linalg.norm(actual.residual - elastic.residual) > 1e-3
        assert np.linalg.norm(trial.stress_correction) > 1e-2


@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed"])
def test_nonlinear_evaluation_is_repeatable_and_preserves_committed_history(
    grid: pp.Grid, boundary_type: str,
) -> None:
    assembler = _assembler(grid, boundary_type)
    layout = assembler.layout
    faces, epsilon = _affine_material_increment(assembler)
    committed = MaterialHistory(layout.dim, epsilon, [
        _reference_material_update(point, MaterialState(), epsilon[:, :, cell])
        for cell, point in enumerate(assembler.material_points)
    ])
    snapshot = committed.copy()
    committed.epsilon.setflags(write=False)
    # Stored points supply configuration, while the caller supplies the history.
    slots = []
    for point in assembler.material_points:
        point.committed = MaterialState(alpha=8.0)
        point.trial = MaterialState(alpha=9.0)
        slots.append((point.committed, point.trial))
    z = layout.pack(np.zeros(layout.cell_size), 0.3 * faces[:, layout.free_faces])
    loads = replace(_loads(assembler), dirichlet_increment=0.3 * faces[:, layout.dirichlet_faces])
    inputs = (z, loads.dirichlet_increment, loads.neumann_increment, loads.cell_source_increment)
    saved_inputs = [value.copy() for value in inputs]
    for value in inputs:
        value.setflags(write=False)

    first = assembler.evaluate(z, loads, committed)
    other_z = z.copy()
    other_z[layout.face_slice] *= -0.5
    other_loads = replace(loads, dirichlet_increment=-0.5 * loads.dirichlet_increment)
    assembler.evaluate(other_z, other_loads, committed)
    second = assembler.evaluate(z, loads, committed)
    assert first.material_trial is not None and second.material_trial is not None
    for name in ("residual", "face_increment", "traction_increment"):
        a, b = getattr(first, name), getattr(second, name)
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(a, b)
        assert all(not np.shares_memory(a, value) for value in inputs)
        a.fill(-999.0)
    for name in ("epsilon", "stress_correction"):
        a, b = getattr(first.material_trial, name), getattr(second.material_trial, name)
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(a, b)
        assert not np.shares_memory(a, committed.epsilon)
        a.fill(-999.0)
    for cell, (a, b) in enumerate(zip(first.material_trial.material_states, second.material_trial.material_states, strict=True)):
        _assert_material_state(a, b, exact=True)
        assert a is not b and a is not committed.material_states[cell]
        for name in ("stress", "plastic_strain", "backstress"):
            array = getattr(a, name).to_numpy(copy=False)
            assert not np.shares_memory(array, getattr(b, name).to_numpy(copy=False))
            assert not np.shares_memory(array, getattr(committed.material_states[cell], name).to_numpy(copy=False))
        a.alpha = -999.0

    # Mutating one returned evaluation must not affect a later evaluation.
    third = assembler.evaluate(z, loads, committed)
    assert third.material_trial is not None
    for name in ("residual", "face_increment", "traction_increment"):
        np.testing.assert_array_equal(getattr(second, name), getattr(third, name))
    for name in ("epsilon", "stress_correction"):
        np.testing.assert_array_equal(getattr(second.material_trial, name), getattr(third.material_trial, name))
    for a, b in zip(second.material_trial.material_states, third.material_trial.material_states, strict=True):
        _assert_material_state(a, b, exact=True)
    np.testing.assert_array_equal(committed.epsilon, snapshot.epsilon)
    for cell, point in enumerate(assembler.material_points):
        _assert_material_state(committed.material_states[cell], snapshot.material_states[cell], exact=True)
        assert point.committed is slots[cell][0]
        assert point.trial is slots[cell][1]
        _assert_material_state(point.committed, MaterialState(alpha=8.0), exact=True)
        _assert_material_state(point.trial, MaterialState(alpha=9.0), exact=True)
    for value, saved in zip(inputs, saved_inputs, strict=True):
        np.testing.assert_array_equal(value, saved)


def test_nonlinear_evaluation_forwards_tolerances_and_updates_each_cell_once(
    assembler: ResidualAssembler, monkeypatch: pytest.MonkeyPatch,
) -> None:
    update = MaterialPoint.update
    calls: list[tuple[int, float, float]] = []

    def recording_update(
        point: MaterialPoint, strain_increment: strain, tol: float = 1e-10, *, rtol: float = 1e-12,
    ) -> tuple[MaterialState, None]:
        calls.append((id(point.material), tol, rtol))
        return update(point, strain_increment, tol, rtol=rtol)

    monkeypatch.setattr(MaterialPoint, "update", recording_update)
    layout = assembler.layout
    faces, _ = _affine_material_increment(assembler)
    z = layout.pack(np.zeros(layout.cell_size), faces[:, layout.free_faces])
    loads = replace(_loads(assembler), dirichlet_increment=faces[:, layout.dirichlet_faces])
    assembler.evaluate(z, loads, MaterialHistory.zeros(layout), tol=2e-11, rtol=3e-13)
    assert calls == [(id(point.material), 2e-11, 3e-13) for point in assembler.material_points]


@pytest.mark.parametrize("field", ["z", "dirichlet_increment", "neumann_increment", "cell_source_increment"])
def test_nonlinear_evaluation_validates_inputs_before_material_updates(
    assembler: ResidualAssembler, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    def unexpected_update(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid global inputs must be rejected before any local material update.")

    monkeypatch.setattr(MaterialPoint, "update", unexpected_update)
    z = np.zeros(assembler.layout.size)
    loads = _loads(assembler)
    if field == "z":
        z = z[:-1]
    elif field == "dirichlet_increment":
        loads = replace(loads, dirichlet_increment=loads.dirichlet_increment.astype(complex))
    elif field == "neumann_increment":
        loads.neumann_increment.flat[0] = np.nan
    else:
        loads = replace(loads, cell_source_increment=loads.cell_source_increment[:, None])
    with pytest.raises(ValueError, match=field):
        assembler.evaluate(z, loads, MaterialHistory.zeros(assembler.layout))


def test_traction_correction_matches_independent_surface_tractions(grid: pp.Grid) -> None:
    # Oblique normals exercise all tensor components; unequal face areas expose
    # missing or repeated area factors. Rotate a 2D grid within its own plane.
    c, s = np.cos(0.37), np.sin(0.37)
    transform = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if grid.dim == 3:
        transform = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ transform
    grid.nodes[:] = transform @ grid.nodes
    grid.compute_geometry()
    assembler = _assembler(grid)
    operators = assembler.operators
    rng = np.random.default_rng(24)
    traction = rng.normal(size=(grid.dim, operators.num_sides))
    correction = rng.normal(size=(3, 3, grid.num_cells))
    correction = 0.5 * (correction + correction.swapaxes(0, 1))
    expected = np.empty_like(traction)
    for side, (face, cell) in enumerate(zip(operators.side_faces, operators.side_cells, strict=True)):
        # Independent reference: PorePy's face normals already include area.
        area_normal = float(cast(float, grid.cell_faces[face, cell])) * grid.face_normals[:grid.dim, face]
        expected[:, side] = traction[:, side] - correction[:grid.dim, :grid.dim, cell] @ area_normal

    actual = assembler._correct_one_sided_traction(traction, correction)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert actual.shape == traction.shape


def test_zero_traction_correction_preserves_elastic_result(assembler: ResidualAssembler) -> None:
    z = np.random.default_rng(25).normal(size=assembler.layout.size)
    elastic = assembler.evaluate_elastic(z, _loads(assembler)).traction_increment
    correction = np.zeros((3, 3, assembler.layout.num_cells))
    actual = assembler._correct_one_sided_traction(elastic, correction)
    np.testing.assert_array_equal(actual, elastic)
    assert not np.shares_memory(actual, elastic)


@pytest.mark.parametrize("uniform", [True, False], ids=["continuous", "jump"])
def test_traction_correction_uses_each_side_cell_without_averaging(
    assembler: ResidualAssembler, uniform: bool,
) -> None:
    grid, operators = assembler.operators.grid, assembler.operators
    tensor = np.array([[2.0, 0.7, -0.3], [0.7, -1.0, 0.4], [-0.3, 0.4, 3.0]])
    factors = np.ones(grid.num_cells) if uniform else 1.0 + np.arange(grid.num_cells)
    correction = tensor[:, :, None] * factors
    actual = assembler._correct_one_sided_traction(
        np.zeros((grid.dim, operators.num_sides)), correction,
    )
    interior = np.setdiff1d(np.arange(grid.num_faces), grid.get_all_boundary_faces())
    for face in interior:
        sides = np.flatnonzero(operators.side_faces == face)
        left, right = operators.side_cells[sides]
        area_normal = float(grid.cell_faces[face, left]) * grid.face_normals[:grid.dim, face]
        jump = correction[:grid.dim, :grid.dim, left] - correction[:grid.dim, :grid.dim, right]
        expected = -jump @ area_normal
        np.testing.assert_allclose(actual[:, sides].sum(axis=1), expected, atol=1e-13)
        if not uniform:
            assert np.linalg.norm(expected) > 0.0
    # A constant tensor within each cell has zero net flux over its closed surface.
    for cell in range(grid.num_cells):
        np.testing.assert_allclose(actual[:, operators.side_cells == cell].sum(axis=1), 0.0, atol=1e-13)


def test_traction_correction_uses_out_of_plane_components_only_in_3d(
    assembler: ResidualAssembler,
) -> None:
    layout, operators = assembler.layout, assembler.operators
    correction = np.repeat(
        np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -3.0], [2.0, -3.0, 4.0]])[:, :, None],
        layout.num_cells, axis=2,
    )
    actual = assembler._correct_one_sided_traction(
        np.zeros((layout.dim, operators.num_sides)), correction,
    )
    if layout.dim == 2:
        np.testing.assert_array_equal(actual, 0.0)
    else:
        # For the positive-z boundary, Q n = (2, -3, 4).
        grid = operators.grid
        sides = np.flatnonzero(operators.side_normals[2] > 0.5)
        expected = -np.array([2.0, -3.0, 4.0])[:, None] * grid.face_areas[operators.side_faces[sides]]
        np.testing.assert_allclose(actual[:, sides], expected, atol=1e-13)


@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "uint64"])
def test_traction_correction_preserves_readonly_strided_inputs_and_returns_float64(
    assembler: ResidualAssembler, dtype: str,
) -> None:
    layout, operators = assembler.layout, assembler.operators

    def readonly_strided(values: FloatArray) -> FloatArray:
        storage = np.empty(values.shape[:-1] + (2 * values.shape[-1],), dtype=dtype)
        storage[..., ::2] = values
        result = storage[..., ::2]
        result.setflags(write=False)
        return cast(FloatArray, result)

    traction = readonly_strided(np.ones((layout.dim, operators.num_sides)))
    correction = readonly_strided(np.repeat((2.0 * np.eye(3))[:, :, None], layout.num_cells, axis=2))
    saved = (traction.copy(), correction.copy())
    grid = operators.grid
    area_normals = grid.face_normals[:layout.dim, operators.side_faces] * operators.side_signs
    expected = np.ones_like(traction, dtype=np.float64) - 2.0 * area_normals
    first = assembler._correct_one_sided_traction(traction, correction)
    second = assembler._correct_one_sided_traction(traction, correction)
    np.testing.assert_allclose(first, expected, atol=1e-13)
    assert first.dtype == np.float64
    assert first.flags.owndata
    assert not np.shares_memory(first, second)
    for value in (traction, correction):
        assert not value.flags.c_contiguous and not value.flags.writeable
        assert not np.shares_memory(first, value)
    first.fill(-999.0)
    np.testing.assert_allclose(second, expected, atol=1e-13)
    np.testing.assert_array_equal(traction, saved[0])
    np.testing.assert_array_equal(correction, saved[1])


@pytest.mark.parametrize("field", ["elastic_traction", "stress_correction"])
@pytest.mark.parametrize("invalid", ["shape", "short", "nan", "inf", "negative_inf", "complex", "string", "object", "bool"])
def test_traction_correction_rejects_invalid_fields(
    assembler: ResidualAssembler, field: str, invalid: str,
) -> None:
    layout, operators = assembler.layout, assembler.operators
    fields: dict[str, FloatArray] = {
        "elastic_traction": np.zeros((layout.dim, operators.num_sides)),
        "stress_correction": np.zeros((3, 3, layout.num_cells)),
    }
    value = fields[field]
    if invalid == "shape":
        value = value.ravel()
    elif invalid == "short":
        value = value[..., :-1]
    elif invalid in ("nan", "inf", "negative_inf"):
        value.flat[0] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[invalid]
    else:
        value = value.astype({"complex": "complex128", "string": "U32", "object": "O", "bool": "bool"}[invalid])
    fields[field] = value
    with pytest.raises(ValueError, match=field):
        assembler._correct_one_sided_traction(fields["elastic_traction"], fields["stress_correction"])


@pytest.mark.parametrize("component", [(0, 1), (0, 2)], ids=["xy", "xz"])
def test_traction_correction_rejects_nonsymmetric_tensors(
    assembler: ResidualAssembler, component: tuple[int, int],
) -> None:
    layout = assembler.layout
    correction = np.zeros((3, 3, layout.num_cells))
    correction[component[0], component[1], 0] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        assembler._correct_one_sided_traction(
            np.zeros((layout.dim, assembler.operators.num_sides)), correction,
        )


def test_traction_correction_rejects_overflow_from_finite_inputs(assembler: ResidualAssembler) -> None:
    layout = assembler.layout
    with pytest.raises(ValueError, match="Corrected traction.*finite"):
        assembler._correct_one_sided_traction(
            np.full((layout.dim, assembler.operators.num_sides), 1.7e308),
            np.full((3, 3, layout.num_cells), 1.7e308),
        )


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
