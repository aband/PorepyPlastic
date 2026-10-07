"""State isolation, fixed operators, and TPSA auxiliary-equation checks."""

from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.grid import Grid
from porepy.grids.structured import CartGrid

from coupling.plane_strain import PlaneStrainTpsa
from coupling.residual import TpsaOperators, TpsaState
from material_state import MaterialState
from tensor import strain, stress


@pytest.fixture
def grid() -> Grid:
    return CartGrid(np.array([2, 1]), physdims=np.array([1.0, 1.0]))


def test_zero_state_uses_old_boundary_data_not_target_load() -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.prepare_simulation()
    target = case.bc_values.copy()
    assert np.max(np.abs(target)) > 0.0
    state = TpsaState.zeros(case.grid)
    nc, nf = case.grid.num_cells, case.grid.num_faces
    for name, shape in {
        "x": (4 * nc,), "bc_values": (2, nf),
        "epsilon": (3, 3, nc), "traction": (2, nf),
    }.items():
        value = getattr(state, name)
        assert value.shape == shape and value.dtype == np.float64
        np.testing.assert_array_equal(value, 0.0)
    assert len(state.material_states) == nc
    for cell, history in enumerate(state.material_states):
        assert history is not case.material_points[cell].committed
        assert history.alpha == 0.0
        for name in ("stress", "plastic_strain", "backstress"):
            np.testing.assert_array_equal(getattr(history, name).to_numpy(), np.zeros((3, 3)))
    np.testing.assert_array_equal(case.bc_values, target)
    assert case.x is None


def test_constructor_copies_inputs_and_separates_cell_histories(grid: Grid) -> None:
    source = TpsaState.zeros(grid)
    source.x[:] = np.arange(source.x.size)
    source.bc_values[:] = 1e-4
    source.epsilon[0, 0] = 1e-3
    source.traction[:] = 12.0
    history = MaterialState(
        stress=stress(np.diag([100.0, 40.0, 30.0])),
        plastic_strain=strain(np.diag([2e-4, -1e-4, -1e-4])),
        backstress=stress(np.diag([10.0, -5.0, -5.0])), alpha=2e-4,
    )
    # Even repeated references supplied by a caller must produce separate cells.
    state = replace(source, material_states=[history] * grid.num_cells)
    for name in ("x", "bc_values", "epsilon", "traction"):
        original, owned = getattr(source, name), getattr(state, name)
        np.testing.assert_array_equal(owned, original)
        assert not np.shares_memory(owned, original)
        original[:] = 0.0
        assert np.any(owned != 0.0)
    for name in ("stress", "plastic_strain", "backstress"):
        owned = [getattr(item, name).to_numpy(copy=False) for item in state.material_states]
        assert not np.shares_memory(owned[0], getattr(history, name).to_numpy(copy=False))
        assert not np.shares_memory(owned[0], owned[1])
        np.testing.assert_array_equal(owned[0], owned[1])
    history.alpha = 1.0
    assert state.material_states[0].alpha == 2e-4
    state.material_states[0].plastic_strain = strain.zeros((3, 3))
    assert state.material_states[1].plastic_strain.to_numpy()[2, 2] == -1e-4
    assert state.material_states[1].stress.to_numpy()[2, 2] == 30.0


def test_trial_copy_leaves_committed_arrays_and_histories_unchanged(grid: Grid) -> None:
    committed = TpsaState.zeros(grid)
    trial = committed.copy()
    for name in ("x", "bc_values", "epsilon", "traction"):
        original, candidate = getattr(committed, name), getattr(trial, name)
        assert not np.shares_memory(original, candidate)
        candidate.flat[0] = 1e-3
        np.testing.assert_array_equal(original, 0.0)
    assert trial.material_states is not committed.material_states
    for name in ("stress", "plastic_strain", "backstress"):
        for before, after in zip(committed.material_states, trial.material_states):
            assert not np.shares_memory(
                getattr(before, name).to_numpy(copy=False),
                getattr(after, name).to_numpy(copy=False),
            )
    trial.material_states[0].alpha = 0.2
    trial.material_states[0].stress = stress(np.eye(3))
    assert committed.material_states[0].alpha == 0.0
    np.testing.assert_array_equal(committed.material_states[0].stress.to_numpy(), 0.0)
    assert trial.material_states[1].alpha == 0.0
    repeated = committed.copy()
    np.testing.assert_array_equal(repeated.x, 0.0)
    assert repeated.material_states[0].alpha == 0.0


@pytest.mark.parametrize("dimension", [1, 3])
def test_zero_state_rejects_non_2d_grid(dimension: int) -> None:
    with pytest.raises(ValueError, match="2D"):
        TpsaState.zeros(CartGrid(np.full(dimension, 2)))


@pytest.mark.parametrize(
    "name, value",
    [
        ("x", np.zeros(7)),
        ("x", np.full(8, np.nan)),
        ("bc_values", np.zeros(14)),
        ("bc_values", np.zeros((3, 7))),
        ("bc_values", np.full((2, 7), np.inf)),
        ("epsilon", np.zeros((2, 2, 2))),
        ("epsilon", np.full((3, 3, 2), np.nan)),
        ("traction", np.zeros((7, 2))),
        ("traction", np.full((2, 7), np.inf)),
    ],
)
def test_rejects_invalid_state_arrays(
    grid: Grid, name: str, value: NDArray[np.float64],
) -> None:
    state = TpsaState.zeros(grid)
    setattr(state, name, value)
    with pytest.raises(ValueError, match="finite|boundary"):
        replace(state)


@pytest.mark.parametrize("component", [(0, 1), (2, 2), (0, 2), (1, 2)])
def test_rejects_invalid_total_plane_strain(grid: Grid, component: tuple[int, int]) -> None:
    state = TpsaState.zeros(grid)
    i, j = component
    state.epsilon[i, j, 0] = 1e-3
    if j == 2:
        state.epsilon[j, i, 0] = 1e-3
    with pytest.raises(ValueError, match="symmetric|plane strain"):
        replace(state)


def test_rejects_inconsistent_history_count(grid: Grid) -> None:
    with pytest.raises(ValueError, match="shape"):
        replace(TpsaState.zeros(grid), material_states=[MaterialState()])
    with pytest.raises(ValueError, match="nonempty"):
        replace(TpsaState.zeros(grid), material_states=[])


@pytest.mark.parametrize("alpha", [-1.0, np.nan, np.inf])
def test_rejects_invalid_material_alpha(grid: Grid, alpha: float) -> None:
    state = TpsaState.zeros(grid)
    state.material_states[0].alpha = alpha
    with pytest.raises(ValueError, match="alpha"):
        replace(state)


@pytest.mark.parametrize("name", ["stress", "plastic_strain", "backstress"])
@pytest.mark.parametrize("defect", ["shape", "nonfinite", "asymmetric"])
def test_rejects_invalid_material_tensors(grid: Grid, name: str, defect: str) -> None:
    state = TpsaState.zeros(grid)
    value = np.zeros((2, 2)) if defect == "shape" else np.zeros((3, 3))
    if defect == "nonfinite":
        value[0, 0] = np.nan
    elif defect == "asymmetric":
        value[0, 1] = 1.0
    tensor_type = strain if name == "plastic_strain" else stress
    setattr(state.material_states[0], name, tensor_type(value))
    with pytest.raises(ValueError, match=f"Material {name}"):
        replace(state)


@pytest.fixture
def prepared_case() -> PlaneStrainTpsa:
    case = PlaneStrainTpsa(cells_per_axis=3)
    case.prepare_simulation()
    return case


@pytest.mark.parametrize("cells_per_axis", [1, 3])
def test_auxiliaries_reproduce_elastic_solution(cells_per_axis: int) -> None:
    case = PlaneStrainTpsa(
        cells_per_axis=cells_per_axis,
        displacement_gradient=np.array([[1e-4, 2e-4], [-3e-4, 4e-4]]),
        translation=np.array([2e-5, -1e-5]),
    )
    x = case.solve()
    saved_x, saved_boundary = x.copy(), case.bc_values.copy()
    operators = TpsaOperators(case)
    nc, nf = case.grid.num_cells, case.grid.num_faces
    rhs = operators.assemble_rhs(case.bc_values)
    r, p = operators.solve_auxiliary(x[:2 * nc], rhs)
    np.testing.assert_allclose(r, x[2 * nc:3 * nc], rtol=1e-10, atol=1e-5)
    np.testing.assert_allclose(p, x[3 * nc:], rtol=1e-10, atol=1e-5)
    np.testing.assert_allclose(r, case.reference_rotation_stress(), rtol=1e-10, atol=1e-5)
    np.testing.assert_allclose(p, case.reference_total_pressure(), rtol=1e-10, atol=1e-5)
    assert operators.A_uu.shape == (2 * nc, 2 * nc)
    assert operators.T.shape == (2 * nf, 4 * nc)
    assert operators.T_g.shape == (2 * nf, 2 * nf)
    assert operators.D_u.shape == (2 * nc, 2 * nf)
    traction = operators.T @ x + operators.T_g @ case.bound_vec
    expected = case.reference_stress()[:2, :2] @ case.grid.face_normals[:2]
    np.testing.assert_allclose(
        traction.reshape((2, nf), order="F"), expected, rtol=1e-10, atol=1e-5,
    )
    np.testing.assert_allclose(operators.D_u @ traction, 0.0, atol=1e-5)
    np.testing.assert_array_equal(case.x, saved_x)
    np.testing.assert_array_equal(case.bc_values, saved_boundary)
    assert all(point.trial is None and point.committed.alpha == 0 for point in case.material_points)


def test_auxiliary_equations_with_nonaffine_boundary_and_integrated_sources(
    prepared_case: PlaneStrainTpsa,
) -> None:
    case = prepared_case
    operators = TpsaOperators(case)
    nc = case.grid.num_cells
    bc_values = np.zeros_like(case.bc_values)
    bf = operators.boundary_faces
    x, y = case.grid.face_centers[:2, bf]
    bc_values[:, bf] = 1e-4 * np.array([x*x + x*y, y*y - x*y])
    rng = np.random.default_rng(3)
    u = rng.normal(size=2 * nc) * 1e-4
    sources = rng.normal(size=4 * nc)
    sources[:2 * nc] *= 1e6
    sources[2 * nc:] *= 1e-5
    rhs = operators.assemble_rhs(bc_values, sources=sources)
    saved_u, saved_rhs, saved_bc, saved_sources = (
        u.copy(), rhs.copy(), bc_values.copy(), sources.copy(),
    )
    r, p = operators.solve_auxiliary(u, rhs)
    candidate = np.concatenate((u, r, p))
    flux = case.face_discretization @ candidate + case.rhs_matrix @ bc_values.ravel(order="F")
    balance = case.div @ flux - case.accum @ candidate
    np.testing.assert_allclose(balance[2 * nc:], sources[2 * nc:], rtol=1e-10, atol=1e-14)
    assert np.count_nonzero(
        operators.A_pp.toarray() - np.diag(operators.A_pp.diagonal())
    ) > 0  # This case exercises the coupled pressure solve.
    np.testing.assert_array_equal(u, saved_u)
    np.testing.assert_array_equal(rhs, saved_rhs)
    np.testing.assert_array_equal(bc_values, saved_bc)
    np.testing.assert_array_equal(sources, saved_sources)
    assert case.x is None
    assert all(point.trial is None and point.committed.alpha == 0 for point in case.material_points)


def test_auxiliary_factorizations_are_reused_for_changed_loads(
    prepared_case: PlaneStrainTpsa, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = prepared_case
    operators = TpsaOperators(case)

    def forbid_refactorization(*args: object, **kwargs: object) -> None:
        raise AssertionError("The fixed auxiliary blocks must not be refactored.")

    monkeypatch.setattr(sps.linalg, "splu", forbid_refactorization)
    for factor in [1.0, 2.0, -0.5, 0.0, 1.0]:
        rhs = operators.assemble_rhs(factor * case.bc_values)
        u = factor * case.reference_displacement(case.grid.cell_centers).ravel(order="F")
        r, p = operators.solve_auxiliary(u, rhs)
        np.testing.assert_allclose(r, factor * case.reference_rotation_stress(), atol=1e-5)
        np.testing.assert_allclose(p, factor * case.reference_total_pressure(), atol=1e-5)


@pytest.mark.parametrize("unused_value", [np.nan, np.inf, -np.inf])
def test_load_rhs_ignores_interior_boundary_entries(
    prepared_case: PlaneStrainTpsa, unused_value: float,
) -> None:
    case = prepared_case
    operators = TpsaOperators(case)
    expected = operators.assemble_rhs(case.bc_values)
    bc_values = case.bc_values.copy()
    interior = np.setdiff1d(np.arange(case.grid.num_faces), operators.boundary_faces)
    bc_values[:, interior] = unused_value
    saved = bc_values.copy()
    bc_values.flags.writeable = False
    np.testing.assert_array_equal(operators.assemble_rhs(bc_values), expected)
    np.testing.assert_array_equal(bc_values, saved)


def test_fixed_operators_own_their_sparse_matrices(prepared_case: PlaneStrainTpsa) -> None:
    case = prepared_case
    operators = TpsaOperators(case)
    boundary = case.bc_values.copy()
    expected = operators.assemble_rhs(boundary)
    u = case.reference_displacement(case.grid.cell_centers).ravel(order="F")
    expected_r, expected_p = operators.solve_auxiliary(u, expected)
    for name in ("face_discretization", "rhs_matrix", "div", "accum"):
        original, fixed = getattr(case, name), getattr(operators, name)
        assert not np.shares_memory(original.data, fixed.data)
        original.data[:] = 0.0
    np.testing.assert_array_equal(operators.assemble_rhs(boundary), expected)
    r, p = operators.solve_auxiliary(u, expected)
    np.testing.assert_array_equal(r, expected_r)
    np.testing.assert_array_equal(p, expected_p)


def test_operators_require_preparation_and_full_dirichlet_boundaries() -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    with pytest.raises(RuntimeError, match="prepare_simulation"):
        TpsaOperators(case)
    case.prepare_simulation()
    face = case.grid.get_all_boundary_faces()[0]
    case.bc.is_dir[0, face] = False
    case.bc.is_neu[0, face] = True
    with pytest.raises(ValueError, match="Dirichlet"):
        TpsaOperators(case)


@pytest.mark.parametrize("row, column", [(2, 3), (3, 2)])
def test_rejects_coupled_rotation_pressure_blocks(
    prepared_case: PlaneStrainTpsa, row: int, column: int,
) -> None:
    case = prepared_case
    nc = case.grid.num_cells
    changed = case.accum.tolil()
    changed[row * nc, column * nc] = 1.0
    case.accum = changed.tocsr()
    with pytest.raises(ValueError, match="rotation-pressure coupling"):
        TpsaOperators(case)


def test_operators_reject_stale_or_nonfinite_matrices(prepared_case: PlaneStrainTpsa) -> None:
    case = prepared_case
    case.face_discretization.data[0] = np.nan
    with pytest.raises(ValueError, match="face_discretization"):
        TpsaOperators(case)
    case.prepare_simulation()
    case.accum = case.accum[:-1, :-1]
    with pytest.raises(ValueError, match="accum"):
        TpsaOperators(case)


def test_zero_auxiliary_accumulation_is_outside_current_scope(prepared_case: PlaneStrainTpsa) -> None:
    case = prepared_case
    diagonal = case.accum.diagonal()
    diagonal[3 * case.grid.num_cells:] = 0.0
    case.accum = sps.diags_array(diagonal).tocsr()
    with pytest.raises(ValueError, match="finite positive elastic coefficients"):
        TpsaOperators(case)


@pytest.mark.parametrize("field", ["bc_values", "sources", "u", "rhs"])
@pytest.mark.parametrize("defect", ["shape", "nonfinite"])
def test_auxiliary_input_validation(
    prepared_case: PlaneStrainTpsa, field: str, defect: str,
) -> None:
    case = prepared_case
    operators = TpsaOperators(case)
    nc = case.grid.num_cells
    data = {
        "bc_values": case.bc_values.copy(), "sources": np.zeros(4 * nc),
        "u": np.zeros(2 * nc), "rhs": operators.assemble_rhs(case.bc_values),
    }
    if defect == "shape":
        data[field] = data[field][:-1]
    else:
        data[field].flat[0] = np.nan
    with pytest.raises(ValueError, match=f"finite {field}|{field}.*finite"):
        if field in ("bc_values", "sources"):
            operators.assemble_rhs(data["bc_values"], sources=data["sources"])
        else:
            operators.solve_auxiliary(data["u"], data["rhs"])
