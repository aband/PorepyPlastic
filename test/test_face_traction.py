"""Integrated face tractions for coupled TPSA candidates and plastic increments."""

from copy import deepcopy
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray
from porepy.grids.grid import Grid
from porepy.grids.structured import CartGrid

from coupling.plane_strain import PlaneStrainTpsa, _assemble_matrices, _discretize_get_matrices
from coupling.residual import (
    TpsaMaterialTrial, TpsaOperators, TpsaState, evaluate_face_traction,
    evaluate_material_trial,
)
from coupling.transfer import CellToFaceTransfer
from material_state import MaterialState

FloatArray: TypeAlias = NDArray[np.float64]
Setup: TypeAlias = tuple[PlaneStrainTpsa, TpsaOperators, CellToFaceTransfer, TpsaState]


@pytest.fixture
def setup() -> Setup:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.prepare_simulation()
    # Rectangular, rotated cells expose missing/doubled face areas and axis assumptions.
    angle = np.pi / 5
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    case.grid.nodes[:2] = rotation @ (np.array([2.0, 0.75])[:, None] * case.grid.nodes[:2])
    case.grid.compute_geometry()
    case.matrices = _discretize_get_matrices(case.grid, case.d)
    case.face_discretization, case.rhs_matrix, case.div, case.accum = _assemble_matrices(
        case.matrices, case.grid, case.d,
    )
    return case, TpsaOperators(case), CellToFaceTransfer(case.grid), TpsaState.zeros(case.grid)


def affine_candidate(case: PlaneStrainTpsa, gradient: FloatArray) -> tuple[FloatArray, FloatArray]:
    nc = case.grid.num_cells
    displacement = gradient @ case.grid.cell_centers[:2]
    r = np.full(nc, case.material.isotropic_shear_modulus * (gradient[0, 1] - gradient[1, 0]))
    p = np.full(nc, case.material.lame_parameter * np.trace(gradient))
    x = np.concatenate((displacement.ravel(order="F"), r, p))
    boundary = np.zeros_like(case.bc_values)
    bf = case.grid.get_all_boundary_faces()
    boundary[:, bf] = gradient @ case.grid.face_centers[:2, bf]
    return x, boundary


def zero_trial(nc: int) -> TpsaMaterialTrial:
    return TpsaMaterialTrial(
        np.zeros((3, 3, nc)), [MaterialState() for _ in range(nc)], np.zeros((3, 3, nc)),
    )


def test_elastic_affine_traction_matches_analytical_stress(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    gradient = np.array([[1e-4, 2e-4], [-3e-4, 4e-4]])
    x, boundary = affine_candidate(case, gradient)
    trial = evaluate_material_trial(operators, case.material_points, committed, x, boundary)
    traction = evaluate_face_traction(operators, transfer, committed, x, boundary, trial)
    strain = 0.5 * (gradient + gradient.T)
    stress = 2 * case.material.isotropic_shear_modulus * strain
    stress += case.material.lame_parameter * np.trace(strain) * np.eye(2)
    np.testing.assert_allclose(traction, stress @ case.grid.face_normals[:2], rtol=1e-10, atol=1e-5)
    np.testing.assert_allclose(operators.D_u @ traction.ravel(order="F"), 0.0, atol=1e-5)
    assert traction.shape == (2, case.grid.num_faces)
    assert case.x is None


@pytest.mark.parametrize("field, key", [(0, "stress"), (1, "stress_rotation"), (2, "stress_total_pressure")])
def test_unconverged_coupled_fields_are_used_without_auxiliary_solve(
    setup: Setup, field: int, key: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, operators, transfer, committed = setup
    nc = case.grid.num_cells
    x = np.zeros(4 * nc)
    block = [slice(0, 2 * nc), slice(2 * nc, 3 * nc), slice(3 * nc, 4 * nc)][field]
    size = x[block].size
    x[block] = np.arange(1, size + 1) * (1e-5 if field == 0 else 1e6)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("A coupled candidate must not be overwritten by an auxiliary solve.")

    monkeypatch.setattr(operators, "solve_auxiliary", forbidden)
    boundary = np.zeros_like(case.bc_values)
    # This trial need not satisfy the auxiliary equations yet.
    trial = evaluate_material_trial(operators, case.material_points, committed, x, boundary)
    expected_elastic = np.asarray(case.matrices[key] @ x[block]).reshape((2, -1), order="F")
    elastic_traction = evaluate_face_traction(
        operators, transfer, committed, x, boundary, zero_trial(nc),
    )
    np.testing.assert_allclose(elastic_traction, expected_elastic, rtol=1e-12, atol=1e-5)
    assert np.linalg.norm(elastic_traction) > 0.0
    np.testing.assert_array_equal(trial.stress_correction, 0.0)


def test_dirichlet_increment_uses_old_data_and_fixed_normal_signs(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    grid = case.grid
    bf = operators.boundary_faces
    rng = np.random.default_rng(12)
    committed.bc_values[:, bf] = rng.normal(size=(2, bf.size)) * 1e-4
    committed.traction[:] = rng.normal(size=committed.traction.shape) * 1e6
    boundary = committed.bc_values.copy()
    delta_g = rng.normal(size=(2, bf.size)) * 1e-5
    boundary[:, bf] += delta_g
    expected = committed.traction.copy()
    incidence = grid.cell_faces.tocsr()
    for index, face in enumerate(bf):
        start = incidence.indptr[face]
        cell, sign = incidence.indices[start], incidence.data[start]
        outward = sign * grid.face_normals[:2, face] / grid.face_areas[face]
        distance = np.dot(grid.face_centers[:2, face] - grid.cell_centers[:2, cell], outward)
        expected[:, face] += (
            sign * grid.face_areas[face] * 2 * case.material.isotropic_shear_modulus
            / distance * delta_g[:, index]
        )
    actual = evaluate_face_traction(
        operators, transfer, committed, committed.x, boundary, zero_trial(grid.num_cells),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-5)


@pytest.mark.parametrize("biased", [False, True], ids=["equal-weights", "custom-weights"])
def test_cell_correction_is_subtracted_once_with_area_weighted_normals(
    setup: Setup, biased: bool,
) -> None:
    case, operators, transfer, committed = setup
    if biased:
        def rule(grid: Grid, face: int, cells: NDArray[np.int64]) -> FloatArray:
            return np.array([0.2, 0.8])
        transfer = CellToFaceTransfer.from_rule(case.grid, rule)
    nc = case.grid.num_cells
    trial = zero_trial(nc)
    tensor = np.array([[2.0, 3.0, 100.0], [3.0, -4.0, 200.0], [100.0, 200.0, 500.0]]) * 1e6
    trial.stress_correction[:] = tensor[:, :, None] * np.arange(1, nc + 1)
    traction = evaluate_face_traction(
        operators, transfer, committed, committed.x, committed.bc_values, trial,
    )
    incidence = case.grid.cell_faces.tocsr()
    expected = np.zeros_like(traction)
    for face in range(case.grid.num_faces):
        cells = np.sort(incidence.indices[incidence.indptr[face]:incidence.indptr[face + 1]])
        weights = np.array([1.0]) if cells.size == 1 else np.array([0.2, 0.8]) if biased else np.array([0.5, 0.5])
        correction = sum(weight * trial.stress_correction[:2, :2, cell] for weight, cell in zip(weights, cells, strict=True))
        expected[:, face] = -correction @ case.grid.face_normals[:2, face]
    np.testing.assert_allclose(traction, expected, rtol=1e-12, atol=1e-8)
    # One stored face force has opposite contributions in its neighboring cells.
    interior = int(np.flatnonzero(np.diff(incidence.indptr) == 2)[0])
    force = np.zeros_like(traction)
    force[:, interior] = traction[:, interior]
    balance = np.asarray(operators.D_u @ force.ravel(order="F")).reshape((2, nc), order="F")
    assert np.linalg.norm(balance) > 0.0
    np.testing.assert_allclose(balance.sum(axis=1), 0.0, atol=1e-8)


@pytest.mark.parametrize("factor", [1.0, 0.99, 1.2], ids=["hold", "unload", "reload"])
def test_plastic_history_is_retained_across_load_steps(setup: Setup, factor: float) -> None:
    case, operators, transfer, virgin = setup
    gradient = np.array([[0.004, 0.001], [0.0, -0.001]])
    x_n, g_n = affine_candidate(case, gradient)
    first_trial = evaluate_material_trial(operators, case.material_points, virgin, x_n, g_n)
    t_n = evaluate_face_traction(operators, transfer, virgin, x_n, g_n, first_trial)
    assert first_trial.material_states[0].alpha > 0.0
    expected_first = first_trial.material_states[0].stress.to_numpy()[:2, :2] @ case.grid.face_normals[:2]
    np.testing.assert_allclose(t_n, expected_first, rtol=1e-10, atol=1e-5)
    committed = TpsaState(x_n, g_n, first_trial.epsilon, first_trial.material_states, t_n)
    x, boundary = affine_candidate(case, factor * gradient)
    trial = evaluate_material_trial(operators, case.material_points, committed, x, boundary)
    traction = evaluate_face_traction(operators, transfer, committed, x, boundary, trial)
    expected = trial.material_states[0].stress.to_numpy()[:2, :2] @ case.grid.face_normals[:2]
    np.testing.assert_allclose(traction, expected, rtol=1e-10, atol=1e-5)
    np.testing.assert_allclose(operators.D_u @ traction.ravel(order="F"), 0.0, atol=1e-5)
    if factor <= 1:
        np.testing.assert_allclose(trial.stress_correction, 0.0, atol=1e-5)
    else:
        assert trial.material_states[0].alpha > committed.material_states[0].alpha


def test_zero_increment_preserves_stored_numerical_traction(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    rng = np.random.default_rng(3)
    committed.traction[:] = rng.normal(size=committed.traction.shape)
    actual = evaluate_face_traction(
        operators, transfer, committed, committed.x, committed.bc_values,
        zero_trial(case.grid.num_cells),
    )
    np.testing.assert_array_equal(actual, committed.traction)
    assert not np.shares_memory(actual, committed.traction)


def test_repeated_evaluation_preserves_read_only_inputs_and_histories(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    x, boundary = affine_candidate(case, np.array([[0.004, 0.001], [0.0, 0.0]]))
    trial = evaluate_material_trial(operators, case.material_points, committed, x, boundary)
    saved_state, saved_trial = committed.copy(), deepcopy(trial)
    arrays = (x, boundary, committed.x, committed.bc_values, committed.traction, trial.stress_correction)
    snapshots = [array.copy() for array in arrays]
    for array in arrays:
        array.flags.writeable = False
    first = evaluate_face_traction(operators, transfer, committed, x, boundary, trial)
    expected = first.copy()
    evaluate_face_traction(operators, transfer, committed, -x, -boundary, trial)
    repeated = evaluate_face_traction(operators, transfer, committed, x, boundary, trial)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(repeated, expected)
    first[:] = 0.0
    np.testing.assert_array_equal(repeated, expected)
    for array, snapshot in zip(arrays, snapshots, strict=True):
        np.testing.assert_array_equal(array, snapshot)
    np.testing.assert_array_equal(committed.epsilon, saved_state.epsilon)
    np.testing.assert_array_equal(trial.epsilon, saved_trial.epsilon)
    for actual, saved in zip(
        committed.material_states + trial.material_states,
        saved_state.material_states + saved_trial.material_states, strict=True,
    ):
        assert actual.alpha == saved.alpha
        for name in ("stress", "plastic_strain", "backstress"):
            np.testing.assert_array_equal(getattr(actual, name).to_numpy(), getattr(saved, name).to_numpy())
    assert all(point.trial is None and point.committed.alpha == 0.0 for point in case.material_points)


def test_unused_old_and_new_boundary_entries_are_ignored(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    x, boundary = affine_candidate(case, np.array([[1e-4, 0.0], [0.0, 0.0]]))
    trial = evaluate_material_trial(operators, case.material_points, committed, x, boundary)
    expected = evaluate_face_traction(operators, transfer, committed, x, boundary, trial)
    interior = np.setdiff1d(np.arange(case.grid.num_faces), operators.boundary_faces)
    boundary[:, interior] = np.nan
    committed.bc_values[:, interior] = np.inf
    actual = evaluate_face_traction(operators, transfer, committed, x, boundary, trial)
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.isnan(boundary[:, interior]))
    assert np.all(np.isinf(committed.bc_values[:, interior]))


@pytest.mark.parametrize("field", ["x", "committed.x", "bc_values", "committed.bc_values", "committed.traction", "stress_correction"])
@pytest.mark.parametrize("defect", ["shape", "nonfinite"])
def test_invalid_used_arrays_are_rejected(setup: Setup, field: str, defect: str) -> None:
    case, operators, transfer, committed = setup
    trial = zero_trial(case.grid.num_cells)
    inputs = {
        "x": committed.x.copy(), "committed.x": committed.x,
        "bc_values": committed.bc_values.copy(), "committed.bc_values": committed.bc_values,
        "committed.traction": committed.traction, "stress_correction": trial.stress_correction,
    }
    inputs[field] = inputs[field][:-1].copy() if defect == "shape" else np.full_like(inputs[field], np.nan)
    committed.x, committed.bc_values = inputs["committed.x"], inputs["committed.bc_values"]
    committed.traction, trial.stress_correction = inputs["committed.traction"], inputs["stress_correction"]
    with pytest.raises(ValueError, match=field):
        evaluate_face_traction(operators, transfer, committed, inputs["x"], inputs["bc_values"], trial)


def test_transfer_with_incompatible_grid_size_is_rejected(setup: Setup) -> None:
    case, operators, _, committed = setup
    wrong_transfer = CellToFaceTransfer(CartGrid(np.array([1, 1])))
    with pytest.raises(ValueError, match="Transfer.*counts"):
        evaluate_face_traction(
            operators, wrong_transfer, committed, committed.x, committed.bc_values,
            zero_trial(case.grid.num_cells),
        )
