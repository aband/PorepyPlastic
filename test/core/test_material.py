"""Nested retained-face material evaluation with fixed committed history."""

from copy import deepcopy
from dataclasses import replace
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from alternatives.plane_strain import PlaneStrainTpsa
from utils.state import TpsaState
from material.material_state import MaterialState
from core.operators import RetainedFaceOperators
from core.residual import RetainedFaceResidual, evaluate_global_residual
from material.tensor import strain

Setup: TypeAlias = tuple[PlaneStrainTpsa, RetainedFaceOperators, TpsaState]
FloatArray: TypeAlias = NDArray[np.float64]


@pytest.fixture
def setup() -> Setup:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.prepare_simulation()
    ops = RetainedFaceOperators(case.grid, case.bc, mu=case.material.isotropic_shear_modulus,
                                lam=case.material.lame_parameter)
    return case, ops, TpsaState.zeros(case.grid)


def affine_increment(
    ops: RetainedFaceOperators, old: TpsaState, gradient: FloatArray,
    translation: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    assert ops.mu is not None and ops.lam is not None
    shift = np.zeros((2, 1)) if translation is None else translation[:, None]
    u = gradient @ ops.grid.cell_centers[:2] + shift
    d = gradient @ ops.grid.face_centers[:2] + shift
    cell = np.r_[u.ravel("F"), ops.mu * (gradient[0, 1] - gradient[1, 0]), ops.lam * np.trace(gradient)]
    return ops.pack(cell, d[:, ops.free_faces]), old.bc_values + d


def assert_history_equal(actual: MaterialState, expected: MaterialState) -> None:
    assert actual.alpha == expected.alpha
    for name in ("stress", "plastic_strain", "backstress"):
        np.testing.assert_array_equal(getattr(actual, name).to_numpy(), getattr(expected, name).to_numpy())


def assert_state_unchanged(actual: TpsaState, expected: TpsaState) -> None:
    for name in ("x", "bc_values", "epsilon", "traction"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for state, snapshot in zip(actual.material_states, expected.material_states, strict=True):
        assert_history_equal(state, snapshot)


def assert_equilibrated(result: RetainedFaceResidual, ops: RetainedFaceOperators) -> None:
    np.testing.assert_allclose(result.residual[ops.u_slice], 0, atol=1e-5)
    np.testing.assert_allclose(result.residual[ops.r_slice], 0, atol=1e-14)
    np.testing.assert_allclose(result.residual[ops.p_slice], 0, atol=1e-14)
    np.testing.assert_allclose(result.residual[ops.face_slice], 0, atol=1e-5)


def preloaded(setup: Setup) -> tuple[TpsaState, FloatArray]:
    case, ops, virgin = setup
    gradient = np.array([[0.004, 0.001], [0.0, -0.001]])
    z, boundary = affine_increment(ops, virgin, gradient)
    result = evaluate_global_residual(ops, case.material_points, virgin, z, boundary)
    assert_equilibrated(result, ops)
    material = result.material_trial
    # Explicitly construct a committed homogeneous state for loading-path tests.
    old = TpsaState(
        virgin.x + z[ops.cell_slice], boundary, material.epsilon, material.material_states,
        material.material_states[0].stress.to_numpy()[:2, :2] @ ops.grid.face_normals[:2],
    )
    return old, gradient


@pytest.mark.parametrize("field", ["zero", "translation", "rotation", "affine"])
def test_elastic_candidate_matches_hookes_law(setup: Setup, field: str) -> None:
    case, ops, old = setup
    gradient = {
        "zero": np.zeros((2, 2)), "translation": np.zeros((2, 2)),
        "rotation": np.array([[0, -0.003], [0.003, 0]]),
        "affine": np.array([[1e-4, 2e-4], [-3e-4, 4e-4]]),
    }[field]
    shift = np.array([0.001, -0.002]) if field == "translation" else np.zeros(2)
    z, boundary = affine_increment(ops, old, gradient, shift)
    result = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    material = result.material_trial
    epsilon = np.zeros((3, 3))
    epsilon[:2, :2] = 0.5 * (gradient + gradient.T)
    stress = (2 * case.material.isotropic_shear_modulus * epsilon
              + case.material.lame_parameter * np.trace(epsilon) * np.eye(3))
    np.testing.assert_allclose(material.epsilon, np.repeat(epsilon[:, :, None], ops.num_cells, axis=2), atol=1e-14)
    np.testing.assert_allclose(material.stress_correction, 0, atol=1e-5)
    for history in material.material_states:
        np.testing.assert_allclose(history.stress.to_numpy(), stress, rtol=1e-12, atol=1e-5)
        assert history.alpha == 0
    assert_equilibrated(result, ops)
    assert case.x is None


def test_plastic_candidate_matches_closed_form_j2_and_preserves_plane_strain(setup: Setup) -> None:
    case, ops, old = setup
    gradient = np.array([[0.004, 0.001], [0.0, -0.001]])
    z, boundary = affine_increment(ops, old, gradient)
    result = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    epsilon = np.zeros((3, 3))
    epsilon[:2, :2] = 0.5 * (gradient + gradient.T)
    mu, lam, parameters = case.material.isotropic_shear_modulus, case.material.lame_parameter, case.hardening_parameters
    predictor = 2 * mu * epsilon + lam * np.trace(epsilon) * np.eye(3)
    deviator = predictor - np.trace(predictor) / 3 * np.eye(3)
    norm, c = float(np.linalg.norm(deviator)), np.sqrt(2 / 3)
    # Independent closed-form oracle for virgin J2 with linear mixed hardening.
    gamma = (norm - c * parameters.sigma_y) / (2 * mu + 2 / 3 * parameters.H_bar)
    plastic = gamma * deviator / norm
    correction = 2 * mu * plastic
    expected_backstress = 2 / 3 * (1 - parameters.theta) * parameters.H_bar * plastic
    material = result.material_trial
    for cell, history in enumerate(material.material_states):
        np.testing.assert_allclose(history.stress.to_numpy(), predictor - correction, rtol=1e-12, atol=1e-5)
        np.testing.assert_allclose(history.plastic_strain.to_numpy(), plastic, atol=1e-14)
        np.testing.assert_allclose(history.backstress.to_numpy(), expected_backstress, rtol=1e-12, atol=1e-5)
        assert history.alpha == pytest.approx(c * gamma, rel=1e-12)
        np.testing.assert_allclose(material.stress_correction[:, :, cell], correction, rtol=1e-12, atol=1e-5)
        assert abs(history.stress.to_numpy()[2, 2]) > 1e6
        assert abs(history.plastic_strain.to_numpy()[2, 2]) > 1e-5
    np.testing.assert_array_equal(material.epsilon[2], 0)
    np.testing.assert_array_equal(material.epsilon[:, 2], 0)
    assert_equilibrated(result, ops)


def test_single_cell_plastic_update_uses_prescribed_faces_only() -> None:
    case = PlaneStrainTpsa(cells_per_axis=1)
    case.prepare_simulation()
    ops = RetainedFaceOperators(case.grid, case.bc, mu=case.material.isotropic_shear_modulus,
                                lam=case.material.lame_parameter)
    old = TpsaState.zeros(case.grid)
    z, boundary = affine_increment(ops, old, np.diag([0.004, 0.]))
    result = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    assert result.residual.shape == (4,) and ops.num_free_faces == 0
    assert result.material_trial.material_states[0].alpha > 0
    np.testing.assert_allclose(result.material_trial.epsilon[0, 0], 0.004)
    assert_equilibrated(result, ops)


@pytest.mark.parametrize("factor", [1.0, 0.99, 1.2], ids=["hold", "unload", "reload"])
def test_loading_path_starts_from_committed_strain_and_history(setup: Setup, factor: float) -> None:
    case, ops, _ = setup
    old, gradient = preloaded(setup)
    snapshot = old.copy()
    z, boundary = affine_increment(ops, old, (factor - 1) * gradient)
    result = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    material = result.material_trial
    np.testing.assert_allclose(material.epsilon, factor * old.epsilon, rtol=1e-12, atol=1e-14)
    for cell, history in enumerate(material.material_states):
        previous = old.material_states[cell]
        plastic_increment = history.plastic_strain.to_numpy() - previous.plastic_strain.to_numpy()
        q = 2 * case.material.isotropic_shear_modulus * plastic_increment
        q += case.material.lame_parameter * np.trace(plastic_increment) * np.eye(3)
        np.testing.assert_allclose(material.stress_correction[:, :, cell], q, rtol=1e-10, atol=1e-5)
        if factor <= 1:
            assert history.alpha == previous.alpha
            np.testing.assert_allclose(q, 0, atol=1e-5)
            increment = (factor - 1) * old.epsilon[:, :, cell]
            elastic = (2 * case.material.isotropic_shear_modulus * increment
                       + case.material.lame_parameter * np.trace(increment) * np.eye(3))
            np.testing.assert_allclose(history.stress.to_numpy(), previous.stress.to_numpy() + elastic, atol=1e-5)
        else:
            assert history.alpha > previous.alpha
    assert_equilibrated(result, ops)
    assert_state_unchanged(old, snapshot)
    assert all(point.committed.alpha == 0 and point.trial is None for point in case.material_points)


def test_mixed_cells_have_independent_return_maps_and_face_imbalances(setup: Setup) -> None:
    case, ops, old = setup
    for point, yield_stress in zip(case.material_points, [100e6, 250e6, 800e6, 1000e6], strict=True):
        point.parameters = replace(point.parameters, sigma_y=yield_stress, sigma_u=yield_stress)
    z, boundary = affine_increment(ops, old, np.diag([0.004, 0.0]))
    result = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    material = result.material_trial
    assert [history.alpha > 0 for history in material.material_states] == [True, True, False, False]
    np.testing.assert_allclose(material.epsilon[0, 0], 0.004, atol=1e-14)
    np.testing.assert_allclose(material.stress_correction[:, :, 2:], 0, atol=1e-5)
    assert np.linalg.norm(result.residual[ops.face_slice]) > 1e6
    np.testing.assert_allclose(result.residual[ops.u_slice], 0, atol=1e-5)
    np.testing.assert_allclose(result.residual[ops.r_slice], 0, atol=1e-14)
    np.testing.assert_allclose(result.residual[ops.p_slice], 0, atol=1e-14)


def test_material_response_depends_only_on_face_increments(setup: Setup) -> None:
    case, ops, old = setup
    z, boundary = affine_increment(ops, old, np.diag([0.004, 0.0]))
    first = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    z[ops.cell_slice] += np.arange(4 * ops.num_cells) + 1
    changed = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    np.testing.assert_array_equal(changed.material_trial.epsilon, first.material_trial.epsilon)
    np.testing.assert_array_equal(changed.material_trial.stress_correction, first.material_trial.stress_correction)
    for a, b in zip(changed.material_trial.material_states, first.material_trial.material_states, strict=True):
        assert_history_equal(a, b)
    assert not np.allclose(changed.residual, first.residual)


def test_rejected_trials_preserve_read_only_inputs_and_existing_point_slots(setup: Setup) -> None:
    case, ops, _ = setup
    old, gradient = preloaded(setup)
    snapshot = old.copy()
    for point in case.material_points:
        point.update(strain(np.diag([0.002, 0., 0.])))
        point.commit()  # Deliberately different from the explicit global history.
        point.update(strain(np.diag([0.001, 0., 0.])))
    saved_points = [(p.committed.copy(), deepcopy(p.trial)) for p in case.material_points]
    z, boundary = affine_increment(ops, old, -0.01 * gradient)
    original_z, original_bc = z.copy(), boundary.copy()
    for array in (z, boundary, old.x, old.bc_values, old.epsilon, old.traction):
        array.flags.writeable = False
    first = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    reference = deepcopy(first)
    rejected_z, rejected_bc = affine_increment(ops, old, 0.8 * gradient)
    rejected = evaluate_global_residual(ops, case.material_points, old, rejected_z, rejected_bc)
    assert rejected.material_trial.material_states[0].alpha > old.material_states[0].alpha
    repeated = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    first.residual[:] = first.traction_increment[:] = -1
    first.material_trial.epsilon[:] = first.material_trial.stress_correction[:] = -1
    first.material_trial.material_states[0].alpha = -1
    np.testing.assert_array_equal(repeated.residual, reference.residual)
    np.testing.assert_array_equal(repeated.traction_increment, reference.traction_increment)
    np.testing.assert_array_equal(repeated.material_trial.epsilon, reference.material_trial.epsilon)
    np.testing.assert_array_equal(repeated.material_trial.stress_correction, reference.material_trial.stress_correction)
    for a, b in zip(repeated.material_trial.material_states, reference.material_trial.material_states, strict=True):
        assert_history_equal(a, b)
    np.testing.assert_array_equal(z, original_z)
    np.testing.assert_array_equal(boundary, original_bc)
    assert_state_unchanged(old, snapshot)
    for point, (committed, trial) in zip(case.material_points, saved_points, strict=True):
        assert_history_equal(point.committed, committed)
        assert point.trial is not None and trial is not None
        assert_history_equal(point.trial, trial)


def test_heterogeneous_elastic_moduli_use_each_cells_material(setup: Setup) -> None:
    case, _, old = setup
    for index, point in enumerate(case.material_points):
        point.material = deepcopy(point.material)
        point.material.young_modulus *= 0.5 + 0.2 * index
    mu = np.array([p.material.isotropic_shear_modulus for p in case.material_points])
    lam = np.array([p.material.lame_parameter for p in case.material_points])
    ops = RetainedFaceOperators(case.grid, case.bc, mu=mu, lam=lam)
    z, boundary = affine_increment(ops, old, np.diag([1e-4, 0.]))
    result = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    for cell, history in enumerate(result.material_trial.material_states):
        expected = np.diag([(2 * mu[cell] + lam[cell]) * 1e-4, lam[cell] * 1e-4, lam[cell] * 1e-4])
        np.testing.assert_allclose(history.stress.to_numpy(), expected, rtol=1e-12, atol=1e-5)
    np.testing.assert_allclose(result.material_trial.stress_correction, 0, atol=1e-5)


def test_load_and_committed_residual_are_forwarded_without_changing_material_response(setup: Setup) -> None:
    case, ops, old = setup
    z, boundary = affine_increment(ops, old, np.diag([0.004, 0.]))
    first = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    body = np.arange(2 * ops.num_cells, dtype=float).reshape((2, ops.num_cells)) * 1e6
    offset = np.arange(ops.num_dofs, dtype=float) * 0.001
    expected = first.residual + offset
    expected[ops.u_slice] -= body.ravel("F")
    result = evaluate_global_residual(ops, case.material_points, old, z, boundary,
                                      body_force_increment=body, committed_residual=offset)
    np.testing.assert_allclose(result.residual, expected, atol=1e-8)
    np.testing.assert_array_equal(result.material_trial.stress_correction, first.material_trial.stress_correction)
    np.testing.assert_array_equal(result.traction_increment, first.traction_increment)


def test_unused_boundary_entries_are_ignored(setup: Setup) -> None:
    case, ops, old = setup
    z, boundary = affine_increment(ops, old, np.diag([0.004, 0.]))
    expected = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    boundary[:, ops.free_faces], old.bc_values[:, ops.free_faces] = np.nan, np.inf
    result = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    np.testing.assert_array_equal(result.residual, expected.residual)
    np.testing.assert_array_equal(result.material_trial.epsilon, expected.material_trial.epsilon)


def test_local_failure_propagates_without_partial_history_updates(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    case, ops, old = setup
    snapshot = old.copy()
    z, boundary = affine_increment(ops, old, np.diag([0.004, 0.]))

    def fail(**kwargs: object) -> None:
        raise RuntimeError("Injected second-cell local failure")

    monkeypatch.setattr(case.material_points[1].model, "radial_return_map", fail)
    with pytest.raises(RuntimeError, match="second-cell local failure"):
        evaluate_global_residual(ops, case.material_points, old, z, boundary)
    assert_state_unchanged(old, snapshot)
    for point in case.material_points:
        assert point.trial is None
        assert_history_equal(point.committed, MaterialState())


@pytest.mark.parametrize("defect", ["point_count", "history_count", "mu", "lam", "missing_lam", "old_strain"])
def test_rejects_incompatible_material_setup(setup: Setup, defect: str) -> None:
    case, ops, old = setup
    points = list(case.material_points)
    if defect == "point_count":
        points.pop()
    elif defect == "history_count":
        old.material_states.pop()
    elif defect in ("mu", "lam"):
        mu = np.full(ops.num_cells, case.material.isotropic_shear_modulus)
        lam = np.full(ops.num_cells, case.material.lame_parameter)
        (mu if defect == "mu" else lam)[-1] *= 1.1
        ops = RetainedFaceOperators(case.grid, case.bc, mu=mu, lam=lam)
    elif defect == "missing_lam":
        ops = RetainedFaceOperators(case.grid, case.bc, mu=case.material.isotropic_shear_modulus)
    else:
        old.epsilon[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate_global_residual(ops, points, old, np.zeros(ops.num_dofs), np.zeros((2, ops.num_faces)))
