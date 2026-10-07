"""History isolation and return-map checks for material-point updates."""

import numpy as np
import pytest

from J2 import _materialProperty, vonMisesModel
from material_state import MaterialPoint, MaterialState
from scalar_hardening import (
    HardeningParameters,
    isotropic_hardening_K,
    kinematic_hardening_H,
)
from tensor import strain, stress


def make_point(theta=0.4, saturation=150.0):
    material = _materialProperty()
    material.young_modulus = 210_000.0
    material.poisson_ratio = 0.3
    parameters = HardeningParameters(250.0, 250.0 + saturation, 1200.0, theta, 20.0)
    return MaterialPoint(
        material=material,
        parameters=parameters,
        model=vonMisesModel("J2"),
        K_law=isotropic_hardening_K,
        H_law=kinematic_hardening_H,
    )


def assert_same_state(actual, expected):
    for name in ("stress", "plastic_strain", "backstress"):
        np.testing.assert_allclose(
            getattr(actual, name).to_numpy(), getattr(expected, name).to_numpy()
        )
    assert actual.alpha == pytest.approx(expected.alpha)


def preload(point):
    increment = strain([[0.004, 0.001, 0], [0.001, -0.002, 0], [0, 0, -0.002]])
    point.update(increment)
    point.commit()
    return increment


def test_state_defaults_and_copies_are_independent():
    state = MaterialState()
    for other in (MaterialState(), state.copy()):
        for name in ("stress", "plastic_strain", "backstress"):
            original = getattr(state, name).to_numpy(copy=False)
            copied = getattr(other, name).to_numpy(copy=False)
            np.testing.assert_array_equal(original, np.zeros((3, 3)))
            assert not np.shares_memory(original, copied)
    assert state.alpha == 0.0


def test_elastic_hydrostatic_predictor():
    point = make_point()
    committed = point.committed.copy()
    increment = 0.001 * strain.identity(3)
    trial, tangent = point.update(increment)

    expected = 3.0 * point.material.bulk_modulus * 0.001 * np.eye(3)
    np.testing.assert_allclose(trial.stress.to_numpy(), expected)
    assert tangent is None
    assert trial.alpha == 0.0
    assert_same_state(point.committed, committed)
    for name in ("stress", "plastic_strain", "backstress"):
        assert not np.shares_memory(
            getattr(trial, name).to_numpy(copy=False),
            getattr(point.committed, name).to_numpy(copy=False),
        )


def test_global_iterations_always_start_from_committed_history():
    point = make_point()
    preload(point)
    committed = point.committed.copy()
    increment = strain([[0.002, 0, 0.001], [0, -0.001, 0], [0.001, 0, -0.001]])

    first, first_tangent = point.update(increment)
    repeated, repeated_tangent = point.update(increment)
    assert_same_state(first, repeated)
    assert first_tangent is None and repeated_tangent is None
    assert first.alpha > committed.alpha

    # A revised global iterate must discard the previous plastic correction.
    revised, tangent = point.update(-0.1 * increment)
    reference = MaterialPoint(
        material=point.material,
        parameters=point.parameters,
        model=point.model,
        K_law=point.K_law,
        H_law=point.H_law,
        committed=committed,
    )
    expected, expected_tangent = reference.update(-0.1 * increment)
    assert_same_state(revised, expected)
    assert tangent is None and expected_tangent is None
    assert_same_state(point.committed, committed)


def test_commit_and_rollback():
    point = make_point()
    with pytest.raises(RuntimeError, match="No successful trial"):
        point.commit()

    trial, _ = point.update(strain(np.diag([0.004, -0.002, -0.002])))
    assert point.committed.alpha == 0.0
    assert trial.alpha > 0.0
    point.commit()
    assert_same_state(point.committed, trial)
    assert point.trial is None
    for name in ("stress", "plastic_strain", "backstress"):
        assert not np.shares_memory(
            getattr(trial, name).to_numpy(copy=False),
            getattr(point.committed, name).to_numpy(copy=False),
        )
    accepted = point.committed.copy()
    trial.alpha = -1.0
    assert_same_state(point.committed, accepted)

    point.update(strain(np.diag([0.008, -0.004, -0.004])))
    point.rollback()
    assert point.trial is None
    assert_same_state(point.committed, accepted)
    with pytest.raises(RuntimeError, match="No successful trial"):
        point.commit()


def test_initial_history_is_copied():
    original = make_point()
    preload(original)
    point = MaterialPoint(
        material=original.material,
        parameters=original.parameters,
        model=original.model,
        K_law=original.K_law,
        H_law=original.H_law,
        committed=original.committed,
    )
    saved = point.committed.copy()
    original.committed.alpha = -1.0
    original.committed.stress = stress.zeros((3, 3))
    assert_same_state(point.committed, saved)


def test_failed_update_cannot_commit_a_stale_trial(monkeypatch):
    point = make_point()
    preload(point)
    committed = point.committed.copy()
    increment = strain(np.diag([0.004, -0.002, -0.002]))
    point.update(increment)

    def fail(**kwargs):
        raise RuntimeError("Local update failed")

    monkeypatch.setattr(point.model, "radial_return_map", fail)
    with pytest.raises(RuntimeError, match="Local update failed"):
        point.update(increment)
    assert point.trial is None
    assert_same_state(point.committed, committed)
    with pytest.raises(RuntimeError, match="No successful trial"):
        point.commit()


@pytest.mark.parametrize("increment", [strain.zeros((2, 2)), strain([[0, 1, 0], [0, 0, 0], [0, 0, 0]])])
def test_invalid_strain_clears_trial(increment):
    point = make_point()
    point.update(strain.zeros((3, 3)))
    with pytest.raises(ValueError, match="symmetric 3D"):
        point.update(increment)
    assert point.trial is None
    assert_same_state(point.committed, MaterialState())


@pytest.mark.parametrize("theta", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("saturation", [0.0, 150.0])
@pytest.mark.parametrize("plastic", [False, True])
def test_elastic_and_plastic_updates_preserve_history(theta, saturation, plastic):
    point = make_point(theta, saturation)
    loading = preload(point)
    committed = point.committed.copy()
    increment = (
        strain([[0.002, 0, 0.001], [0, -0.001, 0.002], [0.001, 0.002, -0.001]])
        if plastic else -0.01 * loading
    )
    trial, tangent = point.update(increment)
    if plastic:
        assert trial.alpha > committed.alpha
    else:
        assert trial.alpha == committed.alpha

    assert tangent is None
    shifted_stress = trial.stress.deviatoric() - trial.backstress.deviatoric()
    yield_stress, _ = point.K_law(trial.alpha, point.parameters)
    residual = np.sqrt(shifted_stress.inner(shifted_stress)) - np.sqrt(2.0 / 3.0) * yield_stress
    if plastic:
        assert residual == pytest.approx(0.0, abs=1e-9)
    else:
        assert residual < 0.0
    assert_same_state(point.committed, committed)


@pytest.mark.parametrize("stress_scale", [1.0, 1e3, 1e6, 1e12])
def test_material_update_is_consistent_across_stress_units(stress_scale):
    reference = make_point()
    scaled = make_point()
    scaled.material.young_modulus *= stress_scale
    scaled.parameters.sigma_y *= stress_scale
    scaled.parameters.sigma_u *= stress_scale
    scaled.parameters.H_bar *= stress_scale

    increments = (
        strain([[0.004, 0.001, 0], [0.001, -0.002, 0], [0, 0, -0.002]]),
        strain([[0.002, 0, 0.001], [0, -0.001, 0.002], [0.001, 0.002, -0.001]]),
    )
    for increment in increments:
        expected, _ = reference.update(increment)
        actual, _ = scaled.update(increment)
        assert actual.alpha == pytest.approx(expected.alpha, rel=1e-10)
        np.testing.assert_allclose(
            actual.plastic_strain.to_numpy(), expected.plastic_strain.to_numpy(),
            rtol=1e-10, atol=1e-14,
        )
        for name in ("stress", "backstress"):
            np.testing.assert_allclose(
                getattr(actual, name).to_numpy() / stress_scale,
                getattr(expected, name).to_numpy(), rtol=1e-10, atol=1e-9,
            )
        reference.commit()
        scaled.commit()


def test_relative_tolerance_is_configurable_from_material_update():
    point = make_point()
    increment = strain(np.diag([0.004, -0.002, -0.002]))
    # A deliberately loose relative tolerance accepts the initial residual.
    loose, _ = point.update(increment, tol=0.0, rtol=1.0)
    strict, _ = point.update(increment, tol=0.0, rtol=1e-12)
    assert loose.alpha == 0.0
    assert strict.alpha > 0.0
    assert point.committed.alpha == 0.0
