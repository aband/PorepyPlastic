"""Dimension-aware local tangents, full 3D stress and fixed material history."""

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint, MaterialState
from material.scalar_hardening import (
    HardeningParameters,
    isotropic_hardening_K,
    kinematic_hardening_H,
)
from material.tangent import analytical_material_tangent, finite_difference_material_tangent
from material.tensor import strain


def _point() -> MaterialPoint:
    material = _materialProperty()
    material.young_modulus = 210_000.0
    material.poisson_ratio = 0.3
    return MaterialPoint(
        material, HardeningParameters(250.0, 400.0, 1200.0, 0.4, 20.0),
        vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H,
    )


def _update(
    point: MaterialPoint, old: MaterialState, increment: NDArray[np.float64],
) -> MaterialState:
    local = MaterialPoint(
        point.material, point.parameters, point.model, point.K_law, point.H_law,
        committed=old,
    )
    returned, _ = local.update(strain(increment))
    return returned


def _increment() -> NDArray[np.float64]:
    # Nonzero zz, xz and yz entries exercise every new 3D derivative direction.
    return np.array([
        [0.003, 0.0006, 0.0003],
        [0.0006, -0.0009, -0.0002],
        [0.0003, -0.0002, -0.0001],
    ])


def _assert_state_equal(actual: MaterialState, expected: MaterialState) -> None:
    assert actual.alpha == expected.alpha
    for name in ("stress", "plastic_strain", "backstress"):
        np.testing.assert_array_equal(getattr(actual, name).to_numpy(), getattr(expected, name).to_numpy())


@pytest.mark.parametrize("tangent", [analytical_material_tangent, finite_difference_material_tangent])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("unloading", [False, True])
def test_elastic_tangent_retains_all_stress_components(
    tangent: Callable[..., NDArray[np.float64]], dim: int, unloading: bool,
) -> None:
    point = _point()
    old = MaterialState()
    increment = 0.001 * _increment()
    if unloading:
        old = _update(point, old, _increment())
        assert old.alpha > 0
        increment = -0.01 * _increment()
    assert _update(point, old, increment).alpha == old.alpha

    actual = tangent(point, old, increment, dim=dim)
    expected = point.material.elastic_tensor.to_numpy()[:, :, :dim, :dim]
    assert actual.shape == (3, 3, dim, dim)
    if tangent is analytical_material_tangent:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=5e-7, atol=2e-3)
    # Plane strain retains sigma_zz coupling; tensor shear has no factor of two.
    assert actual[2, 2, 0, 0] == pytest.approx(point.material.lame_parameter, rel=5e-7)
    assert actual[0, 1, 0, 1] == pytest.approx(point.material.isotropic_shear_modulus, rel=5e-7)


@pytest.mark.parametrize("tangent", [analytical_material_tangent, finite_difference_material_tangent])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("preloaded", [False, True])
def test_plastic_tangent_matches_independent_return_map_derivatives(
    tangent: Callable[..., NDArray[np.float64]], dim: int, preloaded: bool,
) -> None:
    point = _point()
    old = MaterialState()
    if preloaded:
        old = _update(point, old, np.diag([0.004, -0.001, -0.0005]))
        assert old.alpha > 0
    increment = _increment()
    assert _update(point, old, increment).alpha > old.alpha
    actual = tangent(point, old, increment, dim=dim)
    expected = np.empty((3, 3, dim, dim))
    h = 1e-8
    for k in range(dim):
        for l in range(dim):
            direction = np.zeros((3, 3))
            direction[k, l] += 0.5
            direction[l, k] += 0.5
            plus = _update(point, old, increment + h * direction)
            minus = _update(point, old, increment - h * direction)
            assert plus.alpha > old.alpha and minus.alpha > old.alpha
            expected[:, :, k, l] = (plus.stress.to_numpy() - minus.stress.to_numpy()) / (2 * h)
    if tangent is analytical_material_tangent:
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-4)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=3e-3)


@pytest.mark.parametrize("tangent", [analytical_material_tangent, finite_difference_material_tangent])
@pytest.mark.parametrize("plastic", [False, True])
def test_default_and_explicit_2d_match_restricted_3d_tangent(
    tangent: Callable[..., NDArray[np.float64]], plastic: bool,
) -> None:
    point = _point()
    old = MaterialState()
    # Preserve support for a full 3D base state even when varying only x/y.
    increment = _increment() * (1.0 if plastic else 0.001)
    default = tangent(point, old, increment)
    explicit = tangent(point, old, increment, dim=2)
    full = tangent(point, old, increment, dim=3)
    assert default.shape == (3, 3, 2, 2)
    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_array_equal(explicit, full[:, :, :2, :2])


@pytest.mark.parametrize("dim", [2, 3])
def test_analytical_tangent_reuses_returned_state_without_update(
    dim: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = _point()
    old = _update(point, MaterialState(), np.diag([0.004, -0.001, -0.0005]))
    increment = _increment()
    returned = _update(point, old, increment)
    saved = returned.copy()
    expected = analytical_material_tangent(point, old, increment, dim=dim)

    def unexpected_update(self: MaterialPoint, increment: strain) -> tuple[MaterialState, None]:
        raise AssertionError("The existing returned state must be reused.")

    monkeypatch.setattr(MaterialPoint, "update", unexpected_update)
    actual = analytical_material_tangent(point, old, increment, dim=dim, returned=returned)
    np.testing.assert_array_equal(actual, expected)
    _assert_state_equal(returned, saved)


@pytest.mark.parametrize("tangent", [analytical_material_tangent, finite_difference_material_tangent])
@pytest.mark.parametrize("dim", [2, 3])
def test_tangents_preserve_supplied_history_and_pending_trial(
    tangent: Callable[..., NDArray[np.float64]], dim: int,
) -> None:
    point = _point()
    point.update(strain(_increment()))
    point.commit()
    pending, _ = point.update(strain(0.2 * _increment()))
    # External history deliberately differs from point.committed and point.trial.
    old = _update(point, MaterialState(), np.diag([0.004, -0.001, -0.0005]))
    old_before, committed_before, trial_before = old.copy(), point.committed.copy(), pending.copy()
    committed_slot = point.committed
    increment = _increment()
    increment_before = increment.copy()
    increment.setflags(write=False)
    first = tangent(point, old, increment, dim=dim)
    np.testing.assert_array_equal(first, tangent(point, old, increment, dim=dim))
    assert point.committed is committed_slot and point.trial is pending
    _assert_state_equal(old, old_before)
    _assert_state_equal(point.committed, committed_before)
    _assert_state_equal(pending, trial_before)
    np.testing.assert_array_equal(increment, increment_before)


@pytest.mark.parametrize("tangent", [analytical_material_tangent, finite_difference_material_tangent])
@pytest.mark.parametrize("dim", [0, 1, 4, True, 2.0, "3", None])
def test_invalid_dimension_is_rejected(
    tangent: Callable[..., NDArray[np.float64]], dim: object,
) -> None:
    with pytest.raises(ValueError, match="dimension must be 2 or 3"):
        tangent(_point(), MaterialState(), np.zeros((3, 3)), dim=dim)


@pytest.mark.parametrize("tangent", [analytical_material_tangent, finite_difference_material_tangent])
@pytest.mark.parametrize("dim", [2, 3])
def test_numpy_integer_dimension_is_accepted(
    tangent: Callable[..., NDArray[np.float64]], dim: int,
) -> None:
    actual = tangent(_point(), MaterialState(), np.zeros((3, 3)), dim=cast(int, np.int64(dim)))
    assert actual.shape == (3, 3, dim, dim)
