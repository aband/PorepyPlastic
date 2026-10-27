"""Shared 2D/3D material trials, analytical responses, and history isolation."""

from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import porepy as pp
import pytest

from core.layout import DofLayout, FloatArray
from core.state import MaterialHistory
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint, MaterialState
from material.scalar_hardening import (
    HardeningParameters, isotropic_hardening_K, kinematic_hardening_H,
)
from material.tensor import strain, stress
from utils.material_evaluation import evaluate_material_at_strain
from utils.state import TpsaState


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def layout(request: pytest.FixtureRequest) -> DofLayout:
    grid = pp.CartGrid(np.array([2] + [1] * (request.param - 1)))
    grid.compute_geometry()
    bc = pp.BoundaryConditionVectorial(grid, grid.get_all_boundary_faces(), "dir")
    return DofLayout.from_grid(grid, bc)


@pytest.fixture
def points(layout: DofLayout) -> list[MaterialPoint]:
    result = []
    for cell in range(layout.num_cells):
        material = _materialProperty()
        material.young_modulus = 210_000.0 + 20_000.0 * cell
        material.poisson_ratio = 0.3 - 0.05 * cell
        result.append(MaterialPoint(
            material, HardeningParameters(250.0, 250.0, 1200.0, 0.4, 20.0),
            vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H,
        ))
    return result


@pytest.fixture
def committed(layout: DofLayout) -> MaterialHistory:
    return MaterialHistory.zeros(layout)


def _epsilon(layout: DofLayout) -> FloatArray:
    value = np.array([[0.004, 0.0006, 0.0], [0.0006, -0.001, 0.0], [0.0, 0.0, 0.0]])
    if layout.dim == 3:
        value[2, 2] = -0.0015
        value[0, 2] = value[2, 0] = -0.0004
        value[1, 2] = value[2, 1] = 0.0007
    return np.asarray(value[:, :, None] * (1.0 + 0.2 * np.arange(layout.num_cells)), dtype=np.float64)


def _hooke(point: MaterialPoint, increment: FloatArray) -> FloatArray:
    return np.asarray(
        2 * point.material.isotropic_shear_modulus * increment
        + point.material.lame_parameter * np.trace(increment) * np.eye(3),
        dtype=np.float64,
    )


def _reference(point: MaterialPoint, old: MaterialState, increment: FloatArray) -> MaterialState:
    """Closed-form J2 with linear mixed hardening; no production return-map calls."""
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
        2 * point.material.isotropic_shear_modulus + 2.0 / 3.0 * parameters.H_bar
    )
    plastic_increment = np.zeros((3, 3)) if gamma == 0 else gamma * shifted / norm
    return MaterialState(
        stress(predictor - _hooke(point, plastic_increment)),
        strain(old.plastic_strain.to_numpy() + plastic_increment),
        stress(backstress + 2.0 / 3.0 * (1 - parameters.theta) * parameters.H_bar * plastic_increment),
        old.alpha + c * gamma,
    )


def _assert_history(actual: MaterialState, expected: MaterialState, *, exact: bool = False) -> None:
    if exact:
        assert actual.alpha == expected.alpha
    else:
        assert actual.alpha == pytest.approx(expected.alpha, rel=1e-11, abs=1e-14)
    for name in ("stress", "plastic_strain", "backstress"):
        a, b = getattr(actual, name).to_numpy(), getattr(expected, name).to_numpy()
        if exact:
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)


def test_elastic_response_with_heterogeneous_moduli(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory,
) -> None:
    epsilon = 0.01 * _epsilon(layout)
    trial = evaluate_material_at_strain(
        layout=layout, material_points=points, committed=committed,
        epsilon=epsilon, dim=layout.dim,
    )
    np.testing.assert_array_equal(trial.epsilon, epsilon)
    np.testing.assert_allclose(trial.stress_correction, 0.0, atol=1e-12)
    for cell, point in enumerate(points):
        _assert_history(trial.material_states[cell], MaterialState(stress=stress(_hooke(point, epsilon[:, :, cell]))))
    if layout.dim == 2:
        assert trial.material_states[0].stress.to_numpy()[2, 2] != 0


@pytest.mark.parametrize("theta", [0.0, 0.4, 1.0], ids=["kinematic", "mixed", "isotropic"])
def test_plastic_response_matches_closed_form_j2(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory, theta: float,
) -> None:
    for point in points:
        point.parameters = replace(point.parameters, theta=theta)
    epsilon = _epsilon(layout)
    trial = evaluate_material_at_strain(layout, points, committed, epsilon, dim=layout.dim)
    for cell, point in enumerate(points):
        expected = _reference(point, committed.material_states[cell], epsilon[:, :, cell])
        actual = trial.material_states[cell]
        _assert_history(actual, expected)
        assert actual.alpha > 0
        q = _hooke(point, expected.plastic_strain.to_numpy())
        np.testing.assert_allclose(trial.stress_correction[:, :, cell], q, rtol=1e-11, atol=1e-12)
        shifted = actual.stress.to_numpy() - actual.backstress.to_numpy()
        shifted -= np.trace(shifted) / 3 * np.eye(3)
        yield_stress = point.parameters.sigma_y + theta * point.parameters.H_bar * actual.alpha
        assert np.sqrt(1.5) * np.linalg.norm(shifted) == pytest.approx(yield_stress, rel=1e-11)
        if layout.dim == 2:
            assert actual.stress.to_numpy()[2, 2] != 0
            assert actual.plastic_strain.to_numpy()[2, 2] != 0
    if layout.dim == 2:
        np.testing.assert_array_equal(trial.epsilon[2], 0.0)
        np.testing.assert_array_equal(trial.epsilon[:, 2], 0.0)


@pytest.mark.parametrize("factor", [1.0, 0.99, 1.3], ids=["hold", "unload", "reload"])
def test_total_strain_is_differenced_against_committed_history(
    layout: DofLayout, points: list[MaterialPoint], factor: float,
) -> None:
    previous = _epsilon(layout)
    old = MaterialHistory(layout.dim, previous, [
        _reference(point, MaterialState(), previous[:, :, cell])
        for cell, point in enumerate(points)
    ])
    trial = evaluate_material_at_strain(layout, points, old, factor * previous, dim=layout.dim)
    for cell, point in enumerate(points):
        increment = (factor - 1) * previous[:, :, cell]
        expected = _reference(point, old.material_states[cell], increment)
        _assert_history(trial.material_states[cell], expected)
        plastic_increment = expected.plastic_strain.to_numpy() - old.material_states[cell].plastic_strain.to_numpy()
        np.testing.assert_allclose(trial.stress_correction[:, :, cell], _hooke(point, plastic_increment), atol=1e-12)
        if factor <= 1:
            assert trial.material_states[cell].alpha == old.material_states[cell].alpha
        else:
            assert trial.material_states[cell].alpha > old.material_states[cell].alpha


def test_cells_can_be_elastic_and_plastic_in_the_same_evaluation(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory,
) -> None:
    points[1].parameters = replace(points[1].parameters, sigma_y=1e8, sigma_u=1e8)
    trial = evaluate_material_at_strain(layout, points, committed, _epsilon(layout), dim=layout.dim)
    assert trial.material_states[0].alpha > 0
    assert trial.material_states[1].alpha == 0
    assert np.linalg.norm(trial.stress_correction[:, :, 0]) > 0
    np.testing.assert_allclose(trial.stress_correction[:, :, 1], 0.0, atol=1e-12)


@pytest.mark.parametrize("component", [(2, 2), (0, 2), (1, 2)], ids=["zz", "xz", "yz"])
@pytest.mark.parametrize("field", ["candidate", "committed"])
def test_out_of_plane_strain_is_allowed_only_in_3d(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory,
    component: tuple[int, int], field: str,
) -> None:
    epsilon = np.zeros_like(committed.epsilon)
    target = epsilon if field == "candidate" else committed.epsilon
    i, j = component
    target[i, j] = target[j, i] = 1e-5
    if layout.dim == 2:
        with pytest.raises(ValueError, match="plane strain"):
            evaluate_material_at_strain(layout, points, committed, epsilon, dim=layout.dim)
    else:
        trial = evaluate_material_at_strain(layout, points, committed, epsilon, dim=3)
        for cell, point in enumerate(points):
            expected = _hooke(point, epsilon[:, :, cell] - committed.epsilon[:, :, cell])
            np.testing.assert_allclose(trial.material_states[cell].stress.to_numpy(), expected, atol=1e-12)


def test_repeated_trials_own_outputs_and_preserve_readonly_inputs_and_point_slots(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory,
) -> None:
    # Point slots deliberately disagree with the explicitly supplied global state.
    for point in points:
        point.update(strain(np.diag([0.003, 0.0, 0.0])))
        point.commit()
        point.update(strain(np.diag([0.001, 0.0, 0.0])))
    slots = [(point.committed, point.trial) for point in points]
    snapshots = [(point.committed.copy(), point.trial.copy()) for point in points if point.trial is not None]
    saved = committed.copy()
    storage = np.zeros((3, 3, 2 * layout.num_cells))
    storage[:, :, ::2] = _epsilon(layout)
    epsilon = storage[:, :, ::2]
    epsilon.setflags(write=False)
    committed.epsilon.setflags(write=False)
    for old in committed.material_states:
        for name in ("stress", "plastic_strain", "backstress"):
            getattr(old, name).to_numpy(copy=False).setflags(write=False)
    original = storage.copy()
    first = evaluate_material_at_strain(layout, points, committed, epsilon, dim=layout.dim)
    evaluate_material_at_strain(layout, points, committed, 2 * epsilon, dim=layout.dim)
    repeated = evaluate_material_at_strain(layout, points, committed, epsilon, dim=layout.dim)
    for name in ("epsilon", "stress_correction"):
        a, b = getattr(first, name), getattr(repeated, name)
        np.testing.assert_array_equal(a, b)
        assert a.flags.owndata and a.dtype == np.float64
        for source in (epsilon, committed.epsilon, b):
            assert not np.shares_memory(a, source)
        a.fill(-999.0)
    for cell, (actual, other) in enumerate(zip(first.material_states, repeated.material_states, strict=True)):
        _assert_history(actual, other, exact=True)
        for name in ("stress", "plastic_strain", "backstress"):
            data = getattr(actual, name).to_numpy(copy=False)
            for source in (other, committed.material_states[cell], points[cell].committed):
                assert not np.shares_memory(data, getattr(source, name).to_numpy(copy=False))
            # Tensor views are intentionally read-only; replace the owned field.
            tensor_type = type(getattr(actual, name))
            setattr(actual, name, tensor_type(np.full((3, 3), -999.0)))
        actual.alpha = -1
    np.testing.assert_array_equal(storage, original)
    np.testing.assert_array_equal(committed.epsilon, saved.epsilon)
    for actual, expected in zip(committed.material_states, saved.material_states, strict=True):
        _assert_history(actual, expected, exact=True)
    for point, (old_slot, trial_slot), (old, trial) in zip(points, slots, snapshots, strict=True):
        assert point.committed is old_slot and point.trial is trial_slot
        assert point.trial is not None
        _assert_history(point.committed, old, exact=True)
        _assert_history(point.trial, trial, exact=True)


def test_later_cell_failure_does_not_accept_partial_history(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = committed.copy()
    original_update = MaterialPoint.update
    calls = 0

    def fail_second(
        self: MaterialPoint, increment: strain, tol: float = 1e-10, *, rtol: float = 1e-12,
    ) -> tuple[MaterialState, None]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Second cell failed")
        return original_update(self, increment, tol=tol, rtol=rtol)

    monkeypatch.setattr(MaterialPoint, "update", fail_second)
    with pytest.raises(RuntimeError, match="Second cell failed"):
        evaluate_material_at_strain(layout, points, committed, _epsilon(layout), dim=layout.dim)
    assert calls == 2
    np.testing.assert_array_equal(committed.epsilon, saved.epsilon)
    for old, snapshot, point in zip(committed.material_states, saved.material_states, points, strict=True):
        _assert_history(old, snapshot, exact=True)
        _assert_history(point.committed, MaterialState(), exact=True)
        assert point.trial is None


@dataclass(frozen=True)
class _CellCount:
    num_cells: int


def test_legacy_state_and_default_dimension_remain_supported() -> None:
    grid = pp.CartGrid(np.array([1, 1]))
    old = TpsaState.zeros(grid)
    material = _materialProperty()
    material.young_modulus, material.poisson_ratio = 210_000.0, 0.3
    points = [MaterialPoint(material, HardeningParameters(250.0, 250.0, 1200.0, 0.4, 0.0),
                            vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H)]
    epsilon = np.diag([0.004, -0.001, 0.0])[:, :, None]
    implicit = evaluate_material_at_strain(_CellCount(1), points, old, epsilon)
    new = MaterialHistory(2, old.epsilon, old.material_states)
    explicit = evaluate_material_at_strain(_CellCount(1), points, new, epsilon, dim=2)
    np.testing.assert_array_equal(implicit.epsilon, explicit.epsilon)
    np.testing.assert_array_equal(implicit.stress_correction, explicit.stress_correction)
    _assert_history(implicit.material_states[0], explicit.material_states[0], exact=True)


@pytest.mark.parametrize("field", ["candidate", "committed"])
@pytest.mark.parametrize("invalid", ["shape", "nan", "inf", "negative_inf", "complex", "string", "object", "bool", "asymmetric"])
def test_invalid_strain_is_rejected(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory,
    field: str, invalid: str,
) -> None:
    epsilon = _epsilon(layout)
    value = (epsilon if field == "candidate" else committed.epsilon).copy()
    if invalid == "shape":
        value = value[:, :, :-1]
    elif invalid in ("nan", "inf", "negative_inf"):
        value[0, 0, -1] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[invalid]
    elif invalid == "asymmetric":
        value[0, 1, -1] += 0.01
    else:
        value = value.astype({"complex": "complex128", "string": "U32", "object": "O", "bool": "bool"}[invalid])
    if field == "candidate":
        epsilon = value
    else:
        committed.epsilon = value
    with pytest.raises(ValueError, match="epsilon"):
        evaluate_material_at_strain(layout, points, committed, epsilon, dim=layout.dim)


@pytest.mark.parametrize("invalid_dim", [0, 1, 4])
def test_invalid_dimension_is_rejected(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory, invalid_dim: int,
) -> None:
    with pytest.raises(ValueError, match="dimension"):
        evaluate_material_at_strain(layout, points, committed, _epsilon(layout), dim=invalid_dim)


def test_history_dimension_must_match_explicit_dimension(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory,
) -> None:
    with pytest.raises(ValueError, match="different dimension"):
        evaluate_material_at_strain(layout, points, committed, _epsilon(layout), dim=5 - layout.dim)


@pytest.mark.parametrize("invalid", ["no_cells", "points_short", "points_long", "histories_short", "histories_long"])
def test_cell_counts_must_match(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory, invalid: str,
) -> None:
    if invalid == "no_cells":
        layout = replace(layout, num_cells=0)
        points, committed.material_states = [], []
    elif invalid == "points_short":
        points.pop()
    elif invalid == "points_long":
        points.append(points[-1])
    elif invalid == "histories_short":
        committed.material_states.pop()
    else:
        committed.material_states.append(MaterialState())
    with pytest.raises(ValueError, match="one material point and history per cell"):
        evaluate_material_at_strain(layout, points, committed, committed.epsilon, dim=layout.dim)


@pytest.mark.parametrize("dtype", ["float32", "int64", "uint64"])
def test_real_numeric_inputs_return_float64_owned_arrays(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory, dtype: str,
) -> None:
    epsilon = np.zeros_like(committed.epsilon, dtype=dtype)
    epsilon[0, 0] = 1
    trial = evaluate_material_at_strain(layout, points, committed, cast(FloatArray, epsilon), dim=layout.dim)
    for value in (trial.epsilon, trial.stress_correction):
        assert value.dtype == np.float64 and value.flags.owndata
        assert not np.shares_memory(value, epsilon)
    for cell, point in enumerate(points):
        expected = _reference(point, MaterialState(), epsilon[:, :, cell].astype(np.float64))
        _assert_history(trial.material_states[cell], expected)


def test_local_tolerances_are_forwarded_without_computing_tangents(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_update = MaterialPoint.update
    calls: list[tuple[float, float]] = []

    def record(
        self: MaterialPoint, increment: strain, tol: float = 1e-10, *, rtol: float = 1e-12,
    ) -> tuple[MaterialState, None]:
        calls.append((tol, rtol))
        return original_update(self, increment, tol=tol, rtol=rtol)

    def unexpected_tangent(*args: object, **kwargs: object) -> None:
        pytest.fail("Residual material trials must not compute tangents.")

    monkeypatch.setattr(MaterialPoint, "update", record)
    monkeypatch.setattr(vonMisesModel, "consistent_tangent", unexpected_tangent)
    evaluate_material_at_strain(layout, points, committed, _epsilon(layout), dim=layout.dim, tol=1e-8, rtol=1e-9)
    assert calls == [(1e-8, 1e-9)] * layout.num_cells


@pytest.mark.parametrize("name", ["tol", "rtol"])
@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf])
def test_invalid_local_tolerances_are_rejected(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory, name: str, invalid: float,
) -> None:
    tolerances = {"tol": 1e-10, "rtol": 1e-12, name: invalid}
    with pytest.raises(ValueError, match="tolerances"):
        evaluate_material_at_strain(layout, points, committed, _epsilon(layout), dim=layout.dim, **tolerances)


def test_overflow_in_increment_is_rejected_before_return_map(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epsilon = np.zeros_like(committed.epsilon)
    epsilon[0, 0] = 1e308
    committed.epsilon[0, 0] = -1e308

    def unexpected_update(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid strain increments must be rejected before return mapping.")

    monkeypatch.setattr(MaterialPoint, "update", unexpected_update)
    with np.errstate(over="raise"), pytest.raises(ValueError, match="increments must remain finite"):
        evaluate_material_at_strain(layout, points, committed, epsilon, dim=layout.dim)


def test_nonfinite_correction_is_rejected(
    layout: DofLayout, points: list[MaterialPoint], committed: MaterialHistory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_return(*args: object, **kwargs: object) -> tuple[MaterialState, None]:
        return MaterialState(stress=stress(np.full((3, 3), np.nan))), None

    monkeypatch.setattr(MaterialPoint, "update", bad_return)
    with pytest.raises(ValueError, match="Stress corrections must remain finite"):
        evaluate_material_at_strain(layout, points, committed, _epsilon(layout), dim=layout.dim)
    assert all(point.trial is None for point in points)
