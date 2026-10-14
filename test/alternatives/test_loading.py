"""Accepted load sequences, constitutive history, restart, and failure isolation."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, TypeAlias

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray

from alternatives.original.jacobian import AnalyticalJacobian, FiniteDifferenceJacobian
from utils.load_schedule import AdaptiveLoadOptions, AdaptiveLoadStepError, LoadStepError
from alternatives.loading import LoadStepController
from alternatives.newton import JacobianCallback
from utils.newton_utils import JacobianMatrix, NewtonConvergenceError
from alternatives.plane_strain import PlaneStrainTpsa
from alternatives.original.residual import TpsaOperators, TpsaResidual
from utils.state import TpsaState
from utils.transfer import CellToFaceTransfer
from material.material_state import MaterialPoint, MaterialState
from material.tensor import strain

Setup: TypeAlias = tuple[PlaneStrainTpsa, TpsaOperators, CellToFaceTransfer, TpsaState]


@pytest.fixture
def setup() -> Setup:
    case = PlaneStrainTpsa(cells_per_axis=3)
    case.prepare_simulation()
    return case, TpsaOperators(case), CellToFaceTransfer(case.grid), TpsaState.zeros(case.grid)


def assert_state_equal(actual: TpsaState, expected: TpsaState) -> None:
    for name in ("x", "epsilon", "bc_values", "traction"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for a, b in zip(actual.material_states, expected.material_states, strict=True):
        assert a.alpha == b.alpha
        for name in ("stress", "plastic_strain", "backstress"):
            np.testing.assert_array_equal(getattr(a, name).to_numpy(), getattr(b, name).to_numpy())


@pytest.mark.parametrize("numerical_jacobian", [False, True])
def test_elastic_sequence_scales_absolute_boundary_and_integrated_sources(
    setup: Setup, numerical_jacobian: bool,
) -> None:
    case, ops, transfer, old = setup
    sources = np.zeros(4 * ops.num_cells)
    sources[:2 * ops.num_cells] = (np.array([[2e5], [-1e5]]) * case.grid.cell_volumes).ravel(order="F")
    sources[2 * ops.num_cells:] = 1e-6
    reference = np.asarray(sps.linalg.spsolve(ops.A, ops.assemble_rhs(case.bc_values, sources=sources)))
    controller = LoadStepController(
        ops, transfer, case.material_points, old, case.bc_values,
        reference_sources=sources,
        jacobian_factory=None if numerical_jacobian else lambda state: lambda x, evaluation: ops.A,
        rtol=(0.0, 0.0, 0.0),
    )
    factors = np.array([0.25, 0.75, 1.0, 0.5, -0.25, 0.0])
    records = controller.run(factors)
    assert len(records) == len(controller.steps) == len(factors)
    for factor, record in zip(factors, records, strict=True):
        assert record.load_factor == factor
        np.testing.assert_allclose(record.state.x, factor * reference, rtol=1e-8, atol=1e-4)
        np.testing.assert_allclose(record.state.x[:2 * ops.num_cells], factor * reference[:2 * ops.num_cells], atol=1e-13)
        np.testing.assert_array_equal(record.state.bc_values, factor * case.bc_values)
        np.testing.assert_array_equal(record.sources, factor * sources)
        assert np.all(record.newton.residual_norms[-1] <= record.newton.thresholds)
        assert all(history.alpha == 0.0 for history in record.state.material_states)
    assert controller.load_factor == 0.0
    assert_state_equal(controller.state, records[-1].state)
    assert_state_equal(old, TpsaState.zeros(case.grid))
    assert case.x is None


def test_plastic_loading_and_elastic_unloading_preserve_full_history() -> None:
    case = PlaneStrainTpsa(cells_per_axis=3, displacement_gradient=np.diag([0.004, 0.0]))
    case.prepare_simulation()
    ops, transfer = TpsaOperators(case), CellToFaceTransfer(case.grid)
    old = TpsaState.zeros(case.grid)
    for point in case.material_points:
        point.update(strain(np.diag([1e-5, 0.0, 0.0])))
    points_before = deepcopy(case.material_points)
    controller = LoadStepController(ops, transfer, case.material_points, old, case.bc_values)
    records = controller.run(np.r_[np.linspace(0.05, 1.0, 20), 0.99, 0.98])
    peak, unloaded = records[-3].state, records[-1].state
    alphas = np.array([[history.alpha for history in record.state.material_states] for record in records])
    assert np.all(alphas[0] == 0.0)
    assert np.all(alphas[-3] > 0.0)
    assert np.all(np.diff(alphas, axis=0) >= -1e-14)
    expected_u = case.reference_displacement(case.grid.cell_centers).ravel(order="F")
    for record in records:
        np.testing.assert_allclose(record.state.x[:2 * ops.num_cells], record.load_factor * expected_u, atol=1e-10)
        assert np.all(record.newton.residual_norms[-1] <= record.newton.thresholds)
    for cell, (a, b) in enumerate(zip(peak.material_states, unloaded.material_states, strict=True)):
        assert a.alpha == pytest.approx(b.alpha, abs=1e-13)
        np.testing.assert_allclose(a.plastic_strain.to_numpy(), b.plastic_strain.to_numpy(), atol=1e-13)
        np.testing.assert_allclose(a.backstress.to_numpy(), b.backstress.to_numpy(), atol=1e-5)
        # Full 3D stress follows the elastic unloading increment, including sigma_zz.
        depsilon = strain(unloaded.epsilon[:, :, cell] - peak.epsilon[:, :, cell])
        expected_stress = a.stress.to_numpy() + case.material.elastic_tensor.double_contract(depsilon).to_numpy()
        np.testing.assert_allclose(b.stress.to_numpy(), expected_stress, rtol=1e-10, atol=1e-3)
    assert_state_equal(old, TpsaState.zeros(case.grid))
    for point, saved in zip(case.material_points, points_before, strict=True):
        assert point.committed.alpha == saved.committed.alpha == 0.0
        assert point.trial is not None and saved.trial is not None
        np.testing.assert_array_equal(point.trial.stress.to_numpy(), saved.trial.stress.to_numpy())


def test_factory_receives_fresh_previous_state_and_controller_can_restart(setup: Setup) -> None:
    case, ops, transfer, old = setup
    seen: list[TpsaState] = []

    def factory(state: TpsaState) -> JacobianCallback:
        seen.append(state.copy())
        return FiniteDifferenceJacobian(ops, transfer, case.material_points, state)

    controller = LoadStepController(
        ops, transfer, case.material_points, old, case.bc_values, jacobian_factory=factory,
        rtol=(0.0, 0.0, 0.0),
    )
    records = controller.run([0.25, 0.5, 0.5])
    assert len(seen) == 3
    assert_state_equal(seen[0], old)
    for previous, factory_state in zip(records[:-1], seen[1:], strict=True):
        assert_state_equal(factory_state, previous.state)
    assert records[-1].newton.iterations == 0
    restart = LoadStepController(
        ops, transfer, case.material_points, controller.state, case.bc_values,
        initial_load_factor=controller.load_factor, jacobian_factory=factory,
        rtol=(0.0, 0.0, 0.0),
    )
    continued = controller.advance(0.75)
    resumed = restart.advance(0.75)
    assert_state_equal(resumed.state, continued.state)
    assert len(restart.steps) == 1 and len(controller.steps) == 4


def test_returned_records_and_state_are_independent_snapshots(setup: Setup) -> None:
    case, ops, transfer, old = setup
    controller = LoadStepController(ops, transfer, case.material_points, old, case.bc_values)
    first = controller.advance(0.25)
    saved = controller.state
    first.state.x[:] = 0.0
    first.state.material_states[0].alpha = 5.0
    first.newton.residual_norms[:] = np.inf
    snapshot = controller.state
    snapshot.traction[:] = 42.0
    history = controller.steps
    history[0].state.epsilon[:] = 3.0
    history[0].sources[:] = 1.0
    assert_state_equal(controller.state, saved)
    assert_state_equal(controller.steps[0].state, saved)
    assert np.all(np.isfinite(controller.steps[0].newton.residual_norms))
    assert np.all(controller.steps[0].sources == 0.0)
    controller.advance(0.5)
    assert_state_equal(controller.steps[0].state, saved)


def test_reference_loads_are_copied_and_interior_boundary_entries_ignored(setup: Setup) -> None:
    case, ops, transfer, old = setup
    boundary = case.bc_values.copy()
    interior = np.setdiff1d(np.arange(ops.num_faces), ops.boundary_faces)
    boundary[:, interior] = np.nan
    sources = np.zeros(4 * ops.num_cells)
    controller = LoadStepController(ops, transfer, case.material_points, old, boundary, reference_sources=sources)
    boundary[:] = np.nan
    sources[:] = np.inf
    old.epsilon[:] = 7.0
    result = controller.advance(1.0)
    np.testing.assert_array_equal(result.state.bc_values, case.bc_values)
    np.testing.assert_array_equal(result.sources, 0.0)


@pytest.mark.parametrize("failure", ["iterations", "linear", "factory", "material"])
def test_failed_step_preserves_accepted_history_and_allows_retry(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    case, ops, transfer, old = setup
    fail = False

    def factory(state: TpsaState) -> JacobianCallback:
        if fail and failure == "factory":
            # Even a factory mutating its supplied copy cannot damage accepted history.
            state.epsilon[:] = 99.0
            state.material_states[0].alpha = 100.0
            raise RuntimeError("Injected factory failure")
        matrix = ops.A
        if fail and failure == "iterations":
            matrix = 2 * matrix  # One correction cannot converge with this matrix.
        if fail and failure == "linear":
            matrix = sps.csr_array(matrix.shape)
        return lambda x, evaluation: matrix

    controller = LoadStepController(
        ops, transfer, case.material_points, old, case.bc_values,
        jacobian_factory=factory, max_iterations=1,
    )
    accepted = controller.advance(0.25)
    fail = True
    with monkeypatch.context() as patch:
        if failure == "material":
            def broken_update(self: MaterialPoint, increment: strain) -> tuple[MaterialState, None]:
                raise RuntimeError("Injected material failure")
            patch.setattr(MaterialPoint, "update", broken_update)
        with pytest.raises(LoadStepError) as caught:
            controller.run([0.5, 0.75])
    error = caught.value
    assert error.step_index == 2 and error.load_factor == 0.5
    assert isinstance(error.__cause__, NewtonConvergenceError if failure == "iterations" else RuntimeError)
    assert len(controller.steps) == 1 and controller.load_factor == 0.25
    assert_state_equal(controller.state, accepted.state)
    assert_state_equal(controller.steps[0].state, accepted.state)
    assert_state_equal(old, TpsaState.zeros(case.grid))
    assert all(point.trial is None for point in case.material_points)
    fail = False
    retried = controller.advance(0.5)
    assert controller.load_factor == 0.5 and len(controller.steps) == 2
    assert_state_equal(controller.state, retried.state)


def test_failure_of_first_step_keeps_initial_state(setup: Setup) -> None:
    case, ops, transfer, old = setup
    controller = LoadStepController(ops, transfer, case.material_points, old, case.bc_values, max_iterations=0)
    with pytest.raises(LoadStepError) as caught:
        controller.advance(1.0)
    assert caught.value.step_index == 1
    assert isinstance(caught.value.__cause__, NewtonConvergenceError)
    assert_state_equal(controller.state, old)
    assert controller.steps == () and controller.load_factor == 0.0


@pytest.mark.parametrize("schedule", [[0.25, np.nan], [np.inf], [[0.25, 0.5]]])
def test_invalid_schedule_is_rejected_before_any_step(setup: Setup, schedule: object) -> None:
    case, ops, transfer, old = setup
    controller = LoadStepController(ops, transfer, case.material_points, old, case.bc_values)
    values = np.array(schedule, dtype=np.float64)
    with pytest.raises(ValueError, match="one-dimensional"):
        controller.run(values)
    assert controller.steps == ()
    assert_state_equal(controller.state, old)


def test_empty_schedule_is_no_op(setup: Setup) -> None:
    case, ops, transfer, old = setup
    controller = LoadStepController(ops, transfer, case.material_points, old, case.bc_values)
    assert controller.run([]) == ()
    assert controller.steps == ()
    assert_state_equal(controller.state, old)


def test_initial_load_factor_must_match_committed_boundary_data(setup: Setup) -> None:
    case, ops, transfer, old = setup
    with pytest.raises(ValueError, match="Committed boundary data must match"):
        LoadStepController(ops, transfer, case.material_points, old, case.bc_values, initial_load_factor=0.5)


@pytest.fixture
def plastic_setup() -> tuple[PlaneStrainTpsa, TpsaOperators, CellToFaceTransfer, LoadStepController]:
    case = PlaneStrainTpsa(cells_per_axis=3, displacement_gradient=np.diag([0.004, 0.0]))
    case.prepare_simulation()
    ops, transfer = TpsaOperators(case), CellToFaceTransfer(case.grid)
    # Keep unrelated trial slots populated: the controller must leave them intact.
    for point in case.material_points:
        point.update(strain(np.diag([1e-5, 0.0, 0.0])))
    controller = LoadStepController(
        ops, transfer, case.material_points, TpsaState.zeros(case.grid), case.bc_values,
    )
    controller.run(np.linspace(0.05, 1.0, 20))
    return case, ops, transfer, controller


def test_proportional_plastic_loading_matches_closed_form_j2(
    plastic_setup: tuple[PlaneStrainTpsa, TpsaOperators, CellToFaceTransfer, LoadStepController],
) -> None:
    case, _, _, controller = plastic_setup
    state = controller.state
    material, parameters = case.material, case.hardening_parameters
    assert parameters.sigma_u == parameters.sigma_y
    # For proportional loading and linear mixed hardening, the accumulated return
    # is available in closed form. This oracle does not call the return map.
    epsilon = np.diag([0.004, 0.0, 0.0])
    mu = material.isotropic_shear_modulus
    predictor = 2 * mu * epsilon + material.lame_parameter * np.trace(epsilon) * np.eye(3)
    deviator = predictor - np.trace(predictor) / 3 * np.eye(3)
    norm = float(np.linalg.norm(deviator))
    c = np.sqrt(2 / 3)
    gamma = (norm - c * parameters.sigma_y) / (2 * mu + 2 / 3 * parameters.H_bar)
    assert gamma > 0.0
    plastic_strain = gamma * deviator / norm
    sigma = predictor - 2 * mu * plastic_strain
    backstress = 2 / 3 * (1 - parameters.theta) * parameters.H_bar * plastic_strain
    np.testing.assert_allclose(state.epsilon, np.repeat(epsilon[:, :, None], case.grid.num_cells, axis=2), atol=1e-11)
    np.testing.assert_allclose(state.traction, sigma[:2, :2] @ case.grid.face_normals[:2], rtol=1e-8, atol=1.0)
    for history in state.material_states:
        assert history.alpha == pytest.approx(c * gamma, rel=1e-7, abs=1e-11)
        np.testing.assert_allclose(history.stress.to_numpy(), sigma, rtol=1e-8, atol=2.0)
        np.testing.assert_allclose(history.plastic_strain.to_numpy(), plastic_strain, rtol=1e-7, atol=1e-11)
        np.testing.assert_allclose(history.backstress.to_numpy(), backstress, rtol=1e-7, atol=0.01)


def test_restart_from_plastic_state_matches_uninterrupted_unload_and_reload(
    plastic_setup: tuple[PlaneStrainTpsa, TpsaOperators, CellToFaceTransfer, LoadStepController],
) -> None:
    case, ops, transfer, original = plastic_setup
    saved = original.state
    previous_records = original.steps
    assert all(history.alpha > 0 for history in saved.material_states)
    restarted = LoadStepController(
        ops, transfer, case.material_points, saved, case.bc_values,
        initial_load_factor=original.load_factor,
    )
    schedule = [0.99, 0.98, 1.0, 1.01]
    continuous, resumed = original.run(schedule), restarted.run(schedule)
    assert len(resumed) == len(restarted.steps) == len(schedule)
    assert len(original.steps) == len(previous_records) + len(schedule)
    for a, b in zip(continuous, resumed, strict=True):
        assert a.load_factor == b.load_factor
        assert a.newton.iterations == b.newton.iterations
        assert_state_equal(a.state, b.state)
        np.testing.assert_array_equal(a.newton.residual_norms, b.newton.residual_norms)
    for before, after in zip(saved.material_states, resumed[-1].state.material_states, strict=True):
        assert after.alpha > before.alpha  # Reloading past the previous peak adds plastic strain.
    assert_state_equal(saved, previous_records[-1].state)
    for previous_record, current_record in zip(previous_records, original.steps[:len(previous_records)], strict=True):
        assert_state_equal(previous_record.state, current_record.state)


def test_partial_material_failure_after_plastic_loading_preserves_state_and_retry(
    plastic_setup: tuple[PlaneStrainTpsa, TpsaOperators, CellToFaceTransfer, LoadStepController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, ops, transfer, controller = plastic_setup
    saved, records, points = controller.state, controller.steps, deepcopy(case.material_points)
    update = MaterialPoint.update
    completed: list[tuple[float, float]] = []
    injected_error = RuntimeError("Material failure after three updated cells")

    def fail_on_fourth_cell(self: MaterialPoint, increment: strain) -> tuple[MaterialState, None]:
        if len(completed) == 3:
            raise injected_error
        result = update(self, increment)
        completed.append((self.committed.alpha, result[0].alpha))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(MaterialPoint, "update", fail_on_fourth_cell)
        with pytest.raises(LoadStepError) as caught:
            controller.run([1.01, 1.02])
    assert caught.value.__cause__ is injected_error
    assert caught.value.step_index == len(records) + 1 and caught.value.load_factor == 1.01
    assert len(completed) == 3 and any(new > old for old, new in completed)
    assert controller.load_factor == 1.0 and len(controller.steps) == len(records)
    assert_state_equal(controller.state, saved)
    for a, b in zip(controller.steps, records, strict=True):
        assert_state_equal(a.state, b.state)
        np.testing.assert_array_equal(a.newton.residual_norms, b.newton.residual_norms)
    for point, before in zip(case.material_points, points, strict=True):
        for slot in ("committed", "trial"):
            actual, expected = getattr(point, slot), getattr(before, slot)
            assert actual is not None and expected is not None
            assert actual.alpha == expected.alpha
            for field in ("stress", "plastic_strain", "backstress"):
                np.testing.assert_array_equal(getattr(actual, field).to_numpy(), getattr(expected, field).to_numpy())
    clean = LoadStepController(
        ops, transfer, case.material_points, saved, case.bc_values, initial_load_factor=1.0,
    )
    retried, fresh = controller.advance(1.01), clean.advance(1.01)
    assert_state_equal(retried.state, fresh.state)
    np.testing.assert_array_equal(retried.newton.residual_norms, fresh.newton.residual_norms)


def test_run_retains_successful_prefix_and_stops_at_failed_target(setup: Setup) -> None:
    case, ops, transfer, old = setup
    calls = 0

    def factory(state: TpsaState) -> JacobianCallback:
        nonlocal calls
        calls += 1
        matrix = sps.csr_array(ops.A.shape) if calls == 3 else ops.A
        return lambda x, evaluation: matrix

    controller = LoadStepController(ops, transfer, case.material_points, old, case.bc_values, jacobian_factory=factory)
    with pytest.raises(LoadStepError) as caught:
        controller.run([0.25, 0.5, 0.75, 1.0])
    assert calls == 3  # The final target is never attempted, and no step is retried.
    assert caught.value.step_index == 3 and caught.value.load_factor == 0.75
    assert [record.load_factor for record in controller.steps] == [0.25, 0.5]
    expected = np.asarray(sps.linalg.spsolve(ops.A, ops.assemble_rhs(0.5 * case.bc_values)))
    np.testing.assert_allclose(controller.state.x, expected, rtol=1e-9, atol=1e-6)
    assert controller.load_factor == 0.5
    assert_state_equal(controller.state, controller.steps[-1].state)
    assert_state_equal(old, TpsaState.zeros(case.grid))


@pytest.mark.parametrize("factor", [np.nan, np.inf, -np.inf])
def test_nonfinite_single_target_preserves_existing_accepted_step(setup: Setup, factor: float) -> None:
    case, ops, transfer, old = setup
    controller = LoadStepController(ops, transfer, case.material_points, old, case.bc_values)
    accepted = controller.advance(0.25)
    with pytest.raises(ValueError, match="load_factor must be finite"):
        controller.advance(factor)
    assert controller.load_factor == 0.25 and len(controller.steps) == 1
    assert_state_equal(controller.state, accepted.state)


def test_adaptive_cutbacks_retry_from_last_accepted_state(setup: Setup) -> None:
    case, ops, transfer, initial = setup
    seen: list[TpsaState] = []
    amplitude = float(np.max(np.abs(case.bc_values)))

    def factory(old: TpsaState) -> JacobianCallback:
        seen.append(old.copy())

        # Large increments use a deliberately poor tangent in a real Newton solve.
        def callback(x: NDArray[np.float64], evaluation: TpsaResidual) -> JacobianMatrix:
            change = float(np.max(np.abs(evaluation.trial.bc_values - old.bc_values))) / amplitude
            return (2 * ops.A) if change > 0.25 else ops.A

        return callback

    controller = LoadStepController(
        ops, transfer, case.material_points, initial, case.bc_values,
        jacobian_factory=factory, max_iterations=1, atol=(1e-4, 1e-11, 1e-11),
    )
    records = controller.run_adaptive([1.0, 0.7], options=AdaptiveLoadOptions(
        initial_increment=0.6, max_increment=0.6, min_increment=0.05,
        growth_factor=2, target_iterations=1,
    ))
    attempts = controller.attempts
    np.testing.assert_allclose([a.load_factor for a in attempts[:3]], [0.6, 0.3, 0.15])
    assert [a.accepted for a in attempts[:3]] == [False, False, True]
    for old in seen[:3]:
        assert_state_equal(old, initial)
    assert len(records) == sum(a.accepted for a in attempts) == len(controller.steps)
    assert any(record.load_factor == 1.0 for record in records)
    assert records[-1].load_factor == controller.load_factor == 0.7
    for record in records:
        expected = record.load_factor * case.reference_displacement(case.grid.cell_centers).ravel(order="F")
        np.testing.assert_allclose(record.state.x[:2 * ops.num_cells], expected, atol=1e-13)
        assert all(state.alpha == 0 for state in record.state.material_states)
    assert_state_equal(initial, TpsaState.zeros(case.grid))
    assert all(point.trial is None for point in case.material_points)


def test_adaptive_growth_reversal_and_short_endpoint(setup: Setup) -> None:
    case, ops, transfer, initial = setup
    controller = LoadStepController(
        ops, transfer, case.material_points, initial, case.bc_values,
        jacobian_factory=lambda old: lambda x, evaluation: ops.A,
    )
    options = AdaptiveLoadOptions(initial_increment=0.125, min_increment=0.1,
                                  max_increment=0.25, growth_factor=2)
    records = controller.run_adaptive([0, 0.4, 0.4, -0.1], options=options)
    factors = [record.load_factor for record in records]
    np.testing.assert_allclose(factors[:3], [0.125, 0.375, 0.4])
    assert factors[-1] == -0.1
    assert factors.count(0.4) == 1  # Repeated adaptive targets are no-ops.
    assert factors[3] == 0.4 - 0.125  # Restart increment at direction reversal.
    assert all(abs(a.load_factor - a.start_factor) <= 0.25 for a in controller.attempts)
    assert all(a.accepted for a in controller.attempts)
    records[0].state.material_states[0].alpha = 10
    assert controller.steps[0].state.material_states[0].alpha == 0


@pytest.mark.parametrize("limit", ["minimum", "retries", "steps"])
def test_adaptive_limits_preserve_last_accepted_state(setup: Setup, limit: str) -> None:
    case, ops, transfer, initial = setup
    broken = False

    def factory(old: TpsaState) -> JacobianCallback:
        return lambda x, evaluation: (2 * ops.A if broken else ops.A)

    controller = LoadStepController(ops, transfer, case.material_points, initial,
                                    case.bc_values, jacobian_factory=factory,
                                    max_iterations=1, atol=(1e-4, 1e-11, 1e-11))
    first = controller.advance(0.25)
    broken = limit != "steps"
    options = AdaptiveLoadOptions(initial_increment=0.25, max_increment=0.25,
                                  min_increment=0.125 if limit == "minimum" else 1e-4,
                                  max_retries=1 if limit == "retries" else 8,
                                  max_steps=2)
    with pytest.raises(AdaptiveLoadStepError, match={
        "minimum": "minimum increment", "retries": "retry limit", "steps": "accepted-step limit",
    }[limit]) as caught:
        controller.run_adaptive([1.0], options=options)
    assert caught.value.target == 1.0
    if broken:
        assert controller.load_factor == 0.25
        assert len(controller.steps) == 1
        assert_state_equal(controller.state, first.state)
        assert len(controller.attempts) == 3  # One acceptance, two failed attempts.
        assert isinstance(caught.value.__cause__, LoadStepError)
    else:
        assert controller.load_factor == 0.75
        assert len(controller.steps) == 3
        assert_state_equal(controller.state, controller.steps[-1].state)


@pytest.mark.parametrize("failure", [ValueError("invalid Jacobian"), RuntimeError("linear/material failure")])
def test_adaptive_does_not_retry_other_errors(setup: Setup, failure: Exception) -> None:
    case, ops, transfer, initial = setup

    def factory(old: TpsaState) -> JacobianCallback:
        raise failure

    controller = LoadStepController(ops, transfer, case.material_points, initial,
                                    case.bc_values, jacobian_factory=factory)
    with pytest.raises(LoadStepError) as caught:
        controller.run_adaptive([1.0])
    assert caught.value.__cause__ is failure
    assert len(controller.attempts) == 1 and not controller.attempts[0].accepted
    assert len(controller.steps) == 0
    assert_state_equal(controller.state, initial)


@pytest.mark.parametrize("changes", [
    {"initial_increment": 0}, {"min_increment": 0}, {"max_increment": float("inf")},
    {"min_increment": 0.1}, {"max_increment": 0.01}, {"cutback_factor": 1},
    {"cutback_factor": 0}, {"growth_factor": 0.9}, {"growth_factor": float("nan")},
    {"target_iterations": -1}, {"max_retries": -1}, {"max_steps": 0},
    {"max_retries": 1.5}, {"max_steps": True},
])
def test_adaptive_options_reject_invalid_values(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(AdaptiveLoadOptions(), **changes)


def test_adaptive_validates_all_targets_before_advancing(setup: Setup) -> None:
    case, ops, transfer, initial = setup
    controller = LoadStepController(ops, transfer, case.material_points, initial, case.bc_values)
    for targets in (np.array([0.1, np.nan]), np.array([[0.1]]), np.array([1e308, -1e308])):
        with pytest.raises(ValueError):
            controller.run_adaptive(targets)
        assert len(controller.attempts) == 0
        assert_state_equal(controller.state, initial)
    assert controller.run_adaptive([]) == ()


def test_adaptive_tries_minimum_increment_before_stopping(setup: Setup) -> None:
    case, ops, transfer, initial = setup
    controller = LoadStepController(
        ops, transfer, case.material_points, initial, case.bc_values,
        jacobian_factory=lambda old: lambda x, evaluation: 2 * ops.A, max_iterations=1,
    )
    with pytest.raises(AdaptiveLoadStepError, match="minimum increment"):
        controller.run_adaptive([1], options=AdaptiveLoadOptions(
            initial_increment=0.15, min_increment=0.1, max_increment=0.15,
        ))
    assert [a.load_factor for a in controller.attempts] == [0.15, 0.1]
    assert_state_equal(controller.state, initial)


def test_adaptive_precision_stall_stops_without_attempting_a_zero_increment(setup: Setup) -> None:
    case, ops, transfer, initial = setup
    controller = LoadStepController(
        ops, transfer, case.material_points, initial, np.zeros_like(case.bc_values),
        initial_load_factor=1e16,
    )
    with pytest.raises(AdaptiveLoadStepError, match="floating-point progress"):
        controller.run_adaptive([1e16 + 2])
    assert controller.attempts == () and controller.steps == ()
    assert controller.load_factor == 1e16


def test_adaptive_iterator_checks_each_accepted_step_before_advancing(setup: Setup) -> None:
    case, ops, transfer, initial = setup
    controller = LoadStepController(ops, transfer, case.material_points, initial, case.bc_values)
    records = controller.iter_adaptive([1.0])
    first = next(records)
    assert len(controller.steps) == 1
    assert first.load_factor == controller.load_factor == 0.05
    records.close()
    assert len(controller.steps) == 1
    assert_state_equal(controller.state, first.state)


def test_adaptive_zero_retry_budget_stops_after_first_failure(setup: Setup) -> None:
    case, ops, transfer, initial = setup
    controller = LoadStepController(
        ops, transfer, case.material_points, initial, case.bc_values,
        jacobian_factory=lambda old: lambda x, evaluation: 2 * ops.A,
        max_iterations=1,
    )
    with pytest.raises(AdaptiveLoadStepError, match="retry limit") as caught:
        controller.run_adaptive([1], options=AdaptiveLoadOptions(max_retries=0))
    assert caught.value.load_factor == 0.05
    assert len(controller.attempts) == 1
    assert not controller.attempts[0].accepted
    assert controller.steps == () and controller.load_factor == 0
    assert_state_equal(controller.state, initial)


def test_adaptive_retry_budget_resets_after_acceptance_and_during_unloading(setup: Setup) -> None:
    case, ops, transfer, initial = setup
    amplitude = float(np.max(np.abs(case.bc_values)))

    def factory(old: TpsaState) -> JacobianCallback:
        def tangent(x: NDArray[np.float64], evaluation: TpsaResidual) -> JacobianMatrix:
            increment = float(np.max(np.abs(evaluation.trial.bc_values - old.bc_values))) / amplitude
            return 2 * ops.A if increment > 0.15 else ops.A
        return tangent

    controller = LoadStepController(
        ops, transfer, case.material_points, initial, case.bc_values,
        jacobian_factory=factory, max_iterations=1, atol=(1e-4, 1e-11, 1e-11),
    )
    options = AdaptiveLoadOptions(initial_increment=0.25, max_increment=0.25,
                                  growth_factor=2, max_retries=1)
    records = controller.run_adaptive([0.75, 0], options=options)
    failures = [attempt for attempt in controller.attempts if not attempt.accepted]
    assert len(failures) > options.max_retries
    assert any(a.load_factor > a.start_factor for a in failures)
    assert any(a.load_factor < a.start_factor for a in failures)
    # Each failed attempt can use one retry from exactly the same accepted state.
    for failed, retried in zip(controller.attempts, controller.attempts[1:]):
        if not failed.accepted:
            assert retried.accepted and retried.start_factor == failed.start_factor
            assert abs(retried.load_factor - retried.start_factor) < abs(failed.load_factor - failed.start_factor)
    assert records[-1].load_factor == controller.load_factor == 0
    np.testing.assert_allclose(controller.state.x[:2 * ops.num_cells], 0, atol=1e-13)


@pytest.mark.parametrize("growth_factor,target_iterations", [(1.0, 4), (2.0, 0)])
def test_adaptive_growth_can_be_disabled_or_withheld_for_slow_solves(
    setup: Setup, growth_factor: float, target_iterations: int,
) -> None:
    case, ops, transfer, initial = setup
    controller = LoadStepController(
        ops, transfer, case.material_points, initial, case.bc_values,
        jacobian_factory=lambda old: lambda x, evaluation: ops.A,
    )
    records = controller.run_adaptive([0.5], options=AdaptiveLoadOptions(
        initial_increment=0.125, max_increment=0.5,
        growth_factor=growth_factor, target_iterations=target_iterations,
    ))
    assert [r.load_factor for r in records] == [0.125, 0.25, 0.375, 0.5]
    assert all(r.newton.iterations > 0 for r in records)


def test_adaptive_short_waypoint_preserves_proposed_increment(setup: Setup) -> None:
    case, ops, transfer, initial = setup
    controller = LoadStepController(ops, transfer, case.material_points, initial, case.bc_values)
    records = controller.run_adaptive([0.1, 0.5], options=AdaptiveLoadOptions(
        initial_increment=0.25, max_increment=0.25, growth_factor=1,
    ))
    np.testing.assert_allclose([r.load_factor for r in records], [0.1, 0.35, 0.5])
    assert records[0].load_factor == 0.1 and records[-1].load_factor == 0.5


def test_adaptive_failed_endpoint_below_minimum_is_not_subdivided(setup: Setup) -> None:
    case, ops, transfer, initial = setup
    controller = LoadStepController(
        ops, transfer, case.material_points, initial, case.bc_values,
        jacobian_factory=lambda old: lambda x, evaluation: 2 * ops.A,
        max_iterations=1,
    )
    with pytest.raises(AdaptiveLoadStepError, match="minimum increment"):
        controller.run_adaptive([0.01], options=AdaptiveLoadOptions(
            initial_increment=0.2, min_increment=0.1,
        ))
    assert [a.load_factor for a in controller.attempts] == [0.01]
    assert_state_equal(controller.state, initial)
    assert controller.steps == ()


def test_adaptive_substeps_scale_body_and_auxiliary_sources_absolutely(setup: Setup) -> None:
    case, ops, transfer, initial = setup
    sources = np.zeros(4 * ops.num_cells)
    sources[:2 * ops.num_cells] = (np.array([[2e5], [-1e5]]) * case.grid.cell_volumes).ravel(order="F")
    sources[2 * ops.num_cells:] = 1e-6
    exact = np.asarray(sps.linalg.spsolve(ops.A, ops.assemble_rhs(case.bc_values, sources=sources)))
    controller = LoadStepController(
        ops, transfer, case.material_points, initial, case.bc_values,
        reference_sources=sources, jacobian_factory=lambda old: lambda x, evaluation: ops.A,
        rtol=(0, 0, 0),
    )
    records = controller.run_adaptive([0.6, -0.1])
    for record in records:
        np.testing.assert_array_equal(record.sources, record.load_factor * sources)
        np.testing.assert_array_equal(record.state.bc_values, record.load_factor * case.bc_values)
        np.testing.assert_allclose(record.state.x, record.load_factor * exact, rtol=1e-8, atol=1e-4)
        assert np.all(record.newton.residual_norms[-1] <= record.newton.thresholds)
    assert records[-1].load_factor == -0.1


def test_adaptive_restart_after_step_limit_retains_plastic_history(
    plastic_setup: tuple[PlaneStrainTpsa, TpsaOperators, CellToFaceTransfer, LoadStepController],
) -> None:
    case, ops, transfer, original = plastic_setup
    peak = original.state
    options = AdaptiveLoadOptions(initial_increment=0.005, max_increment=0.005,
                                  growth_factor=1, max_steps=1)
    with pytest.raises(AdaptiveLoadStepError, match="accepted-step limit"):
        original.run_adaptive([0.98], options=options)
    assert original.load_factor == 0.995
    saved = original.state
    earlier = original.steps
    restart = LoadStepController(
        ops, transfer, case.material_points, saved, case.bc_values,
        initial_load_factor=original.load_factor,
    )
    options = replace(options, max_steps=10)
    continued = original.run_adaptive([0.98], options=options)
    resumed = restart.run_adaptive([0.98], options=options)
    assert original.load_factor == restart.load_factor == 0.98
    assert len(continued) == len(resumed)
    for a, b in zip(continued, resumed, strict=True):
        assert a.load_factor == b.load_factor
        assert_state_equal(a.state, b.state)
        np.testing.assert_array_equal(a.newton.residual_norms, b.newton.residual_norms)
    for before, after in zip(earlier, original.steps[:len(earlier)], strict=True):
        assert_state_equal(before.state, after.state)
    for final, at_peak in zip(restart.state.material_states, peak.material_states, strict=True):
        assert final.alpha == pytest.approx(at_peak.alpha, abs=1e-11)
        np.testing.assert_allclose(final.plastic_strain.to_numpy(), at_peak.plastic_strain.to_numpy(), atol=1e-11)
    assert_state_equal(saved, earlier[-1].state)


@pytest.mark.parametrize("adaptive", [False, True], ids=["fixed", "adaptive"])
@pytest.mark.parametrize("analytical", [False, True], ids=["finite_difference", "analytical"])
def test_callback_boundary_scales_through_loading_and_unloading(
    adaptive: bool, analytical: bool,
) -> None:
    calls = 0

    def displacement(centers: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        x, y = centers[:2]
        return 1e-5 * np.vstack((1 + x**2 + y, y**2 - 0.5*x))

    case = PlaneStrainTpsa(cells_per_axis=2, boundary_displacement=displacement)
    reference_solution = case.solve()
    reference_bc = case.bc_values.copy()
    ops, transfer = TpsaOperators(case), CellToFaceTransfer(case.grid)
    controller = LoadStepController(
        ops, transfer, case.material_points, TpsaState.zeros(case.grid), reference_bc,
        jacobian_factory=(lambda old: AnalyticalJacobian(ops, transfer, case.material_points, old))
        if analytical else None,
    )
    targets = [0.3, 1.0, 0.45, -0.1, 0.0]
    records = controller.run_adaptive(targets) if adaptive else controller.run(targets)
    assert calls == 1  # Spatial input is sampled during preparation, not each step.
    assert controller.load_factor == targets[-1]
    for target in targets:
        assert target in [record.load_factor for record in records]
    for record in records:
        factor, state = record.load_factor, record.state
        np.testing.assert_array_equal(state.bc_values, factor * reference_bc)
        # This small nonuniform problem stays elastic: compare with a direct TPSA solve.
        np.testing.assert_allclose(state.x, factor * reference_solution, rtol=1e-8, atol=1e-4)
        np.testing.assert_allclose(state.x[:2 * ops.num_cells],
                                   factor * reference_solution[:2 * ops.num_cells], atol=1e-13)
        assert all(history.alpha == 0 for history in state.material_states)
        assert np.all(record.newton.residual_norms[-1] <= record.newton.thresholds)
    np.testing.assert_array_equal(case.bc_values, reference_bc)
