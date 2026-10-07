"""Accepted load sequences, constitutive history, restart, and failure isolation."""

from copy import deepcopy
from typing import TypeAlias

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]

from coupling.jacobian import FiniteDifferenceJacobian
from coupling.loading import LoadStepController, LoadStepError
from coupling.newton import JacobianCallback, NewtonConvergenceError
from coupling.plane_strain import PlaneStrainTpsa
from coupling.residual import TpsaOperators, TpsaState
from coupling.transfer import CellToFaceTransfer
from material_state import MaterialPoint, MaterialState
from tensor import strain

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
