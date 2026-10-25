"""Accepted total histories, restart, rollback, and retained-face load schedules."""

from copy import deepcopy
from typing import TypeAlias

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray

from utils.load_schedule import AdaptiveLoadOptions, AdaptiveLoadStepError, LoadStepError
from utils.newton_utils import JacobianMatrix, NewtonConvergenceError
from alternatives.plane_strain import PlaneStrainTpsa
from utils.state import TpsaState
from material.material_state import MaterialPoint, MaterialState
from prototype import loading
from prototype.jacobian import RetainedAnalyticalJacobian, RetainedFiniteDifferenceJacobian
from prototype.loading import LoadStepController
from prototype.newton import BacktrackingOptions, JacobianCallback
from prototype.operators import RetainedFaceOperators
from prototype.residual import RetainedFaceResidual
from prototype.state import RetainedState, assemble_state_residual
from material.tensor import strain

FloatArray: TypeAlias = NDArray[np.float64]
Setup: TypeAlias = tuple[PlaneStrainTpsa, RetainedFaceOperators, RetainedState, FloatArray]


def make_setup(*, plastic: bool = False, cells: int = 2) -> Setup:
    case = PlaneStrainTpsa(cells_per_axis=cells)
    case.prepare_simulation()
    ops = RetainedFaceOperators(case.grid, case.bc,
        mu=case.material.isotropic_shear_modulus, lam=case.material.lame_parameter)
    gradient = np.array([[1., .15], [-.05, -.075]]) * (.004 if plastic else 1e-5)
    return case, ops, RetainedState.zeros(ops), gradient @ case.grid.face_centers[:2]


def assert_history_equal(a: MaterialState, b: MaterialState) -> None:
    assert a.alpha == b.alpha
    for name in ("stress", "plastic_strain", "backstress"):
        np.testing.assert_array_equal(getattr(a, name).to_numpy(), getattr(b, name).to_numpy())


def assert_state_equal(a: RetainedState, b: RetainedState) -> None:
    for name in ("side_traction", "body_force"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in ("x", "bc_values", "epsilon", "traction", "face_displacement"):
        np.testing.assert_array_equal(getattr(a.tpsa, name), getattr(b.tpsa, name))
    for x, y in zip(a.tpsa.material_states, b.tpsa.material_states, strict=True):
        assert_history_equal(x, y)


def independent_balance(ops: RetainedFaceOperators, state: RetainedState) -> FloatArray:
    """Loop over each outward side; do not use the production balance assembler."""
    result = np.zeros(ops.num_dofs)
    assert ops.mu is not None and ops.lam is not None and state.tpsa.face_displacement is not None
    for s, (face, cell) in enumerate(zip(ops.side_faces, ops.side_cells, strict=True)):
        t = state.side_traction[:, s]
        d = state.tpsa.face_displacement[:, face]
        area, n = ops.side_areas[s], ops.side_normals[:, s]
        result[2 * cell:2 * cell + 2] += t
        result[2 * ops.num_cells + cell] += area * (d[0] * n[1] - d[1] * n[0])
        result[3 * ops.num_cells + cell] += area * np.dot(d, n)
        if ops.face_dofs[0, face] >= 0:
            result[ops.face_dofs[:, face]] += t
    result[ops.u_slice] -= state.body_force.ravel("F")
    result[ops.r_slice] -= ops.grid.cell_volumes / ops.mu * state.tpsa.x[ops.r_slice]
    result[ops.p_slice] -= ops.grid.cell_volumes / ops.lam * state.tpsa.x[ops.p_slice]
    return result


@pytest.mark.parametrize("cells", [1, 3])
def test_zero_state_and_elastic_schedule_accumulate_all_fields(cells: int) -> None:
    case, ops, old, boundary = make_setup(cells=cells)
    controller = LoadStepController(ops, case.material_points, old, boundary)
    records = controller.run([.25, 1., 1., .5, 0., -.2])
    previous = old
    for record in records:
        current, ev = record.state, record.newton.evaluation
        np.testing.assert_array_equal(current.tpsa.x, previous.tpsa.x + record.newton.z[ops.cell_slice])
        np.testing.assert_array_equal(current.side_traction, previous.side_traction + ev.traction_increment)
        assert current.tpsa.face_displacement is not None and previous.tpsa.face_displacement is not None
        np.testing.assert_allclose(current.tpsa.face_displacement, record.load_factor * boundary, atol=1e-18)
        np.testing.assert_array_equal(current.tpsa.face_displacement[:, ops.boundary_faces],
                                      current.tpsa.bc_values[:, ops.boundary_faces])
        np.testing.assert_array_equal(current.tpsa.epsilon, ev.material_trial.epsilon)
        for a, b in zip(current.tpsa.material_states, ev.material_trial.material_states, strict=True):
            assert_history_equal(a, b)
        stress = current.tpsa.material_states[0].stress.to_numpy()
        np.testing.assert_allclose(current.side_traction,
            stress[:2, :2] @ ops.side_normals * ops.side_areas, atol=2e-8)
        np.testing.assert_allclose(current.tpsa.traction, stress[:2, :2] @ case.grid.face_normals[:2], atol=2e-8)
        residual = independent_balance(ops, current)
        np.testing.assert_allclose(assemble_state_residual(ops, current), residual, atol=1e-18)
        for block, threshold in zip((ops.u_slice, ops.r_slice, ops.p_slice, ops.face_slice), record.newton.thresholds, strict=True):
            assert np.linalg.norm(residual[block]) <= threshold
        previous = current
    assert records[2].newton.iterations == 0
    assert_state_equal(controller.state, records[-1].state)
    assert all(attempt.accepted for attempt in controller.attempts)
    assert_state_equal(old, RetainedState.zeros(ops))
    assert all(p.trial is None and p.committed.alpha == 0 for p in case.material_points)


@pytest.mark.parametrize("numerical", [False, True])
def test_plastic_path_matches_independent_sequential_return_maps(numerical: bool) -> None:
    case, ops, old, boundary = make_setup(plastic=True)
    factory = (lambda state: RetainedFiniteDifferenceJacobian(ops, case.material_points, state)) if numerical else None
    controller = LoadStepController(ops, case.material_points, old, boundary,
        jacobian_factory=factory, line_search=BacktrackingOptions())
    local = deepcopy(case.material_points[0])
    previous = old
    for record in controller.run([.25, .5, 1., .99, .98, 1.1]):
        total = record.state.tpsa
        history, _ = local.update(strain(total.epsilon[:, :, 0] - previous.tpsa.epsilon[:, :, 0]))
        for cell in total.material_states:
            assert cell.alpha == pytest.approx(history.alpha, rel=1e-7, abs=1e-12)
            np.testing.assert_allclose(cell.stress.to_numpy(), history.stress.to_numpy(), rtol=1e-7, atol=1.)
        np.testing.assert_allclose(record.state.side_traction,
            history.stress.to_numpy()[:2, :2] @ ops.side_normals * ops.side_areas, rtol=1e-7, atol=1.)
        local.commit()
        previous = record.state
    assert all(h.alpha > 0 for h in controller.state.tpsa.material_states)
    assert all(p.trial is None and p.committed.alpha == 0 for p in case.material_points)


def test_body_force_is_differenced_once_and_absolute_equilibrium_is_preserved() -> None:
    case, ops, old, boundary = make_setup()
    body = np.arange(2 * ops.num_cells, dtype=float).reshape(2, -1) * 1e4
    controller = LoadStepController(ops, case.material_points, old, boundary, reference_body_force=body)
    for record in controller.run([.3, 1., .8, -.2, 0.]):
        np.testing.assert_array_equal(record.state.body_force, record.load_factor * body)
        np.testing.assert_allclose(independent_balance(ops, record.state), 0, atol=2e-7)
    resumed = LoadStepController(ops, case.material_points, controller.state, boundary,
        reference_body_force=body, initial_load_factor=0.)
    assert_state_equal(controller.advance(.5).state, resumed.advance(.5).state)


def test_restart_from_plastic_history_matches_continuous_unloading_and_reloading() -> None:
    case, ops, old, boundary = make_setup(plastic=True)
    controller = LoadStepController(ops, case.material_points, old, boundary)
    controller.run([.25, .5, 1.])
    restart = controller.state
    resumed = LoadStepController(ops, case.material_points, restart, boundary, initial_load_factor=1.)
    for a, b in zip(controller.run([.99, .98, 1.1]), resumed.run([.99, .98, 1.1]), strict=True):
        assert_state_equal(a.state, b.state)
        np.testing.assert_array_equal(a.newton.residual_norms, b.newton.residual_norms)
    assert_state_equal(restart, controller.steps[2].state)


def test_old_side_imbalance_is_not_erased_by_the_oriented_face_average() -> None:
    case, ops, old, boundary = make_setup()
    side = old.side_traction.copy()
    face = ops.free_faces[0]
    side[:, ops.side_faces == face] = np.array([[3.], [5.]])
    initial = RetainedState(ops, old.tpsa, side, old.body_force)
    np.testing.assert_array_equal(initial.tpsa.traction, 0.)
    assert np.linalg.norm(assemble_state_residual(ops, initial)) > 0
    controller = LoadStepController(ops, case.material_points, initial, boundary)
    accepted = controller.advance(0.)
    assert accepted.newton.iterations == 1
    np.testing.assert_allclose(independent_balance(ops, accepted.state), 0., atol=1e-12)
    np.testing.assert_array_equal(accepted.state.side_traction, side + accepted.newton.evaluation.traction_increment)
    np.testing.assert_array_equal(initial.side_traction, side)


def test_inputs_and_returned_records_do_not_alias_accepted_state() -> None:
    case, ops, old, boundary = make_setup(plastic=True)
    saved_bc = boundary.copy()
    boundary[:, ops.free_faces] = np.nan
    body = np.zeros((2, ops.num_cells))
    controller = LoadStepController(ops, case.material_points, old, boundary, reference_body_force=body)
    boundary[:] = body[:] = np.nan
    old.tpsa.epsilon[:] = 2
    old.side_traction[:] = 3
    record = controller.advance(1.)
    saved = controller.state
    assert saved.tpsa.face_displacement is not None
    np.testing.assert_array_equal(saved.tpsa.face_displacement[:, ops.boundary_faces], saved_bc[:, ops.boundary_faces])
    record.state.side_traction[:] = 0
    record.state.tpsa.x[:] = 0
    record.state.tpsa.material_states[0].alpha = 20
    assert record.newton.evaluation.material_trial.material_states[0].alpha == saved.tpsa.material_states[0].alpha
    record.newton.evaluation.material_trial.material_states[0].alpha = 30
    record.newton.z[:] = 9
    exported = controller.steps[0]
    exported.state.body_force[:] = 4
    exported.state.tpsa.face_displacement = None
    assert_state_equal(controller.state, saved)
    assert_state_equal(controller.steps[0].state, saved)
    assert controller.steps[0].newton.evaluation.material_trial.material_states[0].alpha == saved.tpsa.material_states[0].alpha


@pytest.mark.parametrize("failure", ["iterations", "linear", "factory", "material", "acceptance"])
def test_failed_attempt_is_atomic_and_retry_matches_clean_solve(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    case, ops, old, boundary = make_setup()
    for point in case.material_points:
        point.update(strain(np.diag([.004, 0., 0.])))
    point_states = deepcopy(case.material_points)
    fail = False

    def factory(state: TpsaState) -> JacobianCallback:
        if fail and failure == "factory":
            state.material_states[0].alpha = 100
            state.face_displacement = None
            raise RuntimeError("Injected factory failure")
        matrix = RetainedAnalyticalJacobian(ops, case.material_points, state).L
        if fail and failure == "iterations":
            matrix *= 2
        elif fail and failure == "linear":
            matrix = sps.csr_array(matrix.shape)
        return lambda z, evaluation: matrix

    controller = LoadStepController(ops, case.material_points, old, boundary, jacobian_factory=factory, max_iterations=1)
    accepted = controller.advance(.25)
    fail = True
    with monkeypatch.context() as patch:
        if failure == "material":
            original = MaterialPoint.update
            calls = 0

            def failing_update(self: MaterialPoint, increment: strain) -> tuple[MaterialState, None]:
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("Injected local failure after two cells")
                return original(self, increment)

            patch.setattr(MaterialPoint, "update", failing_update)
        elif failure == "acceptance":
            def invalid_state(*args: object, **kwargs: object) -> RetainedState:
                raise ValueError("Injected accumulated-state validation failure")

            patch.setattr(loading, "RetainedState", invalid_state)
        with pytest.raises(LoadStepError) as error:
            controller.run([.5, 1.])
    assert error.value.step_index == 2 and error.value.load_factor == .5
    assert controller.load_factor == .25 and len(controller.steps) == 1
    assert [a.accepted for a in controller.attempts] == [True, False]
    assert_state_equal(controller.state, accepted.state)
    for a, b in zip(case.material_points, point_states, strict=True):
        assert_history_equal(a.committed, b.committed)
        assert a.trial is not None and b.trial is not None
        assert_history_equal(a.trial, b.trial)
    fail = False
    clean = LoadStepController(ops, case.material_points, accepted.state, boundary, initial_load_factor=.25)
    assert_state_equal(controller.advance(.5).state, clean.advance(.5).state)


def test_adaptive_retries_from_same_total_history_and_reaches_reversed_targets() -> None:
    case, ops, old, boundary = make_setup()
    seen: list[TpsaState] = []
    amplitude = float(np.max(abs(boundary)))

    def factory(state: TpsaState) -> JacobianCallback:
        seen.append(state.copy())
        matrix = RetainedAnalyticalJacobian(ops, case.material_points, state).L

        def callback(z: FloatArray, evaluation: RetainedFaceResidual) -> JacobianMatrix:
            change = np.max(abs(evaluation.bc_values - state.bc_values)) / amplitude
            return 2 * matrix if change > .25 else matrix

        return callback

    controller = LoadStepController(ops, case.material_points, old, boundary, jacobian_factory=factory, max_iterations=1)
    records = controller.run_adaptive([1., 1., .7], options=AdaptiveLoadOptions(
        initial_increment=.6, max_increment=.6, min_increment=.05, growth_factor=2, target_iterations=1))
    np.testing.assert_allclose([a.load_factor for a in controller.attempts[:3]], [.6, .3, .15])
    assert [a.accepted for a in controller.attempts[:3]] == [False, False, True]
    for state in seen[:3]:
        np.testing.assert_array_equal(state.x, 0)
        np.testing.assert_array_equal(state.face_displacement, 0)
        assert all(h.alpha == 0 for h in state.material_states)
    assert [r.load_factor for r in records].count(1.) == 1
    assert controller.load_factor == .7 and records[-1].load_factor == .7
    assert len(controller.steps) == sum(a.accepted for a in controller.attempts)
    for record in records:
        np.testing.assert_allclose(independent_balance(ops, record.state), 0, atol=1e-7)


@pytest.mark.parametrize("failure", ["newton", "material"])
def test_adaptive_exhaustion_and_nonretryable_errors_preserve_last_state(failure: str) -> None:
    case, ops, old, boundary = make_setup()
    broken = False

    def factory(state: TpsaState) -> JacobianCallback:
        if broken:
            if failure == "newton":
                raise NewtonConvergenceError("Injected nonconvergence")
            raise RuntimeError("Injected local failure")
        return RetainedAnalyticalJacobian(ops, case.material_points, state)

    controller = LoadStepController(ops, case.material_points, old, boundary, jacobian_factory=factory)
    saved = controller.advance(.25).state
    broken = True
    error = AdaptiveLoadStepError if failure == "newton" else LoadStepError
    with pytest.raises(error):
        controller.run_adaptive([1.], options=AdaptiveLoadOptions(max_retries=1))
    assert len(controller.attempts) == (3 if failure == "newton" else 2)
    assert controller.load_factor == .25
    assert_state_equal(controller.state, saved)


def test_adaptive_iterator_can_stop_after_one_atomic_acceptance() -> None:
    case, ops, old, boundary = make_setup()
    controller = LoadStepController(ops, case.material_points, old, boundary)
    iterator = controller.iter_adaptive([1.])
    first = next(iterator)
    iterator.close()
    assert len(controller.steps) == 1 and controller.load_factor == first.load_factor
    assert_state_equal(controller.state, first.state)


@pytest.mark.parametrize("defect", ["faces_missing", "faces_bc", "faces_strain", "side_shape", "side_nan", "body_shape", "body_nan", "material", "counts"])
def test_invalid_total_state_is_rejected(defect: str) -> None:
    _, ops, old, _ = make_setup()
    if defect == "faces_missing":
        old.tpsa.face_displacement = None
    elif defect in ("faces_bc", "faces_strain"):
        assert old.tpsa.face_displacement is not None
        face = ops.boundary_faces[0] if defect == "faces_bc" else ops.free_faces[0]
        old.tpsa.face_displacement[0, face] = .01
    elif defect == "side_shape":
        old.side_traction = old.side_traction[:, :-1]
    elif defect == "side_nan":
        old.side_traction[0, 0] = np.nan
    elif defect == "body_shape":
        old.body_force = old.body_force.ravel()
    elif defect == "body_nan":
        old.body_force[0, 0] = np.inf
    elif defect == "material":
        old.tpsa.material_states[0].alpha = -1
    else:
        old.tpsa.material_states.pop()
    with pytest.raises(ValueError):
        RetainedState(ops, old.tpsa, old.side_traction, old.body_force)


@pytest.mark.parametrize("defect", ["factor", "bc_factor", "body_factor", "mu", "missing_lam", "point_count", "body_shape", "bc_nan", "iterations"])
def test_incompatible_controller_inputs_are_rejected(defect: str) -> None:
    case, ops, old, boundary = make_setup()
    body: FloatArray = np.zeros((2, ops.num_cells))
    factor, iterations = 0., 20
    if defect == "factor":
        factor = np.nan
    elif defect == "bc_factor":
        factor = 1.
    elif defect == "body_factor":
        old.body_force[:] = 1.
    elif defect == "mu":
        case.material.young_modulus *= 2
    elif defect == "missing_lam":
        ops = RetainedFaceOperators(case.grid, case.bc, mu=ops.mu)
    elif defect == "point_count":
        case.material_points.pop()
    elif defect == "body_shape":
        body = body.ravel()
    elif defect == "bc_nan":
        boundary[0, ops.boundary_faces[0]] = np.nan
    else:
        iterations = -1
    with pytest.raises(ValueError):
        LoadStepController(ops, case.material_points, old, boundary,
            reference_body_force=body, initial_load_factor=factor, max_iterations=iterations)


def test_invalid_schedules_are_checked_before_any_attempt() -> None:
    case, ops, old, boundary = make_setup()
    controller = LoadStepController(ops, case.material_points, old, boundary)
    for method in (controller.run, controller.run_adaptive):
        for values in ([.2, np.nan], [[.2]]):
            with pytest.raises(ValueError):
                method(np.array(values))
    with pytest.raises(ValueError):
        controller.advance(np.inf)
    assert controller.run([]) == controller.run_adaptive([]) == ()
    assert controller.attempts == () and controller.steps == ()
    assert_state_equal(controller.state, old)
