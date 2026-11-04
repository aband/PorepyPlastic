"""Total-state balances, load acceptance, snapshots and adaptive recovery."""

from copy import deepcopy
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

import core.loading as loading
from core.layout import FloatArray
from core.loading import LoadStepController, LoadStepResult, assemble_state_residual
from core.residual import LoadIncrement, ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory, SolutionState
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import (
    HardeningParameters, isotropic_hardening_K, kinematic_hardening_H,
)
from material.tensor import strain
from utils.load_schedule import AdaptiveLoadOptions, AdaptiveLoadStepError, LoadStepError
from utils.newton_utils import LocalUpdateError, NewtonConvergenceError, NewtonDivergenceError


def _assembler(
    dim: int, boundary_type: str, *, single_cell: bool = False, uniform: bool = False,
) -> ResidualAssembler:
    coordinates = [
        np.array([0.0, 0.3, 1.0]),
        np.array([0.0, 0.4, 1.3]),
        np.array([0.0, 0.2, 0.8]),
    ]
    if single_cell:
        coordinates = [values[[0, -1]] for values in coordinates]
    grid = pp.TensorGrid(
        coordinates[0], coordinates[1], coordinates[2] if dim == 3 else None,
    )
    grid.compute_geometry()
    boundary = grid.get_all_boundary_faces()
    prescribed = {
        "dirichlet": boundary,
        "neumann": boundary[:0],
        "mixed": boundary[::2],
    }[boundary_type]
    bc = pp.BoundaryConditionVectorial(grid, prescribed, "dir")
    points = []
    for cell in range(grid.num_cells):
        material = _materialProperty()
        material.young_modulus = 12.0 + (0.0 if uniform else 2.0 * cell)
        material.poisson_ratio = 0.25
        points.append(MaterialPoint(
            material, HardeningParameters(0.05, 0.05, 1.0, 0.4, 0.0),
            vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H,
        ))
    return ResidualAssembler(grid, bc, points)


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def dim(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture(params=["dirichlet", "neumann", "mixed"])
def assembler(dim: int, request: pytest.FixtureRequest) -> ResidualAssembler:
    return _assembler(dim, str(request.param))


def _zero_state(assembler: ResidualAssembler) -> SolutionState:
    return SolutionState.zeros(
        assembler.layout,
        num_sides=assembler.operators.num_sides,
        num_neumann_faces=assembler.neumann_faces.size,
    )


def _candidate(
    assembler: ResidualAssembler, old: SolutionState, *, scale: float, seed: int,
) -> tuple[SolutionState, ResidualEvaluation]:
    """Accumulate actual trial fields; deliberately retain a nonzero imbalance."""
    layout = assembler.layout
    rng = np.random.default_rng(seed)
    z = scale * rng.normal(size=layout.size)
    loads = LoadIncrement(
        scale * rng.normal(size=(layout.dim, layout.dirichlet_faces.size)),
        scale * rng.normal(size=(layout.dim, assembler.neumann_faces.size)),
        scale * rng.normal(size=layout.cell_size),
    )
    evaluation = assembler.evaluate(z, loads, old.material)
    trial = evaluation.material_trial
    assert trial is not None
    candidate = SolutionState(
        x=old.x + z[layout.cell_slice],
        face_displacement=old.face_displacement + evaluation.face_increment,
        side_traction=old.side_traction + evaluation.traction_increment,
        material=trial.history,
        neumann_values=old.neumann_values + loads.neumann_increment,
        cell_source=old.cell_source + loads.cell_source_increment,
    )
    return candidate, evaluation


def test_zero_state_has_zero_residual(assembler: ResidualAssembler) -> None:
    residual = assemble_state_residual(assembler, _zero_state(assembler))
    assert residual.shape == (assembler.layout.size,)
    assert residual.dtype == np.float64
    np.testing.assert_array_equal(residual, np.zeros(assembler.layout.size))


@pytest.mark.parametrize("scale", [1e-5, 0.02], ids=["elastic", "plastic"])
def test_total_residual_equals_committed_plus_incremental(
    assembler: ResidualAssembler, scale: float,
) -> None:
    old, first = _candidate(assembler, _zero_state(assembler), scale=scale, seed=17)
    candidate, increment = _candidate(assembler, old, scale=scale, seed=23)

    # Exercise nonzero cell sources, boundary loads, all cell blocks and face
    # balances. Assuming the previous residual is exactly zero must fail here.
    previous = assemble_state_residual(assembler, old)
    assert np.linalg.norm(previous) > 1e-6
    np.testing.assert_allclose(previous, first.residual, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        assemble_state_residual(assembler, candidate),
        previous + increment.residual,
        rtol=1e-12, atol=1e-13,
    )

    # Ensure both constitutive branches are actually exercised by the fixtures.
    if scale == 1e-5:
        assert all(state.alpha == 0.0 for state in candidate.material.material_states)
    else:
        assert any(state.alpha > 0.0 for state in old.material.material_states)
        assert increment.material_trial is not None
        assert np.linalg.norm(increment.material_trial.stress_correction) > 0.0


def test_assembly_preserves_state_and_reuses_stored_fields(
    assembler: ResidualAssembler, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, evaluation = _candidate(
        assembler, _zero_state(assembler), scale=0.02, seed=33,
    )
    before = state.copy()
    fields = ("x", "face_displacement", "side_traction", "neumann_values", "cell_source")
    for name in fields:
        getattr(state, name).setflags(write=False)
    state.material.epsilon.setflags(write=False)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Total-state balance must reuse stored fields without constitutive updates.")

    monkeypatch.setattr(assembler, "evaluate", forbidden)
    monkeypatch.setattr(assembler, "_evaluate_material_trial", forbidden)
    monkeypatch.setattr(MaterialPoint, "evaluate_from_with_data", forbidden)

    result = assemble_state_residual(assembler, state)
    np.testing.assert_allclose(result, evaluation.residual, rtol=1e-12, atol=1e-13)
    expected = result.copy()
    result[:] = 123.0  # The returned residual must not alias any state storage.
    np.testing.assert_array_equal(assemble_state_residual(assembler, state), expected)

    for name in fields:
        np.testing.assert_array_equal(getattr(state, name), getattr(before, name))
    assert state.material.dim == before.material.dim
    np.testing.assert_array_equal(state.material.epsilon, before.material.epsilon)
    for actual, saved in zip(
        state.material.material_states, before.material.material_states, strict=True,
    ):
        assert actual.alpha == saved.alpha
        for name in ("stress", "plastic_strain", "backstress"):
            np.testing.assert_array_equal(
                getattr(actual, name).to_numpy(), getattr(saved, name).to_numpy(),
            )


def test_single_dirichlet_cell_has_no_free_face_rows(dim: int) -> None:
    assembler = _assembler(dim, "dirichlet", single_cell=True)
    assert assembler.layout.free_faces.size == 0
    state, evaluation = _candidate(
        assembler, _zero_state(assembler), scale=0.02, seed=41,
    )
    residual = assemble_state_residual(assembler, state)
    assert residual.shape == (assembler.layout.cell_size,)
    np.testing.assert_allclose(residual, evaluation.residual, rtol=1e-12, atol=1e-13)


ATOL = (1e-11, 1e-11, 1e-11, 1e-11)
RTOL = (1e-10, 1e-10, 1e-10, 1e-10)


def _reference_loads(assembler: ResidualAssembler) -> LoadIncrement:
    layout = assembler.layout
    rng = np.random.default_rng(73)
    return LoadIncrement(
        1e-6 * rng.normal(size=(layout.dim, layout.dirichlet_faces.size)),
        1e-6 * rng.normal(size=(layout.dim, assembler.neumann_faces.size)),
        1e-6 * rng.normal(size=layout.cell_size),
    )


def _controller(
    assembler: ResidualAssembler, reference: LoadIncrement, *,
    committed: SolutionState | None = None, factor: float = 0.0,
    max_iterations: int = 20,
) -> LoadStepController:
    return LoadStepController(
        assembler, _zero_state(assembler) if committed is None else committed,
        reference.dirichlet_increment,
        reference_neumann=reference.neumann_increment,
        reference_cell_source=reference.cell_source_increment,
        initial_load_factor=factor, max_iterations=max_iterations,
        atol=ATOL, rtol=RTOL,
    )


def _elastic_reference(assembler: ResidualAssembler, reference: LoadIncrement) -> FloatArray:
    """Independent cell-only PorePy TPSA solve for unit reference loading."""
    layout = assembler.layout
    boundary = np.zeros((layout.dim, layout.num_faces))
    boundary[:, layout.dirichlet_faces] = reference.dirichlet_increment
    boundary[:, assembler.neumann_faces] = reference.neumann_increment
    return np.asarray(sps.linalg.spsolve(
        assembler.elastic.matrices.cell_matrix,
        assembler.elastic.boundary_rhs(boundary.ravel(order="F"))
        + reference.cell_source_increment,
    ))


def _assert_history(actual: MaterialHistory, expected: MaterialHistory) -> None:
    assert actual.dim == expected.dim
    for name in ("epsilon", "stress", "plastic_strain", "backstress", "alpha"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def _assert_state(actual: SolutionState, expected: SolutionState, *, exact: bool = True) -> None:
    compare = np.testing.assert_array_equal if exact else np.testing.assert_allclose
    options = {} if exact else {"rtol": 1e-9, "atol": 1e-11}
    for name in ("x", "face_displacement", "side_traction", "neumann_values", "cell_source"):
        compare(getattr(actual, name), getattr(expected, name), **options)
    assert actual.material.dim == expected.material.dim
    for name in ("epsilon", "stress", "plastic_strain", "backstress", "alpha"):
        compare(getattr(actual.material, name), getattr(expected.material, name), **options)


def _assert_balance(assembler: ResidualAssembler, record: LoadStepResult) -> None:
    residual = assemble_state_residual(assembler, record.state)
    np.testing.assert_array_equal(record.residual, residual)
    layout = assembler.layout
    for block, threshold in zip(
        (layout.u_slice, layout.r_slice, layout.p_slice, layout.face_slice),
        record.newton.thresholds, strict=True,
    ):
        assert np.linalg.norm(residual[block]) <= threshold


@pytest.mark.parametrize("boundary_type", ["dirichlet", "mixed"])
def test_elastic_schedule_matches_porepy_cell_matrix(dim: int, boundary_type: str) -> None:
    assembler = _assembler(dim, boundary_type)
    reference = _reference_loads(assembler)
    expected = _elastic_reference(assembler, reference)
    initial = _zero_state(assembler)
    controller = _controller(assembler, reference, committed=initial)
    factors = [.25, 1.0, 1.0, .5, 0.0, -.2]
    records = controller.run(factors)
    previous = initial
    for factor, record in zip(factors, records, strict=True):
        assert record.load_factor == factor
        np.testing.assert_allclose(record.state.x, factor * expected, rtol=1e-9, atol=1e-12)
        np.testing.assert_array_equal(
            record.state.x, previous.x + record.newton.z[assembler.layout.cell_slice],
        )
        np.testing.assert_array_equal(
            record.state.side_traction,
            previous.side_traction + record.newton.evaluation.traction_increment,
        )
        np.testing.assert_array_equal(
            record.state.face_displacement[:, assembler.layout.dirichlet_faces],
            factor * reference.dirichlet_increment,
        )
        np.testing.assert_array_equal(record.state.neumann_values, factor * reference.neumann_increment)
        np.testing.assert_array_equal(record.state.cell_source, factor * reference.cell_source_increment)
        np.testing.assert_array_equal(record.state.material.alpha, 0.0)
        trial = record.newton.evaluation.material_trial
        assert trial is not None
        _assert_history(record.state.material, trial.history)
        _assert_balance(assembler, record)
        previous = record.state
    assert records[2].newton.iterations == 0
    assert controller.load_factor == factors[-1]
    assert len(controller.steps) == len(records) == len(controller.attempts)
    assert all(attempt.accepted for attempt in controller.attempts)
    _assert_state(initial, _zero_state(assembler))


def test_plastic_path_matches_local_return_map_and_restart(dim: int) -> None:
    assembler = _assembler(dim, "dirichlet", uniform=True)
    layout = assembler.layout
    gradient = .01 * np.array([
        [1.0, .2, -.3], [-.4, -.3, .2], [.1, -.2, -.1],
    ])[:dim, :dim]
    faces = gradient @ assembler.operators.grid.face_centers[:dim]
    reference = LoadIncrement(
        faces[:, layout.dirichlet_faces], np.zeros((dim, 0)), np.zeros(layout.cell_size),
    )
    controller = _controller(assembler, reference)
    local = deepcopy(assembler.material_points[0])
    epsilon = np.zeros((3, 3))
    epsilon[:dim, :dim] = .5 * (gradient + gradient.T)
    previous_factor = 0.0
    factors = [.2, .7, 1.0, .95, .9, 1.1]
    records = []
    for factor in factors:
        record = controller.advance(factor)
        returned, _ = local.update(strain((factor - previous_factor) * epsilon))
        local.commit()
        for name in ("stress", "plastic_strain", "backstress"):
            expected = np.repeat(
                getattr(returned, name).to_numpy()[:, :, None], layout.num_cells, axis=2,
            )
            np.testing.assert_allclose(
                getattr(record.state.material, name), expected, rtol=1e-8, atol=1e-11,
            )
        np.testing.assert_allclose(record.state.material.alpha, returned.alpha, atol=1e-11)
        np.testing.assert_allclose(
            record.state.material.epsilon,
            np.repeat((factor * epsilon)[:, :, None], layout.num_cells, axis=2), atol=1e-11,
        )
        expected_traction = (
            returned.stress.to_numpy()[:dim, :dim] @ assembler.operators.side_normals
        ) * assembler.operators.side_areas
        np.testing.assert_allclose(record.state.side_traction, expected_traction, atol=1e-10)
        _assert_balance(assembler, record)
        records.append(record)
        previous_factor = factor
    assert np.all(records[2].state.material.alpha > 0)
    np.testing.assert_allclose(records[4].state.material.alpha, records[2].state.material.alpha, atol=1e-12)
    resumed = _controller(assembler, reference, committed=records[2].state, factor=1.0)
    for actual, expected in zip(resumed.run(factors[3:]), records[3:], strict=True):
        _assert_state(actual.state, expected.state, exact=False)


def test_reuses_jacobian_and_accepted_residual(dim: int) -> None:
    assembler = _assembler(dim, "mixed")
    reference = _reference_loads(assembler)
    with (
        patch.object(loading, "AnalyticalJacobian", wraps=loading.AnalyticalJacobian) as build,
        patch.object(loading, "assemble_state_residual", wraps=assemble_state_residual) as balance,
        patch.object(loading, "solve_newton", wraps=loading.solve_newton) as solve,
    ):
        controller = _controller(assembler, reference)
        records = controller.run([.25, .5, .5])
        assert build.call_count == 1
        assert balance.call_count == 1 + len(records)
        assert solve.call_count == len(records)
        jacobian = solve.call_args_list[0].kwargs["jacobian"]
        previous = _zero_state(assembler)
        for call, record in zip(solve.call_args_list, records, strict=True):
            assert call.kwargs["jacobian"] is jacobian
            np.testing.assert_array_equal(
                call.kwargs["committed_residual"], assemble_state_residual(assembler, previous),
            )
            previous = record.state
        assert records[-1].newton.iterations == 0


def test_nonzero_initial_imbalance_is_carried_into_newton(dim: int) -> None:
    assembler = _assembler(dim, "mixed")
    reference = _reference_loads(assembler)
    old, _ = _candidate(assembler, _zero_state(assembler), scale=1e-6, seed=71)
    saved = old.copy()
    assert np.linalg.norm(assemble_state_residual(assembler, old)) > 1e-6
    controller = _controller(assembler, reference, committed=old, factor=.25)
    record = controller.advance(.5)
    np.testing.assert_allclose(
        record.state.x, .5 * _elastic_reference(assembler, reference), rtol=1e-9, atol=1e-12,
    )
    _assert_balance(assembler, record)
    _assert_state(old, saved)


@pytest.mark.parametrize("failure", ["newton", "material", "acceptance", "missing_trial"])
def test_failed_step_preserves_state_cache_and_allows_retry(
    dim: int, failure: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembler = _assembler(dim, "mixed")
    reference = _reference_loads(assembler)
    controller = _controller(assembler, reference)
    accepted = controller.advance(.25)
    old = controller.state
    cached = controller._committed_residual.copy()
    original_solve = loading.solve_newton
    cause: type[Exception] = NewtonConvergenceError
    with monkeypatch.context() as context:
        if failure == "newton":
            context.setattr(controller, "_max_iterations", 0)
        elif failure == "material":
            def fail_material(*args: Any, **kwargs: Any) -> None:
                raise LocalUpdateError("Injected local failure")
            context.setattr(assembler, "_evaluate_material_trial", fail_material)
            cause = LocalUpdateError
        elif failure == "acceptance":
            def bad_balance(a: ResidualAssembler, state: SolutionState) -> FloatArray:
                result = assemble_state_residual(a, state)
                result[0] += 1.0
                return result
            context.setattr(loading, "assemble_state_residual", bad_balance)
        else:
            def missing_trial(*args: Any, **kwargs: Any) -> Any:
                result = original_solve(*args, **kwargs)
                return replace(result, evaluation=replace(result.evaluation, material_trial=None))
            context.setattr(loading, "solve_newton", missing_trial)
            cause = RuntimeError
        with pytest.raises(LoadStepError) as caught:
            controller.advance(.5)

    assert caught.value.step_index == 2 and caught.value.load_factor == .5
    assert controller.load_factor == .25 and len(controller.steps) == 1
    assert [attempt.accepted for attempt in controller.attempts] == [True, False]
    _assert_state(controller.state, old)
    np.testing.assert_array_equal(controller._committed_residual, cached)
    _assert_state(controller.steps[0].state, accepted.state)
    assert isinstance(caught.value.__cause__, cause)

    clean = _controller(assembler, reference, committed=old, factor=.25)
    _assert_state(controller.advance(.5).state, clean.advance(.5).state, exact=False)
    assert [attempt.accepted for attempt in controller.attempts] == [True, False, True]


def test_inputs_snapshots_and_records_do_not_alias_controller_state(dim: int) -> None:
    assembler = _assembler(dim, "mixed")
    reference = _reference_loads(assembler)
    expected = _elastic_reference(assembler, reference)
    initial = _zero_state(assembler)
    controller = _controller(assembler, reference, committed=initial)
    initial.x[:] = 123.0
    initial.material.stress[:] = 123.0
    for name in ("dirichlet_increment", "neumann_increment", "cell_source_increment"):
        getattr(reference, name)[:] = np.nan
    record = controller.advance(.5)
    saved = controller.state
    saved_residual = record.residual.copy()
    iterations = record.newton.iterations
    for snapshot in (record.state, controller.state, controller.steps[0].state):
        snapshot.x[:] = 456.0
        snapshot.material.alpha[:] = 456.0
        snapshot.material.stress[:] = 456.0
    record.residual[:] = np.nan
    record.newton.z[:] = np.nan
    record.newton.iterations = 999
    trial = record.newton.evaluation.material_trial
    assert trial is not None
    trial.history.epsilon[:] = np.nan
    _assert_state(controller.state, saved)
    np.testing.assert_array_equal(controller._committed_residual, saved_residual)
    assert controller.steps[0].newton.iterations == iterations
    assert np.all(np.isfinite(controller.steps[0].newton.z))
    np.testing.assert_allclose(controller.advance(1.0).state.x, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("failure_type", [NewtonDivergenceError, LocalUpdateError, ValueError])
def test_adaptive_retries_preserve_history_and_follow_reversed_targets(
    dim: int, failure_type: type[Exception], monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembler = _assembler(dim, "mixed")
    reference = _reference_loads(assembler)
    controller = _controller(assembler, reference)
    original = loading.solve_newton
    histories = []
    offsets = []
    failure = failure_type("Injected adaptive failure")
    def fail_once(*args: Any, **kwargs: Any) -> Any:
        histories.append(args[2].copy())
        offsets.append(kwargs["committed_residual"].copy())
        if len(histories) == 1:
            raise failure
        return original(*args, **kwargs)
    monkeypatch.setattr(loading, "solve_newton", fail_once)
    options = AdaptiveLoadOptions(
        initial_increment=.4, min_increment=.05, max_increment=.4, growth_factor=1.0,
    )
    if failure_type is ValueError:
        with pytest.raises(LoadStepError) as caught:
            controller.run_adaptive([.4], options=options)
        assert caught.value.__cause__ is failure
        assert len(histories) == 1 and controller.load_factor == 0.0
        assert controller.steps == ()
        _assert_state(controller.state, _zero_state(assembler))
        return

    records = controller.run_adaptive([.4, .4, -.2], options=options)
    assert [record.load_factor for record in records] == pytest.approx([.2, .4, 0.0, -.2])
    assert [attempt.accepted for attempt in controller.attempts] == [False, True, True, True, True]
    assert controller.attempts[0].error == "Injected adaptive failure"
    _assert_history(histories[0], histories[1])
    np.testing.assert_array_equal(offsets[0], offsets[1])
    assert controller.load_factor == -.2 and len(controller.steps) == len(records)
    np.testing.assert_allclose(
        controller.state.x, -.2 * _elastic_reference(assembler, reference), rtol=1e-9, atol=1e-12,
    )
    for record in records:
        _assert_balance(assembler, record)


def test_adaptive_exhaustion_preserves_last_accepted_state(dim: int) -> None:
    assembler = _assembler(dim, "mixed")
    reference = _reference_loads(assembler)
    controller = _controller(assembler, reference, max_iterations=0)
    with pytest.raises(AdaptiveLoadStepError) as caught:
        controller.run_adaptive([.5], options=AdaptiveLoadOptions(
            initial_increment=.25, min_increment=.125, max_increment=.25,
        ))
    assert isinstance(caught.value.__cause__, LoadStepError)
    assert isinstance(caught.value.__cause__.__cause__, NewtonConvergenceError)
    assert controller.load_factor == 0.0 and controller.steps == ()
    assert len(controller.attempts) == 2 and not any(a.accepted for a in controller.attempts)
    _assert_state(controller.state, _zero_state(assembler))
    np.testing.assert_array_equal(controller._committed_residual, 0.0)


def test_controller_supports_single_cell_without_free_face_unknowns(dim: int) -> None:
    assembler = _assembler(dim, "dirichlet", single_cell=True)
    reference = _reference_loads(assembler)
    controller = _controller(assembler, reference)
    record = controller.advance(1.0)
    assert assembler.layout.free_faces.size == 0
    assert record.newton.residual_norms[-1, 3] == 0.0
    np.testing.assert_allclose(record.state.x, _elastic_reference(assembler, reference), atol=1e-12)
    _assert_balance(assembler, record)


@pytest.mark.parametrize("invalid", [
    "state_shape", "material_layout", "reference_shape", "reference_nonfinite", "initial_factor",
])
def test_controller_validates_setup(dim: int, invalid: str) -> None:
    assembler = _assembler(dim, "mixed")
    reference = _reference_loads(assembler)
    state = _zero_state(assembler)
    factor = 0.0
    if invalid == "state_shape":
        state.x = np.zeros(1)
    elif invalid == "material_layout":
        state.material = MaterialHistory.zeros(_assembler(dim, "dirichlet", single_cell=True).layout)
    elif invalid == "reference_shape":
        reference = replace(reference, dirichlet_increment=reference.dirichlet_increment[:, :-1])
    elif invalid == "reference_nonfinite":
        reference.neumann_increment[0, 0] = np.nan
    else:
        factor = np.nan
    with pytest.raises(ValueError):
        _controller(assembler, reference, committed=state, factor=factor)


def test_controller_rejects_strain_inconsistent_with_faces(dim: int) -> None:
    assembler = _assembler(dim, "mixed")
    state = _zero_state(assembler)
    # Uniform stretch of the faces; stored strain stays zero.
    state.face_displacement[0] = 1e-3 * assembler.operators.grid.face_centers[0]
    with pytest.raises(ValueError, match=r"maximum absolute difference = 1\.000e-03"):
        _controller(assembler, _reference_loads(assembler), committed=state)


def test_invalid_target_does_not_start_an_attempt(dim: int) -> None:
    assembler = _assembler(dim, "mixed")
    controller = _controller(assembler, _reference_loads(assembler))
    accepted = controller.advance(.25)
    with patch.object(loading, "solve_newton", side_effect=AssertionError("Unexpected solve")):
        with pytest.raises(ValueError, match="finite"):
            controller.advance(np.nan)
    assert len(controller.attempts) == len(controller.steps) == 1
    assert controller.load_factor == .25
    _assert_state(controller.state, accepted.state)
