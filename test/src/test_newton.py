"""One coupled Newton increment: elastic references, plastic history and failures."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

import core.newton as newton_module
from core.jacobian import AnalyticalJacobian
from core.layout import FloatArray
from core.newton import NewtonResult, solve_newton
from core.residual import LoadIncrement, ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint, MaterialState
from material.scalar_hardening import HardeningParameters, isotropic_hardening_K, kinematic_hardening_H
from material.tensor import strain
from utils.newton_utils import NewtonConvergenceError, print_newton_convergence


ATOL = (1e-11, 1e-12, 1e-12, 1e-11)
RTOL = (1e-10, 1e-10, 1e-10, 1e-10)


def _assembler(
    dim: int, *, cells: int = 2, mixed: bool = False, heterogeneous: bool = False,
) -> ResidualAssembler:
    coordinates = (
        [np.array([0.0, 1.0])] * dim if cells == 1
        else [np.array([0.0, 0.3, 1.0]), np.array([0.0, 0.4, 1.3]), np.array([0.0, 0.2, 0.8])][:dim]
    )
    grid = pp.TensorGrid(coordinates[0], coordinates[1], coordinates[2] if dim == 3 else None)
    c, s = np.cos(0.37), np.sin(0.37)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if dim == 3:
        rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ rotation
    grid.nodes[:] = rotation @ grid.nodes
    grid.compute_geometry()
    boundary = grid.get_all_boundary_faces()
    bc = pp.BoundaryConditionVectorial(grid, boundary[::2] if mixed else boundary, "dir")
    points = []
    for cell in range(grid.num_cells):
        material = _materialProperty()
        material.young_modulus = 12.0 + (2.0 * cell if heterogeneous else 0.0)
        material.poisson_ratio = 0.25
        points.append(MaterialPoint(
            material, HardeningParameters(0.05, 0.05, 1.0, 0.4, 0.0),
            vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H,
        ))
    return ResidualAssembler(grid, bc, points)


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def assembler(request: pytest.FixtureRequest) -> ResidualAssembler:
    return _assembler(int(request.param))


def _gradient(dim: int, scale: float) -> FloatArray:
    return scale * np.array([[1.0, 0.2, -0.3], [-0.4, -0.3, 0.2], [0.1, -0.2, -0.1]])[:dim, :dim]


def _strain(gradient: FloatArray) -> FloatArray:
    result = np.zeros((3, 3))
    nd = gradient.shape[0]
    result[:nd, :nd] = 0.5 * (gradient + gradient.T)
    return result


def _affine(assembler: ResidualAssembler, gradient: FloatArray) -> tuple[FloatArray, LoadIncrement]:
    layout, grid = assembler.layout, assembler.operators.grid
    nd, nc = layout.dim, layout.num_cells
    material = assembler.material_points[0].material
    mu, lam = material.isotropic_shear_modulus, material.lame_parameter
    face_displacement = gradient @ grid.face_centers[:nd]
    rotation = (
        np.array([gradient[0, 1] - gradient[1, 0]]) if nd == 2 else np.array([
            gradient[1, 2] - gradient[2, 1], gradient[2, 0] - gradient[0, 2],
            gradient[0, 1] - gradient[1, 0],
        ])
    )
    cell_values = np.concatenate((
        (gradient @ grid.cell_centers[:nd]).ravel(order="F"),
        np.repeat((mu * rotation)[:, None], nc, axis=1).ravel(order="F"),
        np.full(nc, lam * np.trace(gradient)),
    ))
    sigma = mu * (gradient + gradient.T) + lam * np.trace(gradient) * np.eye(nd)
    nf = assembler.neumann_faces
    signs = np.asarray(grid.cell_faces[nf, :].sum(axis=1)).ravel()
    loads = LoadIncrement(
        face_displacement[:, layout.dirichlet_faces],
        sigma @ (grid.face_normals[:nd, nf] * signs), np.zeros(layout.cell_size),
    )
    return layout.pack(cell_values, face_displacement[:, layout.free_faces]), loads


def _assert_state(actual: MaterialState, expected: MaterialState, *, exact: bool = True) -> None:
    assert actual.alpha == pytest.approx(expected.alpha, rel=0, abs=0 if exact else 1e-10)
    for name in ("stress", "plastic_strain", "backstress"):
        a, b = getattr(actual, name).to_numpy(), getattr(expected, name).to_numpy()
        if exact:
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


def _assert_history(actual: MaterialHistory, expected: MaterialHistory) -> None:
    assert actual.dim == expected.dim
    np.testing.assert_array_equal(actual.epsilon, expected.epsilon)
    for a, b in zip(actual.material_states, expected.material_states, strict=True):
        _assert_state(a, b)


@pytest.mark.parametrize("dim,cells,mixed", [(2, 1, False), (3, 1, False), (2, 2, False),
                                             (3, 2, False), (2, 2, True), (3, 2, True)])
@pytest.mark.parametrize("supplied_jacobian", [False, True])
def test_elastic_affine_problem_converges_in_one_correction(
    dim: int, cells: int, mixed: bool, supplied_jacobian: bool,
) -> None:
    assembler = _assembler(dim, cells=cells, mixed=mixed)
    layout, operators = assembler.layout, assembler.operators
    gradient = _gradient(dim, 1e-4)
    expected, loads = _affine(assembler, gradient)
    initial = np.random.default_rng(61).normal(size=layout.size) * 1e-6
    initial_before = initial.copy()
    initial.setflags(write=False)
    old = MaterialHistory.zeros(layout)
    jacobian = AnalyticalJacobian(assembler) if supplied_jacobian else None
    result = solve_newton(assembler, loads, old, jacobian=jacobian, initial_z=initial,
                          max_iterations=1, atol=ATOL, rtol=RTOL)
    assert isinstance(result, NewtonResult)
    assert result.iterations == 1 and result.residual_norms.shape == (2, 4)
    np.testing.assert_allclose(result.z, expected, rtol=1e-10, atol=1e-12)
    sigma = assembler.material_points[0].material.elastic_tensor.double_contract(strain(_strain(gradient))).to_numpy()
    np.testing.assert_allclose(result.evaluation.traction_increment,
                               (sigma[:dim, :dim] @ operators.side_normals) * operators.side_areas, atol=1e-12)
    trial = result.evaluation.material_trial
    assert trial is not None
    np.testing.assert_allclose(trial.epsilon, np.repeat(_strain(gradient)[:, :, None], layout.num_cells, axis=2), atol=1e-12)
    assert all(state.alpha == 0 for state in trial.material_states)
    np.testing.assert_allclose(result.thresholds, np.array(ATOL) + np.array(RTOL) * result.residual_norms[0])
    assert np.all(result.residual_norms[-1] <= result.thresholds)
    np.testing.assert_array_equal(initial, initial_before)
    _assert_history(old, MaterialHistory.zeros(layout))
    assert all(point.trial is None for point in assembler.material_points)
    if cells == 1:
        np.testing.assert_array_equal(result.residual_norms[:, 3], 0.0)


@pytest.mark.parametrize("dim", [2, 3])
def test_heterogeneous_elastic_solution_matches_porepy_cell_matrix(dim: int) -> None:
    assembler = _assembler(dim, mixed=True, heterogeneous=True)
    grid, layout = assembler.operators.grid, assembler.layout
    rng = np.random.default_rng(62)
    boundary = np.zeros((dim, grid.num_faces))
    boundary[:, grid.get_all_boundary_faces()] = rng.normal(size=(dim, grid.get_all_boundary_faces().size)) * 1e-5
    sources = rng.normal(size=layout.cell_size) * 1e-5
    loads = LoadIncrement(boundary[:, layout.dirichlet_faces], boundary[:, assembler.neumann_faces], sources)
    expected = sps.linalg.spsolve(
        assembler.elastic.matrices.cell_matrix,
        assembler.elastic.boundary_rhs(boundary.ravel(order="F")) + sources,
    )
    result = solve_newton(assembler, loads, MaterialHistory.zeros(layout),
                          max_iterations=1, atol=ATOL, rtol=RTOL)
    assert result.iterations == 1
    np.testing.assert_allclose(result.z[layout.cell_slice], expected, rtol=1e-10, atol=1e-12)
    assert np.all(result.residual_norms[-1] <= result.thresholds)
    assert result.evaluation.material_trial is not None
    assert all(state.alpha == 0 for state in result.evaluation.material_trial.material_states)


@pytest.mark.parametrize("zero_load", [False, True])
def test_equilibrated_guess_needs_no_jacobian_or_linear_solve(
    assembler: ResidualAssembler, zero_load: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, loads = _affine(assembler, _gradient(assembler.layout.dim, 0.0 if zero_load else 1e-4))
    initial = expected.copy()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("An equilibrated initial guess must return without a correction.")

    monkeypatch.setattr(newton_module, "AnalyticalJacobian", forbidden)
    monkeypatch.setattr(newton_module, "_solve_linear_correction", forbidden)
    result = solve_newton(assembler, loads, MaterialHistory.zeros(assembler.layout),
                          initial_z=initial, max_iterations=0, atol=ATOL, rtol=RTOL)
    assert result.iterations == 0 and result.residual_norms.shape == (1, 4)
    np.testing.assert_array_equal(result.z, expected)
    assert not np.shares_memory(result.z, initial)
    result.z[:] = 1.0
    np.testing.assert_array_equal(initial, expected)


def test_plastic_solve_matches_local_return_map_and_reuses_fixed_history(
    assembler: ResidualAssembler, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    layout = assembler.layout
    expected, loads = _affine(assembler, _gradient(layout.dim, 0.01))
    old = MaterialHistory.zeros(layout)
    old_before, saved_loads = old.copy(), deepcopy(loads)
    real_evaluate = assembler.evaluate
    records: list[tuple[FloatArray, ResidualEvaluation]] = []
    histories: list[MaterialHistory] = []
    fixed_loads: list[LoadIncrement] = []
    created: list[AnalyticalJacobian] = []
    jacobian_evaluations: list[ResidualEvaluation] = []

    def evaluate(candidate: FloatArray, current_loads: LoadIncrement, history: MaterialHistory) -> ResidualEvaluation:
        assert history is not old and current_loads is not loads
        _assert_history(history, old_before)
        for name in ("dirichlet_increment", "neumann_increment", "cell_source_increment"):
            np.testing.assert_array_equal(getattr(current_loads, name), getattr(saved_loads, name))
            assert not np.shares_memory(getattr(current_loads, name), getattr(loads, name))
        histories.append(history)
        fixed_loads.append(current_loads)
        result = real_evaluate(candidate, current_loads, history)
        records.append((candidate.copy(), result))
        return result

    def build(current: ResidualAssembler) -> AnalyticalJacobian:
        jacobian = AnalyticalJacobian(current)
        original = jacobian.evaluate

        def matrix(history: MaterialHistory, evaluation: ResidualEvaluation) -> sps.csr_array:
            assert history is histories[-1] and evaluation is records[-1][1]
            jacobian_evaluations.append(evaluation)
            return original(history, evaluation)

        monkeypatch.setattr(jacobian, "evaluate", matrix)
        created.append(jacobian)
        return jacobian

    monkeypatch.setattr(assembler, "evaluate", evaluate)
    monkeypatch.setattr(newton_module, "AnalyticalJacobian", build)
    result = solve_newton(assembler, loads, old, atol=ATOL, rtol=RTOL)
    assert result.iterations > 1
    assert len(created) == 1 and len(jacobian_evaluations) == result.iterations
    assert len(records) == result.iterations + 1
    assert all(history is histories[0] for history in histories)
    assert all(values is fixed_loads[0] for values in fixed_loads)
    assert result.evaluation is records[-1][1]
    np.testing.assert_array_equal(result.z, records[-1][0])
    np.testing.assert_allclose(result.z, expected, rtol=1e-8, atol=1e-10)
    blocks = (layout.u_slice, layout.r_slice, layout.p_slice, layout.face_slice)
    norms = np.array([[np.linalg.norm(record.residual[block]) for block in blocks] for _, record in records])
    np.testing.assert_array_equal(result.residual_norms, norms)
    np.testing.assert_allclose(result.thresholds, np.array(ATOL) + np.array(RTOL) * norms[0])
    assert np.all(norms[-1] <= result.thresholds)

    point = assembler.material_points[0]
    local = MaterialPoint(point.material, point.parameters, point.model, point.K_law, point.H_law)
    reference, _ = local.update(strain(_strain(_gradient(layout.dim, 0.01))))
    assert reference.alpha > 0
    trial = result.evaluation.material_trial
    assert trial is not None
    for state in trial.material_states:
        _assert_state(state, reference, exact=False)
    ops = assembler.operators
    np.testing.assert_allclose(result.evaluation.traction_increment,
        (reference.stress.to_numpy()[:layout.dim, :layout.dim] @ ops.side_normals) * ops.side_areas, atol=1e-10)
    _assert_history(old, old_before)
    assert all(point.trial is None and point.committed.alpha == 0 for point in assembler.material_points)
    print_newton_convergence(result, extra_block="f")
    assert "ratio_f" in capsys.readouterr().out


def test_reused_jacobian_handles_plastic_reloading_and_elastic_unloading(assembler: ResidualAssembler) -> None:
    layout = assembler.layout
    jacobian = AnalyticalJacobian(assembler)
    old, reference = MaterialHistory.zeros(layout), MaterialState()
    point = assembler.material_points[0]
    for factor in (1.0, 0.3, -0.02):
        gradient = _gradient(layout.dim, factor * 0.01)
        expected, loads = _affine(assembler, gradient)
        saved = old.copy()
        result = solve_newton(assembler, loads, old, jacobian=jacobian, atol=ATOL, rtol=RTOL)
        local = MaterialPoint(point.material, point.parameters, point.model, point.K_law, point.H_law, committed=reference)
        returned, _ = local.update(strain(_strain(gradient)))
        if factor < 0:
            assert returned.alpha == reference.alpha
        else:
            assert returned.alpha > reference.alpha
        trial = result.evaluation.material_trial
        assert trial is not None
        np.testing.assert_allclose(result.z, expected, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(trial.epsilon, old.epsilon + _strain(gradient)[:, :, None], atol=1e-11)
        for state in trial.material_states:
            _assert_state(state, returned, exact=False)
        ops = assembler.operators
        delta_sigma = returned.stress.to_numpy() - reference.stress.to_numpy()
        np.testing.assert_allclose(result.evaluation.traction_increment,
            (delta_sigma[:layout.dim, :layout.dim] @ ops.side_normals) * ops.side_areas, atol=1e-10)
        _assert_history(old, saved)
        # Acceptance belongs to the caller, never to solve_newton.
        old = MaterialHistory(layout.dim, trial.epsilon, trial.material_states)
        reference = returned.copy()
    assert all(point.trial is None and point.committed.alpha == 0 for point in assembler.material_points)


@pytest.mark.parametrize("block", ["u_slice", "r_slice", "p_slice", "face_slice"])
def test_every_residual_block_must_converge(
    assembler: ResidualAssembler, block: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = assembler.layout
    z, loads = _affine(assembler, np.zeros((layout.dim, layout.dim)))
    old = MaterialHistory.zeros(layout)
    evaluation = assembler.evaluate(z, loads, old)
    evaluation.residual[getattr(layout, block).start] = 1.0

    def evaluate(candidate: FloatArray, values: LoadIncrement, history: MaterialHistory) -> ResidualEvaluation:
        return evaluation

    monkeypatch.setattr(assembler, "evaluate", evaluate)
    with pytest.raises(NewtonConvergenceError, match="after 0 corrections"):
        solve_newton(assembler, loads, old, max_iterations=0,
                     atol=(0.1, 0.1, 0.1, 0.1), rtol=(0.0, 0.0, 0.0, 0.0))


def test_jacobian_from_another_assembler_is_rejected(assembler: ResidualAssembler) -> None:
    _, loads = _affine(assembler, _gradient(assembler.layout.dim, 1e-4))
    other = AnalyticalJacobian(_assembler(assembler.layout.dim))
    with pytest.raises(ValueError, match="Jacobian must use the supplied residual assembler"):
        solve_newton(assembler, loads, MaterialHistory.zeros(assembler.layout), jacobian=other)


@pytest.mark.parametrize("failure", ["limit0", "limit1", "singular", "nonfinite", "shape", "tangent", "material"])
def test_failed_solve_preserves_input_history_loads_and_pending_material_trials(
    assembler: ResidualAssembler, failure: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = assembler.layout
    _, loads = _affine(assembler, _gradient(layout.dim, 0.01))
    old, initial = MaterialHistory.zeros(layout), np.zeros(layout.size)
    old_before, saved_loads, initial_before = old.copy(), deepcopy(loads), initial.copy()
    for values in (old.epsilon, initial, loads.dirichlet_increment,
                   loads.neumann_increment, loads.cell_source_increment):
        values.setflags(write=False)
    for point in assembler.material_points:
        point.update(strain(_strain(_gradient(layout.dim, 0.008))))
        point.commit()
        point.update(strain(_strain(_gradient(layout.dim, 0.001))))
    saved_points = deepcopy(assembler.material_points)
    slots = [(point.committed, point.trial) for point in assembler.material_points]
    jacobian = AnalyticalJacobian(assembler)
    max_iterations = 0 if failure == "limit0" else 1 if failure == "limit1" else 20
    error: type[Exception] = RuntimeError
    message = ""
    calls = 0

    if failure.startswith("limit"):
        error, message = NewtonConvergenceError, f"after {max_iterations} corrections"
    elif failure == "material":
        original = assembler.evaluate

        def failed_evaluation(candidate: FloatArray, values: LoadIncrement, history: MaterialHistory) -> ResidualEvaluation:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Injected material failure")
            return original(candidate, values, history)

        monkeypatch.setattr(assembler, "evaluate", failed_evaluation)
        message = "Injected material failure"
    elif failure == "tangent":
        def failed_tangent(history: MaterialHistory, evaluation: ResidualEvaluation) -> sps.csr_array:
            raise RuntimeError("Injected tangent failure")

        monkeypatch.setattr(jacobian, "evaluate", failed_tangent)
        message = "Injected tangent failure"
    else:
        matrix = jacobian.L.copy()
        if failure == "singular":
            matrix = sps.csr_array((layout.size, layout.size))
            message = "Newton linear solve failed at correction 1"
        elif failure == "nonfinite":
            matrix.data[0] = np.nan
            error, message = ValueError, "Jacobian must be finite"
        else:
            matrix = sps.eye_array(layout.size + 1, format="csr")
            error, message = ValueError, "Jacobian must be finite"

        def invalid_matrix(history: MaterialHistory, evaluation: ResidualEvaluation) -> sps.csr_array:
            return matrix

        monkeypatch.setattr(jacobian, "evaluate", invalid_matrix)

    with pytest.raises(error, match=message):
        solve_newton(assembler, loads, old, jacobian=jacobian, initial_z=initial,
                     max_iterations=max_iterations, atol=ATOL, rtol=RTOL)
    if failure == "material":
        assert calls == 2
    _assert_history(old, old_before)
    np.testing.assert_array_equal(initial, initial_before)
    for name in ("dirichlet_increment", "neumann_increment", "cell_source_increment"):
        np.testing.assert_array_equal(getattr(loads, name), getattr(saved_loads, name))
    for point, saved, (committed_slot, trial_slot) in zip(assembler.material_points, saved_points, slots, strict=True):
        assert point.committed is committed_slot and point.trial is trial_slot
        assert point.trial is not None and saved.trial is not None
        _assert_state(point.committed, saved.committed)
        _assert_state(point.trial, saved.trial)


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("invalid", ["shape", "nan", "overflowing_norm"])
def test_invalid_residual_cannot_be_accepted(
    assembler: ResidualAssembler, stage: int, invalid: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = assembler.layout
    _, loads = _affine(assembler, _gradient(layout.dim, 1e-4))
    old = MaterialHistory.zeros(layout)
    original, calls = assembler.evaluate, 0

    def evaluate(candidate: FloatArray, values: LoadIncrement, history: MaterialHistory) -> ResidualEvaluation:
        nonlocal calls
        result = original(candidate, values, history)
        if calls == stage:
            residual = np.zeros(layout.size + 1 if invalid == "shape" else layout.size)
            if invalid != "shape":
                residual[0] = np.nan if invalid == "nan" else 1e200
            result = replace(result, residual=residual)
        calls += 1
        return result

    monkeypatch.setattr(assembler, "evaluate", evaluate)
    with np.errstate(over="ignore"), pytest.raises(RuntimeError, match="residual"):
        solve_newton(assembler, loads, old, atol=ATOL, rtol=RTOL)
    assert calls == stage + 1
    _assert_history(old, MaterialHistory.zeros(layout))


@pytest.mark.parametrize("invalid", ["shape", "complex", "nan"])
def test_initial_guess_is_validated_before_float_conversion(
    assembler: ResidualAssembler, invalid: str,
) -> None:
    layout = assembler.layout
    _, loads = _affine(assembler, _gradient(layout.dim, 1e-4))
    if invalid == "shape":
        initial = np.zeros(layout.size + 1)
    elif invalid == "complex":
        initial = cast(FloatArray, np.full(layout.size, 1j))
    else:
        initial = np.full(layout.size, np.nan)
    with pytest.raises(ValueError, match="z"):
        solve_newton(assembler, loads, MaterialHistory.zeros(layout), initial_z=initial)


@pytest.mark.parametrize("options", [
    {"max_iterations": -1}, {"max_iterations": True}, {"max_iterations": 1.5},
    {"atol": (1e-8, 1e-8, 1e-8)}, {"atol": (0.0, 0.0, 0.0, -1.0)},
    {"rtol": (0.0, 0.0, 0.0, np.nan)}, {"rtol": (0.0, 0.0, 0.0, 1.0)},
])
def test_invalid_newton_options_are_rejected_before_evaluation(
    assembler: ResidualAssembler, options: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loads = _affine(assembler, _gradient(assembler.layout.dim, 1e-4))

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid options must fail before material evaluation.")

    monkeypatch.setattr(assembler, "evaluate", forbidden)
    with pytest.raises(ValueError):
        solve_newton(assembler, loads, MaterialHistory.zeros(assembler.layout), **options)
