"""One-load coupled Newton control, elastic recovery, and failure isolation."""

from copy import deepcopy
from typing import TypeAlias

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray

from coupling import newton
from coupling.newton import NewtonConvergenceError, solve_newton
from coupling.plane_strain import PlaneStrainTpsa
from coupling.residual import TpsaOperators, TpsaResidual, TpsaState, evaluate_global_residual
from coupling.transfer import CellToFaceTransfer
from tensor import strain

FloatArray: TypeAlias = NDArray[np.float64]
Setup: TypeAlias = tuple[PlaneStrainTpsa, TpsaOperators, CellToFaceTransfer, TpsaState]


@pytest.fixture
def setup() -> Setup:
    case = PlaneStrainTpsa(cells_per_axis=3)
    case.prepare_simulation()
    return case, TpsaOperators(case), CellToFaceTransfer(case.grid), TpsaState.zeros(case.grid)


def assert_same_state(actual: TpsaState, expected: TpsaState) -> None:
    for name in ("x", "bc_values", "epsilon", "traction"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for a, b in zip(actual.material_states, expected.material_states, strict=True):
        assert a.alpha == b.alpha
        for name in ("stress", "plastic_strain", "backstress"):
            np.testing.assert_array_equal(getattr(a, name).to_numpy(), getattr(b, name).to_numpy())


def forbidden_jacobian(x: FloatArray, evaluation: TpsaResidual) -> sps.csr_array:
    raise AssertionError("No Jacobian should be requested.")


@pytest.mark.parametrize("cells_per_axis", [1, 3])
@pytest.mark.parametrize("dense", [False, True], ids=["sparse", "dense"])
def test_elastic_problem_converges_in_one_coupled_correction(cells_per_axis: int, dense: bool) -> None:
    case = PlaneStrainTpsa(
        cells_per_axis=cells_per_axis,
        displacement_gradient=np.array([[1e-5, 2e-5], [-3e-5, 4e-5]]),
    )
    expected = case.solve().copy()
    operators, transfer = TpsaOperators(case), CellToFaceTransfer(case.grid)
    committed = TpsaState.zeros(case.grid)
    rng = np.random.default_rng(3)
    initial = rng.normal(size=expected.size)
    initial[:2 * case.grid.num_cells] *= 1e-6
    initial[2 * case.grid.num_cells:] *= 1e5
    saved = initial.copy()
    calls: list[FloatArray] = []
    matrix = operators.A.toarray() if dense else operators.A
    matrix_before = matrix.copy()

    def jacobian(x: FloatArray, evaluation: TpsaResidual) -> newton.JacobianMatrix:
        assert not x.flags.writeable
        np.testing.assert_array_equal(x, evaluation.trial.x)
        np.testing.assert_array_equal(evaluation.stress_correction, 0.0)
        calls.append(x.copy())
        return matrix

    result = solve_newton(
        operators, transfer, case.material_points, committed, case.bc_values,
        jacobian, initial_x=initial, max_iterations=1,
    )
    assert result.iterations == len(calls) == 1
    assert result.residual_norms.shape == (2, 3)
    np.testing.assert_allclose(result.trial.x, expected, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(
        result.trial.x[:2 * case.grid.num_cells], expected[:2 * case.grid.num_cells], atol=1e-14,
    )
    np.testing.assert_allclose(
        result.trial.traction, case.reference_stress()[:2, :2] @ case.grid.face_normals[:2], atol=1e-5,
    )
    np.testing.assert_allclose(
        result.thresholds, np.array([1e-5, 1e-12, 1e-12]) + 1e-8 * result.residual_norms[0],
    )
    assert np.all(result.residual_norms[-1] <= result.thresholds)
    np.testing.assert_array_equal(initial, saved)
    np.testing.assert_array_equal(matrix if dense else matrix.toarray(), matrix_before if dense else matrix_before.toarray())
    assert_same_state(committed, TpsaState.zeros(case.grid))
    np.testing.assert_array_equal(case.x, expected)
    assert all(point.trial is None for point in case.material_points)


def test_equilibrated_initial_guess_needs_no_jacobian_or_correction(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    expected = np.asarray(sps.linalg.spsolve(operators.A, operators.assemble_rhs(case.bc_values)))
    result = solve_newton(
        operators, transfer, case.material_points, committed, case.bc_values,
        forbidden_jacobian, initial_x=expected, max_iterations=0,
    )
    assert result.iterations == 0
    assert result.residual_norms.shape == (1, 3)
    np.testing.assert_array_equal(result.trial.x, expected)
    assert not np.shares_memory(result.trial.x, expected)


def test_all_zero_load_converges_with_zero_reference_norms(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    result = solve_newton(
        operators, transfer, case.material_points, committed, np.zeros_like(case.bc_values),
        forbidden_jacobian, max_iterations=0, atol=(0.0, 0.0, 0.0),
    )
    np.testing.assert_array_equal(result.residual_norms, np.zeros((1, 3)))
    np.testing.assert_array_equal(result.thresholds, np.zeros(3))
    assert_same_state(result.trial, committed)
    assert result.trial is not committed


@pytest.mark.parametrize("field", ["r", "p"])
def test_convergence_checks_each_auxiliary_block(setup: Setup, field: str) -> None:
    case, operators, transfer, committed = setup
    nc = case.grid.num_cells
    expected = np.asarray(sps.linalg.spsolve(operators.A, operators.assemble_rhs(case.bc_values)))
    initial = expected.copy()
    initial[(2 if field == "r" else 3) * nc] += 1e6
    # Momentum is deliberately given a loose tolerance; the auxiliary imbalance must still be solved.
    result = solve_newton(
        operators, transfer, case.material_points, committed, case.bc_values,
        lambda x, evaluation: operators.A, initial_x=initial, max_iterations=1,
        atol=(1e20, 1e-14, 1e-14), rtol=(0.0, 0.0, 0.0),
    )
    assert result.iterations == 1
    block = 1 if field == "r" else 2
    assert result.residual_norms[0, block] > result.thresholds[block]
    assert result.residual_norms[-1, block] <= result.thresholds[block]
    np.testing.assert_allclose(result.trial.x, expected, rtol=1e-10, atol=1e-6)


def test_nonzero_integrated_sources_are_fixed_during_newton(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    nc = case.grid.num_cells
    sources = np.zeros(4 * nc)
    sources[:2 * nc] = (np.array([[2e5], [-1e5]]) * case.grid.cell_volumes).ravel(order="F")
    sources[2 * nc:] = 1e-6
    expected = np.asarray(sps.linalg.spsolve(operators.A, operators.assemble_rhs(case.bc_values, sources=sources)))
    saved_sources, saved_boundary = sources.copy(), case.bc_values.copy()
    sources.flags.writeable = case.bc_values.flags.writeable = False
    result = solve_newton(
        operators, transfer, case.material_points, committed, case.bc_values,
        lambda x, evaluation: operators.A, sources=sources, max_iterations=1,
    )
    assert result.iterations == 1
    np.testing.assert_allclose(result.trial.x, expected, rtol=1e-10, atol=1e-6)
    np.testing.assert_array_equal(sources, saved_sources)
    np.testing.assert_array_equal(case.bc_values, saved_boundary)
    assert case.x is None


def test_repeated_iterations_use_fixed_plastic_history_and_refresh_callback(setup: Setup) -> None:
    case, operators, transfer, virgin = setup
    nc = case.grid.num_cells
    # Homogeneous plastic preloading supplies an independent compatible committed state.
    gradient = np.array([[0.004, 0.0], [0.0, 0.0]])
    u = (gradient @ case.grid.cell_centers[:2]).ravel(order="F")
    x_n = np.concatenate((u, np.zeros(nc), np.full(nc, case.material.lame_parameter * 0.004)))
    old_boundary = np.zeros_like(case.bc_values)
    bf = operators.boundary_faces
    old_boundary[:, bf] = gradient @ case.grid.face_centers[:2, bf]
    committed = evaluate_global_residual(
        operators, transfer, case.material_points, virgin, x_n, old_boundary,
    ).trial
    saved = committed.copy()
    assert committed.material_states[0].alpha > 0.0
    target = 0.95 * old_boundary
    # Keep every iterate inside elastic unloading despite the boundary/guess mismatch.
    initial = 0.949 * x_n
    calls: list[TpsaResidual] = []
    for point in case.material_points:
        point.update(strain(np.diag([1e-5, 0.0, 0.0])))
    saved_points = deepcopy(case.material_points)

    def jacobian(x: FloatArray, evaluation: TpsaResidual) -> sps.csr_array:
        np.testing.assert_array_equal(evaluation.trial.bc_values, target)
        np.testing.assert_array_equal(evaluation.stress_correction, 0.0)
        for current, old in zip(evaluation.trial.material_states, committed.material_states, strict=True):
            assert current.alpha == old.alpha
        assert_same_state(committed, saved)
        calls.append(deepcopy(evaluation))
        # Deliberately inexact first matrix exercises a second callback/residual evaluation.
        return (2 if len(calls) == 1 else 1) * operators.A

    result = solve_newton(
        operators, transfer, case.material_points, committed, target, jacobian,
        initial_x=initial, max_iterations=2,
    )
    assert result.iterations == len(calls) == 2
    assert not np.array_equal(calls[0].trial.x, calls[1].trial.x)
    np.testing.assert_allclose(result.trial.x, 0.95 * x_n, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(result.residual_norms[1], 0.5 * result.residual_norms[0], rtol=1e-7, atol=1e-6)
    assert_same_state(committed, saved)
    for point, before in zip(case.material_points, saved_points, strict=True):
        assert point.committed.alpha == before.committed.alpha == 0.0
        assert point.trial is not None and before.trial is not None
        np.testing.assert_array_equal(point.trial.stress.to_numpy(), before.trial.stress.to_numpy())
    result.trial.x[:] = 0.0
    result.trial.material_states[0].alpha = -1.0
    assert_same_state(committed, saved)


@pytest.mark.parametrize("max_iterations", [0, 2])
def test_iteration_limit_raises_and_preserves_state(setup: Setup, max_iterations: int) -> None:
    case, operators, transfer, committed = setup
    saved = committed.copy()
    calls: list[int] = []

    def slow_jacobian(x: FloatArray, evaluation: TpsaResidual) -> sps.csr_array:
        calls.append(1)
        return 2 * operators.A

    with pytest.raises(NewtonConvergenceError, match=f"after {max_iterations} corrections"):
        solve_newton(
            operators, transfer, case.material_points, committed, case.bc_values,
            slow_jacobian, max_iterations=max_iterations,
        )
    assert len(calls) == max_iterations
    assert_same_state(committed, saved)
    assert all(point.trial is None and point.committed.alpha == 0.0 for point in case.material_points)


@pytest.mark.parametrize("defect", ["shape", "nonfinite", "singular"])
def test_invalid_or_singular_jacobian_cannot_accept_a_candidate(setup: Setup, defect: str) -> None:
    case, operators, transfer, committed = setup
    saved = committed.copy()
    matrix = operators.A.copy()
    if defect == "shape":
        matrix = matrix[:-1, :-1]
    elif defect == "nonfinite":
        matrix.data[0] = np.nan
    else:
        matrix.data[:] = 0.0
    error = RuntimeError if defect == "singular" else ValueError
    match = "linear solve failed" if defect == "singular" else "Jacobian.*finite.*shape"
    with pytest.raises(error, match=match):
        solve_newton(
            operators, transfer, case.material_points, committed, case.bc_values,
            lambda x, evaluation: matrix,
        )
    assert_same_state(committed, saved)
    assert all(point.trial is None for point in case.material_points)


def test_callback_failure_preserves_committed_history(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    saved = committed.copy()

    def failing_jacobian(x: FloatArray, evaluation: TpsaResidual) -> sps.csr_array:
        raise RuntimeError("Injected Jacobian failure")

    with pytest.raises(RuntimeError, match="Injected Jacobian failure"):
        solve_newton(operators, transfer, case.material_points, committed, case.bc_values, failing_jacobian)
    assert_same_state(committed, saved)
    assert all(point.trial is None for point in case.material_points)


def test_material_failure_after_a_correction_preserves_history(
    setup: Setup, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, operators, transfer, committed = setup
    saved = committed.copy()
    original = evaluate_global_residual
    calls = 0

    def fail_on_second_evaluation(*args: object, **kwargs: object) -> TpsaResidual:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Injected material evaluation failure")
        # Use concrete fixture inputs for the first genuine material evaluation.
        return original(operators, transfer, case.material_points, committed, committed.x, case.bc_values)

    monkeypatch.setattr(newton, "evaluate_global_residual", fail_on_second_evaluation)
    with pytest.raises(RuntimeError, match="Injected material evaluation failure"):
        solve_newton(
            operators, transfer, case.material_points, committed, case.bc_values,
            lambda x, evaluation: operators.A,
        )
    assert calls == 2
    assert_same_state(committed, saved)
    assert all(point.trial is None for point in case.material_points)


def test_nonfinite_linear_correction_is_rejected(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    case, operators, transfer, committed = setup
    saved = committed.copy()

    class BadFactor:
        def solve(self, rhs: FloatArray) -> FloatArray:
            return np.full_like(rhs, np.inf)

    monkeypatch.setattr(sps.linalg, "splu", lambda matrix: BadFactor())
    with pytest.raises(RuntimeError, match="non-finite.*correction"):
        solve_newton(
            operators, transfer, case.material_points, committed, case.bc_values,
            lambda x, evaluation: operators.A,
        )
    assert_same_state(committed, saved)


@pytest.mark.parametrize("norm_block", [0, 1, 2])
def test_nonfinite_residual_is_never_accepted(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, norm_block: int,
) -> None:
    case, operators, transfer, committed = setup
    evaluation = evaluate_global_residual(
        operators, transfer, case.material_points, committed, committed.x, case.bc_values,
    )
    index = [0, 2 * case.grid.num_cells, 3 * case.grid.num_cells][norm_block]
    evaluation.residual[index] = np.inf
    monkeypatch.setattr(newton, "evaluate_global_residual", lambda *args, **kwargs: evaluation)
    with pytest.raises(RuntimeError, match="non-finite.*residual"):
        solve_newton(
            operators, transfer, case.material_points, committed, case.bc_values,
            forbidden_jacobian,
        )


@pytest.mark.parametrize("iterations", [-1, True, 1.5])
def test_invalid_iteration_count_is_rejected(setup: Setup, iterations: int) -> None:
    case, operators, transfer, committed = setup
    with pytest.raises(ValueError, match="max_iterations"):
        solve_newton(
            operators, transfer, case.material_points, committed, case.bc_values,
            forbidden_jacobian, max_iterations=iterations,
        )


@pytest.mark.parametrize("values", [(-1.0, 0.0, 0.0), (np.nan, 0.0, 0.0), (np.inf, 0.0, 0.0)])
@pytest.mark.parametrize("name", ["atol", "rtol"])
def test_invalid_tolerances_are_rejected(setup: Setup, values: newton.BlockTolerances, name: str) -> None:
    case, operators, transfer, committed = setup
    with pytest.raises(ValueError, match=name):
        if name == "atol":
            solve_newton(operators, transfer, case.material_points, committed, case.bc_values, forbidden_jacobian, atol=values)
        else:
            solve_newton(operators, transfer, case.material_points, committed, case.bc_values, forbidden_jacobian, rtol=values)


def test_relative_tolerance_cannot_trivially_accept_initial_imbalance(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    with pytest.raises(ValueError, match="rtol.*smaller than one"):
        solve_newton(
            operators, transfer, case.material_points, committed, case.bc_values,
            forbidden_jacobian, rtol=(1.0, 0.0, 0.0),
        )


def test_unused_boundary_entries_are_ignored(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    boundary = case.bc_values.copy()
    interior = np.setdiff1d(np.arange(case.grid.num_faces), operators.boundary_faces)
    boundary[:, interior] = np.nan
    result = solve_newton(
        operators, transfer, case.material_points, committed, boundary,
        lambda x, evaluation: operators.A, max_iterations=1,
    )
    np.testing.assert_array_equal(result.trial.bc_values[:, interior], 0.0)
    assert np.all(np.isnan(boundary[:, interior]))


@pytest.mark.parametrize("order", [1, 2], ids=["linear-decay", "quadratic-decay"])
def test_observed_rates_recover_known_order_with_different_block_units(order: int) -> None:
    exponents = np.arange(1, 6) if order == 1 else 2 ** np.arange(5)
    values = 0.5 ** exponents
    norms = values[:, None] * np.array([1e12, 1.0, 1e-12])
    saved = norms.copy()
    norms.flags.writeable = False
    ratios, orders = newton.newton_convergence_rates(norms)
    assert ratios.shape == orders.shape == norms.shape
    assert np.all(np.isnan(ratios[0])) and np.all(np.isnan(orders[:2]))
    expected_ratios = np.full(4, 0.5) if order == 1 else values[:-1]
    np.testing.assert_allclose(ratios[1:], np.repeat(expected_ratios[:, None], 3, axis=1))
    np.testing.assert_allclose(orders[2:], order, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(norms, saved)


def test_rates_handle_zero_residuals_stagnation_and_numerical_noise() -> None:
    norms = np.array([[1.0, 0.0, 1.0], [0.1, 1.0, 1.0], [1e-18, 0.1, 1.0], [0.0, 0.01, 1.0]])
    with np.errstate(all="raise"):
        ratios, orders = newton.newton_convergence_rates(norms)
    assert np.all(np.isnan(orders[:, [0, 2]]))  # Noise floor or stagnation.
    assert np.isnan(ratios[1, 1])  # Initial zero: ratio is undefined.
    assert np.isnan(orders[2, 1])
    assert orders[3, 1] == pytest.approx(1.0)  # Three later decreasing values suffice.
    assert ratios[-1, 0] == 0.0  # Exact zero is valid as a reduction, not a log order.
    np.testing.assert_array_equal(ratios[1:, 2], 1.0)


def test_orders_are_unavailable_across_a_residual_increase() -> None:
    norms = np.repeat(np.array([1.0, 0.1, 0.2, 0.02, 0.002])[:, None], 3, axis=1)
    ratios, orders = newton.newton_convergence_rates(norms)
    np.testing.assert_array_equal(ratios[2], 2.0)
    assert np.all(np.isnan(orders[:4]))
    np.testing.assert_allclose(orders[4], 1.0)


@pytest.mark.parametrize("evaluations", [1, 2])
def test_short_convergence_histories_have_no_observed_order(evaluations: int) -> None:
    norms = np.zeros((evaluations, 3))
    ratios, orders = newton.newton_convergence_rates(norms)
    assert ratios.shape == orders.shape == norms.shape
    assert np.all(np.isnan(ratios)) and np.all(np.isnan(orders))


def test_print_convergence_for_an_initially_equilibrated_state(
    setup: Setup, capsys: pytest.CaptureFixture[str],
) -> None:
    case, operators, transfer, committed = setup
    result = solve_newton(
        operators, transfer, case.material_points, committed, committed.bc_values,
        forbidden_jacobian, max_iterations=0,
    )
    newton.print_newton_convergence(result)
    output = capsys.readouterr().out
    assert "ratio_u" in output and "order_u" in output
    assert output.splitlines()[-1].split() == ["0", "0.000e+00", "0.000e+00", "0.000e+00", "--", "--"]
    assert "nan" not in output.lower() and "inf" not in output.lower()
