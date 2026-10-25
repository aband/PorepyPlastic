"""Nested Newton equilibrium, independent material trials, and failure isolation."""

from copy import deepcopy
from dataclasses import replace
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from utils.newton_utils import JacobianMatrix, NewtonConvergenceError, newton_convergence_rates
from alternatives.plane_strain import PlaneStrainTpsa
from utils.state import TpsaState
from material.material_state import MaterialPoint
from prototype import newton
from prototype.jacobian import RetainedAnalyticalJacobian, RetainedFiniteDifferenceJacobian
from prototype.newton import BacktrackingOptions, solve_newton
from prototype.operators import RetainedFaceOperators
from prototype.residual import RetainedFaceResidual, evaluate_global_residual
from material.tensor import strain

FloatArray: TypeAlias = NDArray[np.float64]
Setup: TypeAlias = tuple[PlaneStrainTpsa, RetainedFaceOperators, TpsaState]


def assert_same_state(actual: TpsaState, expected: TpsaState) -> None:
    for name in ("x", "bc_values", "epsilon", "traction"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for a, b in zip(actual.material_states, expected.material_states, strict=True):
        assert a.alpha == b.alpha
        for name in ("stress", "plastic_strain", "backstress"):
            np.testing.assert_array_equal(getattr(a, name).to_numpy(), getattr(b, name).to_numpy())


def make_setup(cells: int = 3) -> Setup:
    case = PlaneStrainTpsa(cells_per_axis=cells)
    case.prepare_simulation()
    ops = RetainedFaceOperators(case.grid, case.bc,
        mu=case.material.isotropic_shear_modulus, lam=case.material.lame_parameter)
    return case, ops, TpsaState.zeros(case.grid)


def affine(ops: RetainedFaceOperators, gradient: FloatArray) -> tuple[FloatArray, FloatArray]:
    assert ops.mu is not None and ops.lam is not None
    d = gradient @ ops.grid.face_centers[:2]
    cell = np.r_[(gradient @ ops.grid.cell_centers[:2]).ravel("F"),
                 ops.mu * (gradient[0, 1] - gradient[1, 0]), ops.lam * np.trace(gradient)]
    return ops.pack(cell, d[:, ops.free_faces]), d


def forbidden(z: FloatArray, evaluation: RetainedFaceResidual) -> JacobianMatrix:
    raise AssertionError("No correction should be needed")


@pytest.mark.parametrize("cells", [1, 3])
@pytest.mark.parametrize("dense", [False, True])
def test_elastic_affine_solution_and_reactions_in_one_correction(cells: int, dense: bool) -> None:
    case, ops, old = make_setup(cells)
    gradient = np.array([[1e-5, 2e-5], [-3e-5, 4e-5]])
    expected, boundary = affine(ops, gradient)
    matrix = RetainedAnalyticalJacobian(ops, case.material_points, old).L
    initial = expected * np.random.default_rng(7).normal(size=ops.num_dofs)
    saved_initial, saved_matrix = initial.copy(), matrix.toarray()

    def callback(z: FloatArray, evaluation: RetainedFaceResidual) -> JacobianMatrix:
        assert not z.flags.writeable
        np.testing.assert_array_equal(z, evaluation.z)
        return matrix.toarray() if dense else matrix

    result = solve_newton(ops, case.material_points, old, boundary, callback,
                          initial_z=initial, max_iterations=1)
    assert result.iterations == 1 and result.residual_evaluations == 2
    assert result.residual_norms.shape == (2, 4)
    np.testing.assert_array_equal(result.step_lengths, [1.])
    np.testing.assert_allclose(result.z, expected, rtol=1e-11, atol=1e-7)
    np.testing.assert_allclose(result.z[ops.u_slice], expected[ops.u_slice], atol=1e-17)
    eps = np.zeros((3, 3))
    eps[:2, :2] = 0.5 * (gradient + gradient.T)
    stress = np.einsum("ijkl,kl->ij", case.material.elastic_tensor.to_numpy(), eps)
    np.testing.assert_allclose(result.evaluation.traction_increment,
        stress[:2, :2] @ ops.side_normals * ops.side_areas, atol=1e-7)
    np.testing.assert_allclose(result.thresholds,
        np.array([1e-5, 1e-12, 1e-12, 1e-5]) + 1e-8 * result.residual_norms[0])
    assert np.all(result.residual_norms[-1] <= result.thresholds)
    assert newton_convergence_rates(result.residual_norms)[0].shape == (2, 4)
    np.testing.assert_array_equal(initial, saved_initial)
    np.testing.assert_array_equal(matrix.toarray(), saved_matrix)
    assert_same_state(old, TpsaState.zeros(case.grid))
    assert all(p.trial is None for p in case.material_points)
    if cells == 1:
        np.testing.assert_array_equal(result.residual_norms[:, 3], 0)


@pytest.mark.parametrize("numerical", [False, True])
@pytest.mark.parametrize("search", [None, BacktrackingOptions()])
def test_plastic_affine_solution_matches_independent_return_map(
    numerical: bool, search: BacktrackingOptions | None,
) -> None:
    case, ops, old = make_setup()
    gradient = np.array([[.004, .0006], [-.0002, -.0003]])
    expected, boundary = affine(ops, gradient)
    callback = RetainedFiniteDifferenceJacobian(ops, case.material_points, old) if numerical else None
    result = solve_newton(ops, case.material_points, old, boundary, callback, line_search=search)
    assert result.iterations > 1
    np.testing.assert_allclose(result.z, expected, rtol=2e-8, atol=1e-3)
    np.testing.assert_allclose(result.z[ops.u_slice], expected[ops.u_slice], atol=1e-11)
    eps = np.zeros((3, 3))
    eps[:2, :2] = 0.5 * (gradient + gradient.T)
    local = deepcopy(case.material_points[0])
    reference, _ = local.update(strain(eps))
    for state in result.evaluation.material_trial.material_states:
        assert state.alpha == pytest.approx(reference.alpha, rel=1e-8)
        np.testing.assert_allclose(state.stress.to_numpy(), reference.stress.to_numpy(), rtol=1e-8, atol=.5)
    assert np.all(result.residual_norms[-1] <= result.thresholds)
    assert_same_state(old, TpsaState.zeros(case.grid))


@pytest.mark.parametrize("factor", [1.0, 0.99, 1.2])
def test_hold_unload_and_reload_start_from_fixed_preloaded_history(factor: float) -> None:
    case, ops, old = make_setup()
    gradient = np.diag([.004, 0.])
    _, boundary = affine(ops, gradient)
    # Use absolute equilibrium accuracy so a subsequent hold needs no cleanup
    # of imbalance accepted under the previous step's relative tolerance.
    preload = solve_newton(ops, case.material_points, old, boundary, rtol=(0., 0., 0., 0.))
    trial = preload.evaluation
    # Only these committed fields enter the incremental residual. Accumulation
    # of total one-sided tractions belongs to the future load controller.
    old = TpsaState(preload.z[ops.cell_slice], trial.bc_values, trial.material_trial.epsilon,
                    trial.material_trial.material_states, np.zeros((2, ops.num_faces)))
    saved = old.copy()
    result = solve_newton(ops, case.material_points, old, factor * boundary,
                         committed_residual=trial.residual)
    expected, _ = affine(ops, (factor - 1) * gradient)
    np.testing.assert_allclose(result.z, expected, rtol=2e-7, atol=.02)
    np.testing.assert_allclose(result.z[ops.u_slice], expected[ops.u_slice], atol=2e-11)
    if factor == 1:
        assert result.iterations == 0
        np.testing.assert_array_equal(result.z, 0)
    for before, after in zip(old.material_states, result.evaluation.material_trial.material_states, strict=True):
        if factor <= 1:
            assert after.alpha == pytest.approx(before.alpha, abs=1e-12)
        else:
            assert after.alpha > before.alpha
    assert_same_state(old, saved)


def test_mixed_yielding_cells_converge_with_independent_face_balance() -> None:
    case, ops, old = make_setup()
    for cell, point in enumerate(case.material_points):
        value = 100e6 if cell % 2 else 2e9
        point.parameters = replace(point.parameters, sigma_y=value, sigma_u=value)
    _, boundary = affine(ops, np.diag([.004, 0.]))
    result = solve_newton(ops, case.material_points, old, boundary, line_search=BacktrackingOptions())
    fresh = evaluate_global_residual(ops, case.material_points, old, result.z, boundary)
    np.testing.assert_array_equal(fresh.residual, result.evaluation.residual)
    assert all((history.alpha > 0) == bool(cell % 2)
        for cell, history in enumerate(fresh.material_trial.material_states))
    assert np.all(result.residual_norms[-1] <= result.thresholds)


def test_backtracking_recomputes_trials_without_accepting_rejected_plasticity(monkeypatch: pytest.MonkeyPatch) -> None:
    case, ops, old = make_setup()
    # All prescribed and converged strains are elastic; a deliberately oversized
    # Newton step yields locally and must be discarded by the merit test.
    expected, boundary = affine(ops, np.diag([.0003, 0.]))
    jac = RetainedAnalyticalJacobian(ops, case.material_points, old)
    original = evaluate_global_residual
    trials: list[RetainedFaceResidual] = []

    def evaluate(
        operators: RetainedFaceOperators, points: list[MaterialPoint], committed: TpsaState,
        z: FloatArray, bc: FloatArray, **kwargs: FloatArray | None,
    ) -> RetainedFaceResidual:
        assert_same_state(committed, old)
        result = original(operators, points, committed, z, bc, **kwargs)
        trials.append(result)
        return result

    monkeypatch.setattr(newton, "evaluate_global_residual", evaluate)
    result = solve_newton(ops, case.material_points, old, boundary,
        lambda z, evaluation: .125 * jac.L, line_search=BacktrackingOptions(), max_iterations=2)
    assert result.iterations == 1
    np.testing.assert_array_equal(result.step_lengths, [.125])
    assert result.residual_evaluations == len(trials) == 5
    assert any(s.alpha > 0 for trial in trials[1:-1] for s in trial.material_trial.material_states)
    assert all(s.alpha == 0 for s in result.evaluation.material_trial.material_states)
    np.testing.assert_allclose(result.z, expected, rtol=1e-10, atol=1e-6)
    assert all(p.trial is None and p.committed.alpha == 0 for p in case.material_points)
    assert_same_state(old, TpsaState.zeros(case.grid))


@pytest.mark.parametrize("block", [0, 1, 2, 3])
def test_every_block_is_required_for_convergence(block: int) -> None:
    case, ops, old = make_setup(2)
    imbalance = np.zeros(ops.num_dofs)
    index = [0, 2 * ops.num_cells, 3 * ops.num_cells, 4 * ops.num_cells][block]
    imbalance[index] = 1.
    with pytest.raises(NewtonConvergenceError, match="after 0 corrections"):
        solve_newton(ops, case.material_points, old, old.bc_values, forbidden,
                     committed_residual=imbalance, max_iterations=0)


def test_load_snapshot_and_equilibrated_initial_guess_are_independent() -> None:
    case, ops, old = make_setup(2)
    expected, boundary = affine(ops, np.diag([1e-5, 0.]))
    boundary[:, ops.free_faces] = np.nan
    saved_bc = boundary.copy()
    jac = RetainedAnalyticalJacobian(ops, case.material_points, old)
    body = np.full((2, ops.num_cells), 20.)
    imbalance = np.zeros(ops.num_dofs)
    imbalance[ops.u_slice] = body.ravel("F")  # Equal and opposite fixed loads.
    saved_body, saved_imbalance = body.copy(), imbalance.copy()

    def callback(z: FloatArray, evaluation: RetainedFaceResidual) -> JacobianMatrix:
        boundary[:] = 10
        body[:] = 0
        imbalance[:] = 0
        return jac(z, evaluation)

    result = solve_newton(ops, case.material_points, old, boundary, callback,
        body_force_increment=body, committed_residual=imbalance)
    np.testing.assert_allclose(result.z, expected, rtol=1e-10, atol=1e-7)
    again = solve_newton(ops, case.material_points, old, saved_bc, forbidden,
        initial_z=result.z, body_force_increment=saved_body, committed_residual=saved_imbalance, max_iterations=0)
    assert again.iterations == 0 and again.step_lengths.size == 0
    assert again.residual_evaluations == 1
    result.z[:] = 10
    result.evaluation.material_trial.material_states[0].alpha = 2.
    np.testing.assert_allclose(again.z, expected, rtol=1e-10, atol=1e-7)
    assert_same_state(old, TpsaState.zeros(case.grid))


@pytest.mark.parametrize("failure", ["iterations", "search", "shape", "nan", "singular", "callback", "material"])
def test_failure_preserves_history_and_existing_material_trials(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    case, ops, old = make_setup(2)
    for point in case.material_points:
        point.update(strain(np.diag([.004, 0., 0.])))
    saved = deepcopy(case.material_points)
    _, boundary = affine(ops, np.diag([1e-5, 0.]))
    matrix = RetainedAnalyticalJacobian(ops, case.material_points, old).L
    if failure == "shape":
        matrix = matrix[:-1]
    elif failure == "nan":
        matrix.data[0] = np.nan
    elif failure == "singular":
        matrix.data[:] = 0
    elif failure == "search":
        matrix *= -1
    else:
        matrix *= 2

    def callback(z: FloatArray, evaluation: RetainedFaceResidual) -> JacobianMatrix:
        if failure == "callback":
            raise RuntimeError("Injected callback failure")
        if failure == "material":
            def bad_update(self: MaterialPoint, increment: strain) -> None:
                raise RuntimeError("Injected material failure")
            monkeypatch.setattr(MaterialPoint, "update", bad_update)
        return matrix

    error = ValueError if failure in ("shape", "nan") else RuntimeError
    with pytest.raises(error):
        solve_newton(ops, case.material_points, old, boundary, callback, max_iterations=1,
            line_search=BacktrackingOptions(max_backtracks=2) if failure == "search" else None)
    assert_same_state(old, TpsaState.zeros(case.grid))
    for point, before in zip(case.material_points, saved, strict=True):
        assert point.trial is not None and before.trial is not None
        assert point.trial.alpha == before.trial.alpha
        np.testing.assert_array_equal(point.trial.stress.to_numpy(), before.trial.stress.to_numpy())
        np.testing.assert_array_equal(point.committed.stress.to_numpy(), before.committed.stress.to_numpy())


@pytest.mark.parametrize("defect", ["initial_shape", "initial_nan", "bc", "body", "imbalance", "tol_length", "negative_tol", "rtol", "limit"])
def test_invalid_solver_inputs_are_rejected(defect: str) -> None:
    case, ops, old = make_setup(2)
    z, boundary = affine(ops, np.diag([1e-5, 0.]))
    body: FloatArray = np.zeros((2, ops.num_cells))
    imbalance = np.zeros(ops.num_dofs)
    atol: newton.BlockTolerances = (1e-5, 1e-12, 1e-12, 1e-5)
    rtol: newton.BlockTolerances = (1e-8,) * 4
    if defect == "initial_shape":
        z = z[:-1]
    elif defect == "initial_nan":
        z[0] = np.nan
    elif defect == "bc":
        boundary[0, ops.boundary_faces[0]] = np.inf
    elif defect == "body":
        body = body.ravel()
    elif defect == "imbalance":
        imbalance[0] = np.nan
    elif defect == "tol_length":
        atol = (1e-5, 1e-12, 1e-12)  # type: ignore[assignment]
    elif defect == "negative_tol":
        atol = (-1., 0., 0., 0.)
    elif defect == "rtol":
        rtol = (1., 0., 0., 0.)
    with pytest.raises(ValueError):
        solve_newton(ops, case.material_points, old, boundary, forbidden, initial_z=z,
            body_force_increment=body, committed_residual=imbalance, atol=atol, rtol=rtol,
            max_iterations=-1 if defect == "limit" else 0)


@pytest.mark.parametrize("value", [0., 1., -1., np.nan, np.inf])
def test_invalid_backtracking_coefficients(value: float) -> None:
    with pytest.raises(ValueError):
        BacktrackingOptions(factor=value)
    with pytest.raises(ValueError):
        BacktrackingOptions(armijo=value)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_backtrack_limit(value: int) -> None:
    with pytest.raises(ValueError):
        BacktrackingOptions(max_backtracks=value)
