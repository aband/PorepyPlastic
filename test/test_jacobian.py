"""Numerical material tangents, coupled residual derivatives, and Newton integration."""

from copy import deepcopy
from dataclasses import replace
from typing import TypeAlias

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray

from coupling.jacobian import AnalyticalJacobian, FiniteDifferenceJacobian, finite_difference_material_tangent
from coupling.newton import solve_newton
from coupling.plane_strain import PlaneStrainTpsa
from coupling.postprocessing import gradient_green_gauss, reconstruct_face_displacement
from coupling.residual import TpsaOperators, TpsaState, evaluate_global_residual
from coupling.transfer import CellToFaceTransfer
from material_state import MaterialPoint, MaterialState
from tensor import strain

JacobianType: TypeAlias = type[FiniteDifferenceJacobian] | type[AnalyticalJacobian]
FloatArray: TypeAlias = NDArray[np.float64]
Setup: TypeAlias = tuple[PlaneStrainTpsa, TpsaOperators, CellToFaceTransfer, TpsaState]


@pytest.fixture
def setup() -> Setup:
    case = PlaneStrainTpsa(cells_per_axis=3)
    case.prepare_simulation()
    return case, TpsaOperators(case), CellToFaceTransfer(case.grid), TpsaState.zeros(case.grid)


def affine(case: PlaneStrainTpsa, gradient: FloatArray) -> tuple[FloatArray, FloatArray]:
    nc = case.grid.num_cells
    x = np.concatenate((
        (gradient @ case.grid.cell_centers[:2]).ravel(order="F"),
        np.full(nc, case.material.isotropic_shear_modulus * (gradient[0, 1] - gradient[1, 0])),
        np.full(nc, case.material.lame_parameter * np.trace(gradient)),
    ))
    boundary = np.zeros_like(case.bc_values)
    bf = case.grid.get_all_boundary_faces()
    boundary[:, bf] = gradient @ case.grid.face_centers[:2, bf]
    return x, boundary


def assert_state_equal(actual: TpsaState, expected: TpsaState) -> None:
    for name in ("x", "epsilon", "bc_values", "traction"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for a, b in zip(actual.material_states, expected.material_states, strict=True):
        assert_history_equal(a, b)


def assert_history_equal(actual: MaterialState, expected: MaterialState) -> None:
    assert actual.alpha == expected.alpha
    for name in ("stress", "plastic_strain", "backstress"):
        np.testing.assert_array_equal(getattr(actual, name).to_numpy(), getattr(expected, name).to_numpy())


@pytest.mark.parametrize("preloaded", [False, True])
def test_elastic_tangent_including_tensor_shear_and_out_of_plane_stress(setup: Setup, preloaded: bool) -> None:
    case, _, _, _ = setup
    point = case.material_points[0]
    old = MaterialState()
    increment = np.diag([1e-4, -2e-5, 0.0])
    if preloaded:
        old, _ = point.update(strain(np.diag([0.004, 0.0, 0.0])))
        assert old.alpha > 0
        increment *= -1  # Strictly elastic unloading from a plastic history.
    before = deepcopy(point)
    old_before = old.copy()
    tangent = finite_difference_material_tangent(point, old, increment)
    expected = point.material.elastic_tensor.to_numpy()[:, :, :2, :2]
    np.testing.assert_allclose(tangent, expected, rtol=2e-7, atol=1.0)
    assert tangent.shape == (3, 3, 2, 2)
    # Perturbing ONE gradient shear entry changes each symmetric strain by half.
    assert tangent[0, 1, 0, 1] == pytest.approx(point.material.isotropic_shear_modulus)
    assert tangent[2, 2, 0, 0] == pytest.approx(point.material.lame_parameter)
    assert_history_equal(old, old_before)
    assert_history_equal(point.committed, before.committed)
    if preloaded:
        assert point.trial is not None and before.trial is not None
        assert_history_equal(point.trial, before.trial)
    else:
        assert point.trial is None


@pytest.mark.parametrize("cells", [1, 3])
@pytest.mark.parametrize("jacobian_type", [FiniteDifferenceJacobian, AnalyticalJacobian])
def test_elastic_jacobian_recovers_tpsa_matrix(jacobian_type: JacobianType, cells: int) -> None:
    case = PlaneStrainTpsa(cells_per_axis=cells)
    case.prepare_simulation()
    ops, transfer = TpsaOperators(case), CellToFaceTransfer(case.grid)
    old = TpsaState.zeros(case.grid)
    x, bc = affine(case, np.array([[1e-5, 2e-5], [-3e-5, 4e-5]]))
    evaluation = evaluate_global_residual(ops, transfer, case.material_points, old, x, bc)
    jacobian = jacobian_type(ops, transfer, case.material_points, old)
    matrix = jacobian(x, evaluation)
    assert sps.issparse(matrix)
    # Scale columns to compare u (m) and r,p (Pa) without hiding stress columns.
    scale = sps.diags(np.r_[np.ones(2 * ops.num_cells), np.full(2 * ops.num_cells, 2.1e11)])
    relative = sps.linalg.norm((matrix - ops.A) @ scale) / sps.linalg.norm(ops.A @ scale)
    assert relative < 1e-8
    np.testing.assert_array_equal(matrix[2 * ops.num_cells:].toarray(), ops.A[2 * ops.num_cells:].toarray())


@pytest.mark.parametrize("jacobian_type", [FiniteDifferenceJacobian, AnalyticalJacobian])
def test_sparse_gradient_derivative_matches_reconstruction_at_fixed_boundary(jacobian_type: JacobianType, setup: Setup) -> None:
    case, ops, transfer, old = setup
    jacobian = jacobian_type(ops, transfer, case.material_points, old)
    rng = np.random.default_rng(17)
    dx = rng.normal(size=4 * ops.num_cells)
    dx[2 * ops.num_cells:] *= 2.1e11
    zero_boundary = np.zeros_like(case.bc_values)
    face = reconstruct_face_displacement(ops.grid, ops.face_discretization, ops.rhs_matrix, dx, zero_boundary)
    expected = gradient_green_gauss(ops.grid, face).transpose(2, 0, 1).ravel()
    np.testing.assert_allclose(jacobian._gradient @ dx, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("regime", ["plastic", "mixed", "unloading"])
@pytest.mark.parametrize("custom_weights", [False, True])
@pytest.mark.parametrize("jacobian_type", [FiniteDifferenceJacobian, AnalyticalJacobian])
def test_matches_full_residual_directional_derivative_for_every_field(
    jacobian_type: JacobianType,
    setup: Setup, regime: str, custom_weights: bool,
) -> None:
    case, ops, transfer, old = setup
    if custom_weights:
        transfer = CellToFaceTransfer.from_rule(case.grid, lambda grid, face, cells: np.array([0.2, 0.8]))
    gradient = np.array([[0.004, 0.001], [0.0003, -0.0007]])
    x, bc = affine(case, gradient)
    if regime == "unloading":
        old = evaluate_global_residual(ops, transfer, case.material_points, old, x, bc).trial
        x, bc = affine(case, 0.95 * gradient)
    elif regime == "mixed":
        x, bc = affine(case, 0.2 * gradient)
        x[:2 * ops.num_cells] += np.random.default_rng(1).normal(size=2 * ops.num_cells) * 0.0004
    evaluation = evaluate_global_residual(ops, transfer, case.material_points, old, x, bc)
    yielded = np.array([
        current.alpha > previous.alpha
        for current, previous in zip(evaluation.trial.material_states, old.material_states, strict=True)
    ])
    if regime == "mixed":
        assert np.any(yielded) and not np.all(yielded)
    else:
        assert np.all(yielded) if regime == "plastic" else not np.any(yielded)
    saved, evaluated_before = old.copy(), deepcopy(evaluation)
    jacobian = jacobian_type(ops, transfer, case.material_points, old)
    matrix = jacobian(x, evaluation)
    rng = np.random.default_rng(23)
    nc = ops.num_cells
    # Independent centered differences of the FULL nonlinear residual are only an oracle.
    for indices, scale in ((slice(0, 2 * nc), 0.003), (slice(2 * nc, 3 * nc), 6e8), (slice(3 * nc, 4 * nc), 6e8)):
        direction = np.zeros_like(x)
        direction[indices] = scale * rng.normal(size=direction[indices].size)
        eta = 1e-6
        plus = evaluate_global_residual(ops, transfer, case.material_points, old, x + eta * direction, bc)
        minus = evaluate_global_residual(ops, transfer, case.material_points, old, x - eta * direction, bc)
        expected = (plus.residual - minus.residual) / (2 * eta)
        actual = np.asarray(matrix @ direction, dtype=np.float64)
        for block in (slice(0, 2 * nc), slice(2 * nc, 3 * nc), slice(3 * nc, 4 * nc)):
            tolerance = 2e-7 * max(float(np.linalg.norm(expected[block])), 1e-12)
            np.testing.assert_allclose(actual[block], expected[block], rtol=2e-5, atol=tolerance)
    assert_state_equal(old, saved)
    assert_state_equal(evaluation.trial, evaluated_before.trial)
    np.testing.assert_array_equal(evaluation.residual, evaluated_before.residual)
    np.testing.assert_array_equal(evaluation.stress_correction, evaluated_before.stress_correction)
    assert all(point.trial is None for point in case.material_points)


@pytest.mark.parametrize("jacobian_type", [FiniteDifferenceJacobian, AnalyticalJacobian])
def test_callback_solves_elastic_load_without_changing_newton_api(jacobian_type: JacobianType, setup: Setup) -> None:
    case, ops, transfer, old = setup
    expected = np.asarray(sps.linalg.spsolve(ops.A, ops.assemble_rhs(case.bc_values)))
    result = solve_newton(
        ops, transfer, case.material_points, old, case.bc_values,
        jacobian_type(ops, transfer, case.material_points, old),
        max_iterations=2, rtol=(0.0, 0.0, 0.0),
    )
    assert result.iterations <= 2
    np.testing.assert_allclose(result.trial.x, expected, rtol=1e-8, atol=1e-6)
    assert_state_equal(old, TpsaState.zeros(case.grid))


@pytest.mark.parametrize("jacobian_type", [FiniteDifferenceJacobian, AnalyticalJacobian])
def test_callback_solves_plastic_load_and_preserves_existing_trials(jacobian_type: JacobianType, setup: Setup) -> None:
    case, ops, transfer, old = setup
    x, bc = affine(case, np.array([[0.004, 0.001], [0.0, -0.0005]]))
    initial = x.copy()
    initial[:2 * ops.num_cells] += 1e-5 * np.random.default_rng(2).normal(size=2 * ops.num_cells)
    for point in case.material_points:
        point.update(strain(np.diag([1e-5, 0.0, 0.0])))
    before = deepcopy(case.material_points)
    result = solve_newton(
        ops, transfer, case.material_points, old, bc,
        jacobian_type(ops, transfer, case.material_points, old),
        initial_x=initial, max_iterations=12,
    )
    assert result.iterations > 0
    assert np.all(result.residual_norms[-1] <= result.thresholds)
    np.testing.assert_allclose(result.trial.x, x, rtol=1e-7, atol=1e-3)
    np.testing.assert_allclose(result.trial.x[:2 * ops.num_cells], x[:2 * ops.num_cells], atol=1e-11)
    assert all(history.alpha > 0 for history in result.trial.material_states)
    assert_state_equal(old, TpsaState.zeros(case.grid))
    for point, saved in zip(case.material_points, before, strict=True):
        assert_history_equal(point.committed, saved.committed)
        assert point.trial is not None and saved.trial is not None
        assert_history_equal(point.trial, saved.trial)


@pytest.mark.parametrize("step", [0.0, -1e-10, np.nan, np.inf])
def test_invalid_step_is_rejected(setup: Setup, step: float) -> None:
    case, ops, transfer, old = setup
    with pytest.raises(ValueError, match="step.*finite and positive"):
        FiniteDifferenceJacobian(ops, transfer, case.material_points, old, step=step)
    with pytest.raises(ValueError, match="step.*finite and positive"):
        finite_difference_material_tangent(case.material_points[0], old.material_states[0], np.zeros((3, 3)), step=step)


@pytest.mark.parametrize("jacobian_type", [FiniteDifferenceJacobian, AnalyticalJacobian])
def test_mismatched_candidate_is_rejected(jacobian_type: JacobianType, setup: Setup) -> None:
    case, ops, transfer, old = setup
    evaluation = evaluate_global_residual(ops, transfer, case.material_points, old, old.x, case.bc_values)
    jacobian = jacobian_type(ops, transfer, case.material_points, old)
    with pytest.raises(ValueError, match="matching evaluation"):
        jacobian(old.x + 1.0, evaluation)


def test_failure_during_local_perturbation_preserves_all_input_histories(
    setup: Setup, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, ops, transfer, old = setup
    point = case.material_points[0]
    point.update(strain(np.diag([0.004, 0.0, 0.0])))
    before = deepcopy(point)
    evaluation = evaluate_global_residual(ops, transfer, case.material_points, old, old.x, case.bc_values)
    saved = old.copy()
    jacobian = FiniteDifferenceJacobian(ops, transfer, case.material_points, old)
    original = MaterialPoint.update
    count = 0

    def failing_update(self: MaterialPoint, increment: strain) -> tuple[MaterialState, None]:
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("Injected perturbation failure")
        return original(self, increment)

    monkeypatch.setattr(MaterialPoint, "update", failing_update)
    with pytest.raises(RuntimeError, match="Injected perturbation failure"):
        jacobian(old.x, evaluation)
    assert count == 3
    assert_state_equal(old, saved)
    assert_history_equal(point.committed, before.committed)
    assert point.trial is not None and before.trial is not None
    assert_history_equal(point.trial, before.trial)


def linear_j2_tangent(point: MaterialPoint, old: MaterialState, increment: FloatArray) -> FloatArray:
    """Analytical plastic-branch derivative for linear mixed hardening only.

    Differentiate sigma = sigma_predictor - 2*mu*gamma*n in closed form,
    independently of the numerical tangent and the local return-map routine.
    """
    parameters = point.parameters
    assert parameters.sigma_u == parameters.sigma_y
    mu, bulk = point.material.isotropic_shear_modulus, point.material.bulk_modulus
    identity = np.eye(3)
    predictor = old.stress.to_numpy() + 2 * mu * increment
    predictor += point.material.lame_parameter * np.trace(increment) * identity
    backstress = old.backstress.to_numpy()
    shifted = predictor - np.trace(predictor) / 3 * identity
    shifted -= backstress - np.trace(backstress) / 3 * identity
    q = float(np.linalg.norm(shifted))
    radius = np.sqrt(2 / 3) * (parameters.sigma_y + parameters.theta * parameters.H_bar * old.alpha)
    assert q > radius  # Stay strictly inside the plastic branch.
    denominator = 2 * mu + 2 / 3 * parameters.H_bar
    gamma = (q - radius) / denominator
    direction = shifted / q
    spherical = np.einsum("ij,kl->ijkl", identity, identity)
    symmetric = 0.5 * (
        np.einsum("ik,jl->ijkl", identity, identity)
        + np.einsum("il,jk->ijkl", identity, identity)
    )
    tangent = bulk * spherical + 2 * mu * (1 - 2 * mu * gamma / q) * (symmetric - spherical / 3)
    tangent += 4 * mu**2 * (gamma / q - 1 / denominator) * np.einsum("ij,kl->ijkl", direction, direction)
    return np.asarray(tangent[:, :, :2, :2], dtype=np.float64)


@pytest.mark.parametrize("theta", [0.0, 0.4, 1.0], ids=["kinematic", "mixed", "isotropic"])
@pytest.mark.parametrize("preloaded", [False, True], ids=["virgin", "plastic-history"])
def test_plastic_material_tangent_matches_analytical_j2(
    setup: Setup, theta: float, preloaded: bool,
) -> None:
    case, _, _, _ = setup
    point = deepcopy(case.material_points[0])
    point.parameters = replace(point.parameters, theta=theta)
    old = MaterialState()
    if preloaded:
        old, _ = point.update(strain(np.diag([0.004, -0.0005, 0.0])))
        assert old.alpha > 0
    increment = np.array([[0.0025, 0.0007, 0.0], [0.0007, -0.0008, 0.0], [0.0, 0.0, 0.0]])
    increment.flags.writeable = False
    saved, saved_point = old.copy(), deepcopy(point)
    expected = linear_j2_tangent(point, old, increment)
    actual = finite_difference_material_tangent(point, old, increment)
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e3)
    assert_history_equal(old, saved)
    assert_history_equal(point.committed, saved_point.committed)
    if preloaded:
        assert point.trial is not None and saved_point.trial is not None
        assert_history_equal(point.trial, saved_point.trial)
    else:
        assert point.trial is None


def test_forward_difference_error_decreases_with_step_size(setup: Setup) -> None:
    case, _, _, _ = setup
    point = case.material_points[0]
    old = MaterialState()
    increment = np.array([[0.004, 0.00065, 0.0], [0.00065, -0.0007, 0.0], [0.0, 0.0, 0.0]])
    expected = linear_j2_tangent(point, old, increment)
    errors = np.array([
        np.linalg.norm(finite_difference_material_tangent(point, old, increment, step=h) - expected)
        for h in (1e-5, 1e-6, 1e-7)
    ])
    # A forward difference has O(h) truncation error away from yield switches.
    ratios = errors[:-1] / errors[1:]
    assert np.all((ratios > 8) & (ratios < 12))
    default = finite_difference_material_tangent(point, old, increment)
    assert np.linalg.norm(default - expected) / np.linalg.norm(expected) < 1e-7


@pytest.mark.parametrize("jacobian_type", [FiniteDifferenceJacobian, AnalyticalJacobian])
def test_every_jacobian_column_matches_residual_during_plastic_reloading(jacobian_type: JacobianType) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.prepare_simulation()
    ops = TpsaOperators(case)
    transfer = CellToFaceTransfer.from_rule(case.grid, lambda grid, face, cells: np.array([0.3, 0.7]))
    gradient = np.array([[0.004, 0.0008], [-0.0002, -0.0003]])
    x_n, bc_n = affine(case, gradient)
    old = evaluate_global_residual(
        ops, transfer, case.material_points, TpsaState.zeros(case.grid), x_n, bc_n,
    ).trial
    assert all(history.alpha > 0 for history in old.material_states)
    x, bc = affine(case, 1.2 * gradient)
    # Nonuniform reloading gives distinct local tangents and exposes cell-order errors.
    x[:2 * ops.num_cells] += 1e-5 * np.random.default_rng(7).normal(size=2 * ops.num_cells)
    evaluation = evaluate_global_residual(ops, transfer, case.material_points, old, x, bc)
    assert all(a.alpha > b.alpha for a, b in zip(evaluation.trial.material_states, old.material_states, strict=True))
    matrix = jacobian_type(ops, transfer, case.material_points, old)(x, evaluation).toarray()
    # Stress and displacement variables need different units/scales when perturbed.
    scales = np.r_[np.full(2 * ops.num_cells, 0.003), np.full(2 * ops.num_cells, 6e8)]
    eta = 1e-6
    for column, scale in enumerate(scales):
        direction = np.zeros_like(x)
        direction[column] = scale
        plus = evaluate_global_residual(ops, transfer, case.material_points, old, x + eta * direction, bc)
        minus = evaluate_global_residual(ops, transfer, case.material_points, old, x - eta * direction, bc)
        expected = (plus.residual - minus.residual) / (2 * eta)
        actual = matrix[:, column] * scale
        for start, end in ((0, 2 * ops.num_cells), (2 * ops.num_cells, 3 * ops.num_cells), (3 * ops.num_cells, 4 * ops.num_cells)):
            target = expected[start:end]
            tolerance = 2e-7 * max(float(np.linalg.norm(target)), 1e-12)
            np.testing.assert_allclose(actual[start:end], target, rtol=2e-5, atol=tolerance)


@pytest.mark.parametrize("jacobian_type", [FiniteDifferenceJacobian, AnalyticalJacobian])
def test_callback_repeated_evaluation_uses_its_committed_snapshot(jacobian_type: JacobianType, setup: Setup) -> None:
    case, ops, transfer, virgin = setup
    gradient = np.array([[0.004, 0.001], [0.0, -0.0005]])
    x_n, bc_n = affine(case, gradient)
    old = evaluate_global_residual(ops, transfer, case.material_points, virgin, x_n, bc_n).trial
    saved = old.copy()
    x, bc = affine(case, 1.1 * gradient)
    evaluation = evaluate_global_residual(ops, transfer, case.material_points, old, x, bc)
    jacobian = jacobian_type(ops, transfer, case.material_points, old)
    expected = jacobian(x, evaluation).toarray()
    other_x, other_bc = affine(case, 0.95 * gradient)
    other = evaluate_global_residual(ops, transfer, case.material_points, old, other_x, other_bc)
    assert not np.allclose(jacobian(other_x, other).toarray(), expected)
    np.testing.assert_array_equal(jacobian(x, evaluation).toarray(), expected)
    assert_state_equal(old, saved)
    # Explicitly change the caller's original object: the callback owns a snapshot.
    old.epsilon[:] = 0.0
    for history in old.material_states:
        history.alpha += 1.0
    np.testing.assert_array_equal(jacobian(x, evaluation).toarray(), expected)
    assert all(point.trial is None for point in case.material_points)


def test_unrepresentable_perturbation_is_rejected_without_updating_point(setup: Setup) -> None:
    case, _, _, _ = setup
    point = case.material_points[0]
    old, _ = point.update(strain(np.diag([0.004, 0.0, 0.0])))
    saved, saved_point = old.copy(), deepcopy(point)
    increment = np.diag([-1e-4, 0.0, 0.0])
    with pytest.raises(ValueError, match="distinct perturbation"):
        finite_difference_material_tangent(point, old, increment, step=1e-30)
    assert_history_equal(old, saved)
    assert_history_equal(point.committed, saved_point.committed)
    assert point.trial is not None and saved_point.trial is not None
    assert_history_equal(point.trial, saved_point.trial)
