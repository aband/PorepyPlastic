"""Analytical J2 tangent and the Dirichlet document's face/condensed derivatives."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from coupling.jacobian import AnalyticalJacobian, analytical_material_tangent
from coupling.plane_strain import PlaneStrainTpsa
from coupling.plastic_plane_strain import main, run_example
from coupling.residual import TpsaOperators, TpsaResidual, TpsaState, evaluate_global_residual
from coupling.transfer import CellToFaceTransfer
from material_state import MaterialPoint, MaterialState
from scalar_hardening import HardeningParameters
from tensor import strain, stress


def update_at(point: MaterialPoint, old: MaterialState, increment: NDArray[np.float64]) -> MaterialState:
    local = MaterialPoint(point.material, point.parameters, point.model, point.K_law, point.H_law, old)
    result, _ = local.update(strain(increment))
    return result


def nonlinear_kinematic(alpha: float, parameters: HardeningParameters) -> tuple[float, float]:
    rate, saturation = 120.0, 70e6
    modulus = (1 - parameters.theta) * parameters.H_bar
    return (float(modulus * alpha + saturation * (-np.expm1(-rate * alpha))),
            float(modulus + saturation * rate * np.exp(-rate * alpha)))


@pytest.mark.parametrize("law", ["isotropic", "kinematic", "mixed", "nonlinear", "perfect"])
@pytest.mark.parametrize("preloaded", [False, True])
def test_full_3d_tangent_matches_return_map_derivatives(law: str, preloaded: bool) -> None:
    case = PlaneStrainTpsa(cells_per_axis=1)
    case.prepare_simulation()
    point = case.material_points[0]
    if law in ("isotropic", "kinematic"):
        point.parameters = replace(point.parameters, theta=1.0 if law == "isotropic" else 0.0)
    elif law == "nonlinear":
        point.parameters = replace(point.parameters, sigma_u=450e6, delta=80)
        point.H_law = nonlinear_kinematic
    elif law == "perfect":
        point.parameters = replace(point.parameters, H_bar=0.0)
    old = MaterialState()
    if preloaded:
        old = update_at(point, old, np.diag([0.004, -0.001, -0.0005]))
        assert old.alpha > 0
    # Nonproportional 3D loading exercises the flow-direction derivative too.
    increment = np.array([[0.003, 0.0006, 0.0003], [0.0006, -0.0009, -0.0002], [0.0003, -0.0002, -0.0001]])
    saved_increment = increment.copy()
    old_before, point_before = old.copy(), deepcopy(point)
    returned = update_at(point, old, increment)
    assert returned.alpha > old.alpha
    predictor = old.stress.to_numpy() + np.einsum("ijkl,kl->ij", point.material.elastic_tensor.to_numpy(), increment)
    tangent = point.model.consistent_tangent(
        stress(predictor), old.backstress, returned.alpha,
        float((returned.alpha - old.alpha) / np.sqrt(2 / 3)),
        point.material, point.K_law, point.H_law, point.parameters,
    ).to_numpy()
    assert tangent.shape == (3, 3, 3, 3)
    np.testing.assert_allclose(tangent, tangent.transpose(2, 3, 0, 1), rtol=1e-14, atol=1e-4)
    h = 1e-8
    for k in range(3):
        for l in range(3):
            basis = np.zeros((3, 3))
            basis[k, l] += 0.5
            basis[l, k] += 0.5
            plus = update_at(point, old, increment + h * basis)
            minus = update_at(point, old, increment - h * basis)
            expected = (plus.stress.to_numpy() - minus.stress.to_numpy()) / (2 * h)
            np.testing.assert_allclose(tangent[:, :, k, l], expected, rtol=2e-7, atol=100)
    # Plane-strain adapter retains sigma_zz and both half-shear gradient columns.
    adapter = analytical_material_tangent(point, old, increment, returned=returned)
    np.testing.assert_array_equal(adapter, tangent[:, :, :2, :2])
    np.testing.assert_array_equal(analytical_material_tangent(point, old, increment), adapter)
    np.testing.assert_array_equal(increment, saved_increment)
    assert old.alpha == old_before.alpha and point.trial is point_before.trial is None
    for name in ("stress", "plastic_strain", "backstress"):
        np.testing.assert_array_equal(getattr(old, name).to_numpy(), getattr(old_before, name).to_numpy())
        np.testing.assert_array_equal(getattr(point.committed, name).to_numpy(), getattr(point_before.committed, name).to_numpy())


@pytest.mark.parametrize("regime", ["zero", "hydrostatic", "elastic", "unloading", "yield"])
def test_elastic_tangent_is_exact_and_handles_zero_direction(regime: str) -> None:
    case = PlaneStrainTpsa(cells_per_axis=1)
    case.prepare_simulation()
    point = case.material_points[0]
    old = MaterialState()
    increment = np.zeros((3, 3))
    if regime == "hydrostatic":
        increment = np.eye(3) * 0.001
    elif regime == "elastic":
        increment = np.diag([1e-5, -2e-5, 0.0])
    elif regime == "unloading":
        old = update_at(point, old, np.diag([0.004, 0.0, 0.0]))
        increment[0, 0] = -1e-5
    elif regime == "yield":
        increment[0, 0] = point.parameters.sigma_y / (2 * point.material.isotropic_shear_modulus)
    returned = update_at(point, old, increment)
    assert returned.alpha == old.alpha
    actual = analytical_material_tangent(point, old, increment, returned=returned)
    np.testing.assert_array_equal(actual, point.material.elastic_tensor.to_numpy()[:, :, :2, :2])
    assert actual[0, 1, 0, 1] == point.material.isotropic_shear_modulus
    assert actual[2, 2, 0, 0] == point.material.lame_parameter


def test_document_condensation_and_dirichlet_face_derivatives() -> None:
    case = PlaneStrainTpsa(cells_per_axis=3, displacement_gradient=np.array([[0.004, 0.0006], [-0.0002, -0.0003]]))
    case.prepare_simulation()
    ops = TpsaOperators(case)
    transfer = CellToFaceTransfer.from_rule(case.grid, lambda grid, face, cells: np.array([0.2, 0.8]))
    nc, nf, nu = ops.num_cells, ops.num_faces, 2 * ops.num_cells
    virgin = TpsaState.zeros(case.grid)
    u_n = (case.displacement_gradient @ case.grid.cell_centers[:2]).ravel(order="F")
    r_n, p_n = ops.solve_auxiliary(u_n, ops.assemble_rhs(case.bc_values))
    old = evaluate_global_residual(ops, transfer, case.material_points, virgin, np.r_[u_n, r_n, p_n], case.bc_values).trial
    bc = 1.2 * case.bc_values
    sources = np.r_[np.zeros(nu), np.full(nc, 1e-6), np.full(nc, -1e-6)]
    rhs = ops.assemble_rhs(bc, sources=sources)

    def evaluate(u: NDArray[np.float64]) -> TpsaResidual:
        # Document D10-D12: re-solve both auxiliaries for EVERY displacement trial.
        r, p = ops.solve_auxiliary(u, rhs)
        return evaluate_global_residual(ops, transfer, case.material_points, old, np.r_[u, r, p], bc, sources=sources)

    rng = np.random.default_rng(35)
    u = 1.2 * u_n + rng.normal(size=nu) * 1e-5
    evaluation = evaluate(u)
    assert all(a.alpha > b.alpha for a, b in zip(evaluation.trial.material_states, old.material_states, strict=True))
    callback = AnalyticalJacobian(ops, transfer, case.material_points, old)
    x = evaluation.trial.x
    face = callback.face_jacobian(x, evaluation)
    full = callback(x, evaluation)
    assert face.shape == (2 * nf, 4 * nc) and full.shape == (4 * nc, 4 * nc)
    # Small dense solves are a test oracle only. Production stays sparse/coupled.
    R = np.linalg.solve(ops.A_rr.toarray(), -ops.A_ru.toarray())
    P = np.linalg.solve(ops.A_pp.toarray(), -ops.A_pu.toarray())
    lift = np.vstack((np.eye(nu), R, P))
    condensed_face = face @ lift
    condensed = full[:nu] @ lift
    np.testing.assert_allclose(condensed, ops.D_u @ condensed_face, rtol=1e-13, atol=1e-3)
    np.testing.assert_array_equal(full[nu:].toarray(), ops.A[nu:].toarray())
    assert callback._gradient[:, nu:3 * nc].nnz == 0  # Document D3.
    assert np.linalg.norm(condensed - full[:nu, :nu].toarray()) > 1e8  # Pressure/rotation feedback matters.
    interior = np.setdiff1d(np.arange(nf), ops.boundary_faces)
    assert np.linalg.norm(condensed_face.reshape(nf, 2, nu)[ops.boundary_faces]) > 0
    # Opposite incidence signs cancel shared interior traction derivatives.
    interior_rows = (2 * interior[:, None] + np.arange(2)).ravel()
    cell_forces = (ops.D_u[:, interior_rows] @ condensed_face[interior_rows]).reshape(nc, 2, nu)
    np.testing.assert_allclose(cell_forces.sum(axis=0), 0.0, atol=2e-4)
    h = 1e-6
    for _ in range(3):
        direction = rng.normal(size=nu) * 0.003
        plus, minus = evaluate(u + h * direction), evaluate(u - h * direction)
        expected_face = (plus.trial.traction - minus.trial.traction) / (2 * h)
        actual_face = (condensed_face @ direction).reshape((2, nf), order="F")
        for faces in (interior, ops.boundary_faces):
            np.testing.assert_allclose(actual_face[:, faces], expected_face[:, faces], rtol=2e-7, atol=0.2)
        expected = (plus.residual[:nu] - minus.residual[:nu]) / (2 * h)
        np.testing.assert_allclose(condensed @ direction, expected, rtol=2e-7, atol=0.5)


def test_callback_reuses_returned_histories_without_local_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2, displacement_gradient=np.diag([0.004, 0.0]))
    case.prepare_simulation()
    ops, transfer = TpsaOperators(case), CellToFaceTransfer(case.grid)
    old = TpsaState.zeros(case.grid)
    u = (case.displacement_gradient @ case.grid.cell_centers[:2]).ravel(order="F")
    r, p = ops.solve_auxiliary(u, ops.assemble_rhs(case.bc_values))
    evaluation = evaluate_global_residual(ops, transfer, case.material_points, old, np.r_[u, r, p], case.bc_values)
    callback = AnalyticalJacobian(ops, transfer, case.material_points, old)

    def forbidden(self: MaterialPoint, increment: strain) -> tuple[MaterialState, None]:
        raise AssertionError("Analytical assembly must reuse the evaluated material state")

    monkeypatch.setattr(MaterialPoint, "update", forbidden)
    matrix = callback(evaluation.trial.x, evaluation)
    assert np.all(np.isfinite(matrix.data))
    assert all(point.trial is None for point in case.material_points)


def test_invalid_local_tangent_does_not_change_histories() -> None:
    case = PlaneStrainTpsa(cells_per_axis=1)
    case.prepare_simulation()
    point = case.material_points[0]
    old = MaterialState()
    increment = np.diag([0.004, 0.0, 0.0])
    returned = update_at(point, old, increment)
    saved = returned.copy()
    point.H_law = lambda alpha, parameters: (0.0, np.nan)
    with pytest.raises(ValueError, match="consistency denominator"):
        analytical_material_tangent(point, old, increment, returned=returned)
    assert returned.alpha == saved.alpha and old.alpha == 0 and point.trial is None
    np.testing.assert_array_equal(returned.stress.to_numpy(), saved.stress.to_numpy())
    with pytest.raises(ValueError, match="finite symmetric strain increment"):
        analytical_material_tangent(point, old, np.full((3, 3), np.nan))


def test_both_jacobians_reproduce_the_same_loading_history() -> None:
    analytical = run_example(cells_per_axis=2, jacobian="analytical")
    numerical = run_example(cells_per_axis=2, jacobian="finite-difference")
    for exact, approximate in zip(analytical.controller.steps, numerical.controller.steps, strict=True):
        assert exact.load_factor == approximate.load_factor
        np.testing.assert_allclose(exact.state.x[:8], approximate.state.x[:8], rtol=1e-8, atol=1e-12)
        for a, b in zip(exact.state.material_states, approximate.state.material_states, strict=True):
            np.testing.assert_allclose(a.stress.to_numpy(), b.stress.to_numpy(), rtol=1e-8, atol=0.1)
            assert a.alpha == pytest.approx(b.alpha, rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("method", ["analytical", "finite-difference"])
def test_cli_selects_the_jacobian(method: str, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--jacobian", method, "--cells-per-axis", "1", "--loading-steps", "8", "--no-export"])
    output = capsys.readouterr().out
    assert f"{method} Jacobian" in output and "Numerical checks: PASS" in output


def test_unknown_jacobian_is_rejected() -> None:
    with pytest.raises(ValueError, match="jacobian must be"):
        run_example(jacobian="unknown")


class _StretchedPlaneStrainTpsa(PlaneStrainTpsa):
    """Nine rectangular cells with unequal volumes and face areas."""

    def set_geometry(self) -> None:
        super().set_geometry()
        # Stretch the Cartesian nodes, preserving topology and material ordering.
        original = np.linspace(0.0, 1.0, self.cells_per_axis + 1)
        self.grid.nodes[0] = np.interp(self.grid.nodes[0], original, [0.0, 0.15, 0.45, 1.0])
        self.grid.nodes[1] = np.interp(self.grid.nodes[1], original, [0.0, 0.2, 0.7, 1.0])
        self.grid.compute_geometry()


@pytest.mark.parametrize("stretched", [False, True], ids=["uniform-grid", "stretched-grid"])
@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear-hardening", "nonlinear-hardening"])
def test_coupled_and_face_linearization_errors_are_second_order(
    stretched: bool, nonlinear: bool,
) -> None:
    """R(x+h*d)-R(x)-h*J*d must decrease by four when h is halved.

    This checks the analytical derivative against residual VALUES, including
    variable cell history, fixed unequal transfer weights, and all u,r,p columns.
    Face checks separately expose boundary and interior area/orientation errors.
    """
    case_type = _StretchedPlaneStrainTpsa if stretched else PlaneStrainTpsa
    case = case_type(cells_per_axis=3, displacement_gradient=np.array([[0.004, 0.0006], [-0.0002, -0.0003]]))
    case.prepare_simulation()
    for cell, point in enumerate(case.material_points):
        point.parameters = replace(
            point.parameters, theta=0.15 + 0.07 * cell, H_bar=(1 + 0.1 * cell) * 1e9,
            sigma_u=point.parameters.sigma_y + (100 + 10 * cell) * 1e6 if nonlinear else point.parameters.sigma_y,
            delta=80.0 if nonlinear else 0.0,
        )
        if nonlinear:
            point.H_law = nonlinear_kinematic
    ops = TpsaOperators(case)
    transfer = CellToFaceTransfer.from_rule(case.grid, lambda grid, face, cells: np.array([0.25, 0.75]))
    nc, nf = ops.num_cells, ops.num_faces
    u = (case.displacement_gradient @ case.grid.cell_centers[:2]).ravel(order="F")
    r, p = ops.solve_auxiliary(u, ops.assemble_rhs(case.bc_values))
    x_n = np.r_[u, r, p]
    old = evaluate_global_residual(
        ops, transfer, case.material_points, TpsaState.zeros(case.grid), x_n, case.bc_values,
    ).trial
    rng = np.random.default_rng(41)
    x, bc = 1.3 * x_n, 1.3 * case.bc_values
    x[:2 * nc] += rng.normal(size=2 * nc) * 1e-5
    evaluation = evaluate_global_residual(ops, transfer, case.material_points, old, x, bc)
    assert all(a.alpha > b.alpha for a, b in zip(evaluation.trial.material_states, old.material_states, strict=True))
    # Deliberately perturb every independent field; do not solve r,p in this test.
    direction = rng.normal(size=4 * nc) * np.r_[np.full(2 * nc, 0.003), np.full(2 * nc, 6e8)]
    callback = AnalyticalJacobian(ops, transfer, case.material_points, old)
    residual_action = callback(x, evaluation) @ direction
    face_action = (callback.face_jacobian(x, evaluation) @ direction).reshape((2, nf), order="F")
    interior = np.setdiff1d(np.arange(nf), ops.boundary_faces)
    errors = []
    for h in (0.01, 0.005, 0.0025):
        perturbed = evaluate_global_residual(ops, transfer, case.material_points, old, x + h * direction, bc)
        # Exclude yield crossings: Taylor order is asserted only on a smooth branch.
        assert all(a.alpha > b.alpha for a, b in zip(perturbed.trial.material_states, old.material_states, strict=True))
        remainder = perturbed.residual - evaluation.residual - h * residual_action
        face_remainder = perturbed.trial.traction - evaluation.trial.traction - h * face_action
        np.testing.assert_allclose(remainder[2 * nc:], 0.0, atol=1e-16)
        errors.append([
            np.linalg.norm(remainder[:2 * nc]),
            np.linalg.norm(face_remainder[:, interior]),
            np.linalg.norm(face_remainder[:, ops.boundary_faces]),
        ])
    values = np.array(errors)
    # Keep above numerical noise and distinguish O(h^2) from an incorrect O(h) tangent.
    assert np.all(np.isfinite(values)) and np.all(values[-1] > 1e-3)
    ratios = values[:-1] / values[1:]
    np.testing.assert_allclose(ratios, 4.0, rtol=0.08)


def test_partial_analytical_assembly_failure_preserves_all_histories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2, displacement_gradient=np.diag([0.004, 0.0]))
    case.prepare_simulation()
    for point in case.material_points:
        point.update(strain(np.diag([1e-5, 0.0, 0.0])))  # Existing unrelated trial.
    ops, transfer = TpsaOperators(case), CellToFaceTransfer(case.grid)
    old = TpsaState.zeros(case.grid)
    u = (case.displacement_gradient @ case.grid.cell_centers[:2]).ravel(order="F")
    r, p = ops.solve_auxiliary(u, ops.assemble_rhs(case.bc_values))
    evaluation = evaluate_global_residual(ops, transfer, case.material_points, old, np.r_[u, r, p], case.bc_values)
    callback = AnalyticalJacobian(ops, transfer, case.material_points, old)
    expected = callback(evaluation.trial.x, evaluation).toarray()
    old_before, evaluation_before, points_before = old.copy(), deepcopy(evaluation), deepcopy(case.material_points)
    count = 0

    def fail_on_third_cell(
        point: MaterialPoint, committed: MaterialState, increment: NDArray[np.float64],
        *, returned: MaterialState | None = None,
    ) -> NDArray[np.float64]:
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("Injected analytical tangent failure")
        return analytical_material_tangent(point, committed, increment, returned=returned)

    with monkeypatch.context() as patch:
        patch.setattr("coupling.jacobian.analytical_material_tangent", fail_on_third_cell)
        with pytest.raises(RuntimeError, match="Injected analytical tangent failure"):
            callback(evaluation.trial.x, evaluation)
    assert count == 3
    for actual, saved in ((old, old_before), (evaluation.trial, evaluation_before.trial)):
        for name in ("x", "epsilon", "bc_values", "traction"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(saved, name))
    np.testing.assert_array_equal(evaluation.residual, evaluation_before.residual)
    np.testing.assert_array_equal(evaluation.stress_correction, evaluation_before.stress_correction)
    histories = list(zip(old.material_states, old_before.material_states, strict=True))
    histories.extend(zip(evaluation.trial.material_states, evaluation_before.trial.material_states, strict=True))
    for point, saved_point in zip(case.material_points, points_before, strict=True):
        histories.append((point.committed, saved_point.committed))
        assert point.trial is not None and saved_point.trial is not None
        histories.append((point.trial, saved_point.trial))
    for actual_history, saved_history in histories:
        assert actual_history.alpha == saved_history.alpha
        for name in ("stress", "plastic_strain", "backstress"):
            np.testing.assert_array_equal(getattr(actual_history, name).to_numpy(), getattr(saved_history, name).to_numpy())
    # A failed call must not poison the callback or its cached operators/history.
    np.testing.assert_array_equal(callback(evaluation.trial.x, evaluation).toarray(), expected)
