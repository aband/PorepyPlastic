"""Nested retained-face derivatives against independent residual perturbations."""

from copy import deepcopy
from dataclasses import replace
from typing import TypeAlias

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.params.bc import BoundaryConditionVectorial

from utils.reconstruction import mandel
from alternatives.plane_strain import PlaneStrainTpsa
from utils.state import TpsaState
from material.material_state import MaterialPoint, MaterialState
from core.jacobian import RetainedAnalyticalJacobian, RetainedFiniteDifferenceJacobian
from core.operators import RetainedFaceOperators
from core.residual import (
    RetainedFaceResidual, assemble_global_residual, evaluate_global_residual, evaluate_one_sided_traction,
)
from material.tensor import strain

FloatArray: TypeAlias = NDArray[np.float64]
Setup: TypeAlias = tuple[PlaneStrainTpsa, RetainedFaceOperators, TpsaState]


def make_setup(*, single_cell: bool = False, angle: float = 0.37) -> Setup:
    case = PlaneStrainTpsa(cells_per_axis=1 if single_cell else 2)
    case.prepare_simulation()
    g = case.grid
    g.nodes[0] = 1.6 * g.nodes[0] ** 1.4
    g.nodes[1] = 0.7 * g.nodes[1] ** 1.2
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    g.nodes[:2] = rotation @ g.nodes[:2]
    g.compute_geometry()
    for cell, point in enumerate(case.material_points):
        point.material = deepcopy(point.material)
        point.material.young_modulus *= 0.7 + 0.3 * cell
    mu = np.array([p.material.isotropic_shear_modulus for p in case.material_points])
    lam = np.array([p.material.lame_parameter for p in case.material_points])
    ops = RetainedFaceOperators(g, case.bc, mu=mu, lam=lam)
    return case, ops, TpsaState.zeros(g)


def candidate(ops: RetainedFaceOperators, old: TpsaState, gradient: FloatArray) -> tuple[FloatArray, FloatArray]:
    assert ops.mu is not None and ops.lam is not None
    u, d = gradient @ ops.grid.cell_centers[:2], gradient @ ops.grid.face_centers[:2]
    cell = np.r_[u.ravel("F"), ops.mu * (gradient[0, 1] - gradient[1, 0]), ops.lam * np.trace(gradient)]
    return ops.pack(cell, d[:, ops.free_faces]), old.bc_values + d


def assert_blocks_close(
    actual: FloatArray, expected: FloatArray, ops: RetainedFaceOperators, *, force_atol: float = 1.0,
) -> None:
    for block in (ops.u_slice, ops.face_slice):
        np.testing.assert_allclose(actual[block], expected[block], rtol=4e-6, atol=force_atol)
    for block in (ops.r_slice, ops.p_slice):
        np.testing.assert_allclose(actual[block], expected[block], rtol=2e-7, atol=1e-11)


@pytest.mark.parametrize("single_cell", [False, True])
@pytest.mark.parametrize("angle", [0.0, 0.37])
def test_sparse_fixed_maps_reproduce_kinematics_traction_and_residual(single_cell: bool, angle: float) -> None:
    case, ops, old = make_setup(single_cell=single_cell, angle=angle)
    jac = RetainedAnalyticalJacobian(ops, case.material_points, old)
    rng = np.random.default_rng(12)
    z = rng.normal(size=ops.num_dofs) * np.r_[np.full(2 * ops.num_cells, 0.003),
        np.full(2 * ops.num_cells, 1e8), np.full(2 * ops.num_free_faces, 0.003)]
    boundary = rng.normal(scale=0.001, size=(2, ops.num_faces))
    q = rng.normal(scale=1e7, size=(3, 3, ops.num_cells))
    q = 0.5 * (q + q.swapaxes(0, 1))
    zero_z, zero_q = np.zeros_like(z), np.zeros_like(q)
    t0 = evaluate_one_sided_traction(ops, zero_z, boundary, old.bc_values, zero_q)
    t = evaluate_one_sided_traction(ops, z, boundary, old.bc_values, q)
    np.testing.assert_allclose(t.ravel("F"), t0.ravel("F") + jac.T @ z + jac.P_side @ mandel(q), rtol=1e-12, atol=1e-5)
    r0 = assemble_global_residual(ops, zero_z, boundary, old.bc_values, t0)
    r = assemble_global_residual(ops, z, boundary, old.bc_values, t)
    assert_blocks_close(r, r0 + jac.L @ z + jac.P @ mandel(q), ops)
    e0 = ops.strain_increment(zero_z, boundary, old.bc_values)
    e = ops.strain_increment(z, boundary, old.bc_values)
    np.testing.assert_allclose(mandel(e - e0), jac.B @ z, rtol=1e-12, atol=1e-17)
    assert jac.B[:, ops.cell_slice].nnz == 0
    assert jac.P[ops.cell_slice].nnz == 0
    assert jac.L.shape == (ops.num_dofs, ops.num_dofs)
    for matrix in (jac.B, jac.P, jac.L, jac.T, jac.P_side):
        assert isinstance(matrix, sps.csr_array)
    if single_cell:
        assert jac.B.nnz == 0 and jac.P.nnz == 0


@pytest.mark.parametrize("regime", ["elastic", "plastic", "mixed", "unloading", "reload", "nonlinear"])
@pytest.mark.parametrize("jacobian_type", [RetainedAnalyticalJacobian, RetainedFiniteDifferenceJacobian])
def test_every_residual_and_one_sided_traction_column_matches_finite_differences(
    regime: str, jacobian_type: type[RetainedAnalyticalJacobian],
) -> None:
    case, ops, old = make_setup()
    if regime == "mixed":
        for cell, point in enumerate(case.material_points):
            yield_stress = 100e6 if cell < 2 else 2e9
            point.parameters = replace(point.parameters, sigma_y=yield_stress, sigma_u=yield_stress)
    elif regime == "nonlinear":
        for point in case.material_points:
            point.parameters = replace(point.parameters, sigma_u=450e6, delta=80.0)
    old_imbalance = np.zeros(ops.num_dofs)
    if regime in ("unloading", "reload"):
        initial_z, initial_bc = candidate(ops, old, np.array([[0.004, 0.001], [0., -0.001]]))
        initial = evaluate_global_residual(ops, case.material_points, old, initial_z, initial_bc)
        # A constitutively consistent preload; retain any heterogeneous global imbalance.
        old = TpsaState(initial_z[ops.cell_slice], initial_bc, initial.material_trial.epsilon,
                        initial.material_trial.material_states, np.zeros((2, ops.num_faces)))
        old_imbalance = initial.residual
    gradient = np.array([[0.004, 0.0006], [-0.0002, -0.0003]])
    if regime == "elastic":
        gradient *= 0.01
    elif regime == "unloading":
        gradient = -0.01 * np.array([[0.004, 0.001], [0., -0.001]])
    z, boundary = candidate(ops, old, gradient)
    body = np.arange(2 * ops.num_cells, dtype=float).reshape((2, -1)) * 1e4

    def evaluate(values: FloatArray) -> RetainedFaceResidual:
        return evaluate_global_residual(ops, case.material_points, old, values, boundary,
                                        body_force_increment=body, committed_residual=old_imbalance)

    evaluation = evaluate(z)
    active = [a.alpha > b.alpha for a, b in zip(evaluation.material_trial.material_states, old.material_states, strict=True)]
    assert active == ([False] * 4 if regime in ("elastic", "unloading") else
                      [True, True, False, False] if regime == "mixed" else [True] * 4)
    jac = jacobian_type(ops, case.material_points, old)
    matrix, side = jac(z, evaluation), jac.face_jacobian(z, evaluation)
    assert matrix.shape == (ops.num_dofs, ops.num_dofs)
    assert side.shape == (2 * ops.num_sides, ops.num_dofs)
    scales = np.r_[np.full(2 * ops.num_cells, 0.003), np.full(2 * ops.num_cells, 1e8),
                   np.full(2 * ops.num_free_faces, 0.003)]
    # Subtracting O(1e9) stresses with local h=1e-10 introduces O(1e3)
    # tangent roundoff. After the displacement scaling, allow 10 force units
    # near exact zeros (typical nonzero actions are O(1e9)).
    force_atol = 10.0 if jacobian_type is RetainedFiniteDifferenceJacobian else 1.0
    h = 1e-6
    for column in range(ops.num_dofs):
        direction = np.zeros(ops.num_dofs)
        direction[column] = scales[column]
        plus, minus = evaluate(z + h * direction), evaluate(z - h * direction)
        expected = (plus.residual - minus.residual) / (2 * h)
        expected_side = (plus.traction_increment - minus.traction_increment).ravel("F") / (2 * h)
        assert_blocks_close(matrix @ direction, expected, ops, force_atol=force_atol)
        np.testing.assert_allclose(side @ direction, expected_side, rtol=4e-6, atol=force_atol)
    # Fixed boundary displacement has no DOFs, but its reactions still vary.
    boundary_rows = (2 * np.flatnonzero(np.isin(ops.side_faces, ops.boundary_faces))[:, None] + np.arange(2)).ravel()
    assert side[boundary_rows].nnz > 0
    np.testing.assert_array_equal(matrix[ops.cell_slice].toarray(), jac.L[ops.cell_slice].toarray())
    np.testing.assert_array_equal(matrix[:, ops.cell_slice].toarray(), jac.L[:, ops.cell_slice].toarray())
    if regime in ("plastic", "mixed", "reload", "nonlinear"):
        assert np.linalg.norm((side - jac.T)[boundary_rows].data) > 1e8
        assert np.linalg.norm((matrix - jac.L).data) > 1e8
    elif jacobian_type is RetainedAnalyticalJacobian:
        assert jac.plastic_tangent(z, evaluation).nnz == 0
        np.testing.assert_array_equal(matrix.toarray(), jac.L.toarray())


def test_single_cell_plastic_candidate_has_no_unknown_strain_variations() -> None:
    case, ops, old = make_setup(single_cell=True)
    z, boundary = candidate(ops, old, np.diag([0.004, 0.]))
    evaluation = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    assert evaluation.material_trial.material_states[0].alpha > 0
    jac = RetainedAnalyticalJacobian(ops, case.material_points, old)
    assert jac.plastic_tangent(z, evaluation).nnz > 0 and jac.B.nnz == 0
    np.testing.assert_array_equal(jac(z, evaluation).toarray(), jac.L.toarray())
    np.testing.assert_array_equal(jac.face_jacobian(z, evaluation).toarray(), jac.T.toarray())


def test_analytical_assembly_reuses_returned_histories_and_preserves_data(monkeypatch: pytest.MonkeyPatch) -> None:
    case, ops, old = make_setup()
    for point in case.material_points:
        point.update(strain(np.diag([0.002, 0., 0.])))
    point_snapshots = [(p.committed.copy(), deepcopy(p.trial)) for p in case.material_points]
    z, boundary = candidate(ops, old, np.diag([0.004, 0.]))
    evaluation = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    saved = deepcopy(evaluation)
    jac = RetainedAnalyticalJacobian(ops, case.material_points, old)

    def forbidden(self: MaterialPoint, increment: strain) -> tuple[MaterialState, None]:
        raise AssertionError("Analytical Jacobian must reuse the residual's local roots")

    monkeypatch.setattr(MaterialPoint, "update", forbidden)
    for array in (z, evaluation.z, evaluation.bc_values, evaluation.material_trial.epsilon,
                  evaluation.material_trial.stress_correction, evaluation.residual, evaluation.traction_increment):
        array.flags.writeable = False
    first, side = jac(z, evaluation), jac.face_jacobian(z, evaluation)
    first.data[:] = side.data[:] = 0
    assert jac(z, evaluation).nnz > 0 and jac.face_jacobian(z, evaluation).nnz > 0
    for name in ("residual", "traction_increment", "z", "bc_values"):
        np.testing.assert_array_equal(getattr(evaluation, name), getattr(saved, name))
    for point, (committed, trial) in zip(case.material_points, point_snapshots, strict=True):
        assert point.trial is not None and trial is not None
        for actual, expected in ((point.committed, committed), (point.trial, trial)):
            assert actual.alpha == expected.alpha
            for name in ("stress", "plastic_strain", "backstress"):
                np.testing.assert_array_equal(getattr(actual, name).to_numpy(), getattr(expected, name).to_numpy())
    expected = jac(z, evaluation).toarray()
    old.epsilon[:] = 1  # The callback owns its committed-history snapshot.
    old.material_states[0].alpha = 100
    np.testing.assert_array_equal(jac(z, evaluation).toarray(), expected)


def test_residual_records_owned_candidate_and_masked_boundary_snapshots() -> None:
    case, ops, old = make_setup()
    z, boundary = candidate(ops, old, np.diag([0.004, 0.]))
    saved_z, saved_boundary = z.copy(), boundary.copy()
    boundary[:, ops.free_faces] = np.nan
    evaluation = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    z[:] = 0
    boundary[:] = 0
    np.testing.assert_array_equal(evaluation.z, saved_z)
    np.testing.assert_array_equal(evaluation.bc_values[:, ops.boundary_faces], saved_boundary[:, ops.boundary_faces])
    np.testing.assert_array_equal(evaluation.bc_values[:, ops.free_faces], 0)


@pytest.mark.parametrize("defect", ["z", "z_shape", "z_nan", "strain", "boundary", "history_count"])
def test_rejects_stale_or_invalid_evaluation(defect: str) -> None:
    case, ops, old = make_setup()
    z, boundary = candidate(ops, old, np.diag([0.004, 0.]))
    evaluation = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    jac = RetainedAnalyticalJacobian(ops, case.material_points, old)
    if defect == "z":
        z[-1] += 1e-5
    elif defect == "z_shape":
        z = z[:-1]
    elif defect == "z_nan":
        z[0] = np.nan
    elif defect == "strain":
        evaluation.material_trial.epsilon[0, 0, 0] += 0.001
    elif defect == "boundary":
        evaluation.bc_values[0, ops.boundary_faces[0]] += 0.001
    else:
        evaluation.material_trial.material_states.pop()
    with pytest.raises(ValueError):
        jac(z, evaluation)


def test_jacobian_is_invariant_to_normal_orientation() -> None:
    case, ops, old = make_setup()
    z, boundary = candidate(ops, old, np.diag([0.004, 0.]))
    evaluation = evaluate_global_residual(ops, case.material_points, old, z, boundary)
    jac = RetainedAnalyticalJacobian(ops, case.material_points, old)
    expected, expected_side = jac(z, evaluation).toarray(), jac.face_jacobian(z, evaluation).toarray()
    signs = np.ones(ops.num_faces)
    signs[::2] = -1
    g = ops.grid
    g.face_normals *= signs
    g.cell_faces = sps.csc_matrix(sps.diags(signs) @ g.cell_faces)
    bc = BoundaryConditionVectorial(g, ops.boundary_faces, ["dir"] * len(ops.boundary_faces))
    other = RetainedFaceOperators(g, bc, mu=ops.mu, lam=ops.lam)
    result = evaluate_global_residual(other, case.material_points, old, z, boundary)
    changed = RetainedAnalyticalJacobian(other, case.material_points, old)
    np.testing.assert_array_equal(changed(z, result).toarray(), expected)
    np.testing.assert_array_equal(changed.face_jacobian(z, result).toarray(), expected_side)


@pytest.mark.parametrize("defect", ["missing_lam", "mu_mismatch", "point_count", "history_count", "old_strain"])
def test_rejects_incompatible_setup(defect: str) -> None:
    case, ops, old = make_setup()
    if defect == "missing_lam":
        ops = RetainedFaceOperators(case.grid, case.bc, mu=ops.mu)
    elif defect == "mu_mismatch":
        case.material_points[-1].material.young_modulus *= 1.1
    elif defect == "point_count":
        case.material_points.pop()
    elif defect == "history_count":
        old.material_states.pop()
    else:
        old.epsilon[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        RetainedAnalyticalJacobian(ops, case.material_points, old)


@pytest.mark.parametrize("step", [0.0, -1.0, np.nan, np.inf])
def test_local_finite_difference_step_must_be_positive_and_finite(step: float) -> None:
    case, ops, old = make_setup()
    with pytest.raises(ValueError, match="step"):
        RetainedFiniteDifferenceJacobian(ops, case.material_points, old, step=step)
