"""Predictor and elastic-prefix reuse preserve the existing prototype problem."""

import json
import shutil
import numpy as np
import pytest

import prototype.loading as loading
import prototype.newton as newton
from prototype.cantilever import make_case, run_case
from prototype.cantilever_full_study import (
    FORCE_BALANCE_TOL, boundary_force_imbalance, run, scaled_superlu,
)
from prototype.jacobian import RetainedAnalyticalJacobian
from prototype.residual import assemble_global_residual
from prototype.state import RetainedState
from utils.newton_utils import NewtonConvergenceError
from material.scalar_hardening import HardeningParameters


def test_elastic_prefix_and_predictor_match_original_full_cycle(tmp_path):
    native_newton, native_linear = loading.solve_newton, newton._solve_linear_correction
    original = run_case(make_case(4, 2, 2, grading_ratio=1.), steps=40)
    result = run((4, 2, 2), steps=40, elastic_prefix=8, output=tmp_path)
    np.testing.assert_allclose(result.response[:, :6], original.response[:, :6], rtol=2e-6, atol=2e-6)
    # Compare zero-load roundoff with each block's peak scale: rotation stress
    # is much larger numerically than displacement on this very slender grid.
    ops = original.case.operators
    for block in (ops.u_slice, ops.r_slice, ops.p_slice):
        difference = result.final_state.tpsa.x[block]-original.final_state.tpsa.x[block]
        assert np.linalg.norm(difference) < 2e-7*np.linalg.norm(original.state_A.tpsa.x[block])
    assert all(state.alpha == 0. for state in result.final_state.tpsa.material_states)
    assert loading.solve_newton is native_newton
    assert newton._solve_linear_correction is native_linear
    assert json.loads((tmp_path/'run_status.json').read_text())['status'] == 'complete'
    with np.load(tmp_path/'prototype_checkpoint_latest.npz') as checkpoint:
        assert checkpoint['step'] == 40
        assert checkpoint['load_factor'] == 0.
        np.testing.assert_array_equal(checkpoint['x'], result.final_state.tpsa.x)
        np.testing.assert_array_equal(checkpoint['response'], result.response)


def test_failed_step_restores_functions_and_records_status(tmp_path, monkeypatch):
    native_newton, native_linear = loading.solve_newton, newton._solve_linear_correction
    def fail(*args, **kwargs):
        raise RuntimeError('controlled failure')
    monkeypatch.setattr(loading.LoadStepController, 'advance', fail)
    with pytest.raises(RuntimeError, match='controlled failure'):
        run((4, 2, 2), steps=4, elastic_prefix=0, output=tmp_path)
    assert loading.solve_newton is native_newton
    assert newton._solve_linear_correction is native_linear
    status = json.loads((tmp_path/'run_status.json').read_text())
    assert status['status'] == 'failed' and status['pending_step'] == 1


@pytest.mark.parametrize('linear_solver', ['native', 'superlu'])
def test_predictor_preserves_plastic_cycle_and_history(tmp_path, monkeypatch, linear_solver):
    # Lower yield stress only in this small regression case so that its coarse
    # spatial discretization activates plasticity without a large test mesh.
    def plastic_case(*shape, **kwargs):
        case = make_case(*shape, **kwargs)
        for point in case.material_points:
            point.parameters = HardeningParameters(100., 100., 7000., .5, 0.)
        return case

    monkeypatch.setattr('prototype.cantilever_full_study.make_case', plastic_case)
    # Use native full-step Newton for this reference: the zero-increment
    # initial merit can stall backtracking at first yield even on this grid.
    with monkeypatch.context() as reference_options:
        reference_options.setattr('prototype.cantilever.BacktrackingOptions', lambda **kwargs: None)
        original = run_case(plastic_case(4, 2, 2, grading_ratio=1.), steps=40)
    result = run((4, 2, 2), steps=40, elastic_prefix=0, linear_solver=linear_solver, output=tmp_path)
    assert result.response[:, 4].max() > 0.
    np.testing.assert_allclose(result.response[:, :6], original.response[:, :6], rtol=2e-5, atol=2e-7)
    for reference, actual in zip(original.final_state.tpsa.material_states,
                                 result.final_state.tpsa.material_states, strict=True):
        assert actual.alpha == pytest.approx(reference.alpha, rel=2e-5, abs=1e-10)
        np.testing.assert_allclose(actual.stress.to_numpy(), reference.stress.to_numpy(), atol=2e-4)
        np.testing.assert_allclose(actual.plastic_strain.to_numpy(), reference.plastic_strain.to_numpy(), atol=1e-9)


def test_checkpoint_resume_preserves_cycle(tmp_path):
    complete = run((4, 2, 2), steps=4, elastic_prefix=0, linear_solver='superlu', output=tmp_path)
    shutil.copyfile(tmp_path/'prototype_checkpoint_A.npz', tmp_path/'prototype_checkpoint_latest.npz')
    resumed = run((4, 2, 2), steps=4, elastic_prefix=0, resume=True, linear_solver='superlu', output=tmp_path)
    np.testing.assert_allclose(resumed.response, complete.response, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(resumed.final_state.tpsa.x, complete.final_state.tpsa.x, rtol=0., atol=1e-12)


def test_global_balance_requires_correction_even_when_block_norms_pass():
    case = make_case(32, 2, 2, grading_ratio=1.)
    ops = case.operators
    old = RetainedState.zeros(ops)
    jacobian = RetainedAnalyticalJacobian(ops, case.material_points, old.tpsa)
    interior = ~np.isin(ops.free_faces, ops.boundary_faces)
    # Coherent, individually small errors can have a global sum above 2e-5 N.
    defect = np.zeros(ops.num_dofs)
    defect[ops.u_slice].reshape((3, ops.num_cells), order='F')[2] = 1.5e-6/np.sqrt(ops.num_cells)
    defect[ops.face_slice].reshape((3, ops.num_free_faces), order='F')[2, interior] = -1.5e-6/np.sqrt(interior.sum())
    initial = scaled_superlu(jacobian.L, -defect)
    arguments = (ops, case.material_points, old.tpsa, case.boundary, jacobian)
    options = dict(initial_z=initial, atol=(2e-6, 2e-7, 2e-7, 2e-6), rtol=(0.,)*4,
                   line_search=newton.BacktrackingOptions(), max_iterations=3)
    unchecked = newton.solve_newton(*arguments, **options)
    assert unchecked.iterations == 0
    assert boundary_force_imbalance(ops, unchecked.evaluation.residual) > FORCE_BALANCE_TOL
    boundary = np.isin(ops.side_faces, ops.boundary_faces)
    actual = np.linalg.norm(unchecked.evaluation.traction_increment[:, boundary].sum(axis=1))
    assert boundary_force_imbalance(ops, unchecked.evaluation.residual) == pytest.approx(actual, abs=1e-13)

    check = lambda evaluation: boundary_force_imbalance(ops, evaluation.residual) <= FORCE_BALANCE_TOL
    checked = newton.solve_newton(*arguments, **options, convergence_check=check)
    assert checked.iterations >= 1
    assert check(checked.evaluation)
    assert np.linalg.norm(checked.evaluation.traction_increment[:, boundary].sum(axis=1)) <= FORCE_BALANCE_TOL
    assert np.all(checked.residual_norms[-1] <= checked.thresholds)
    assert all(h.alpha == 0. for h in checked.evaluation.material_trial.material_states)
    # A failed extra criterion must also prevent an immediate zero-iteration return.
    with pytest.raises(NewtonConvergenceError):
        newton.solve_newton(*arguments, **{**options, 'max_iterations': 0}, convergence_check=check)


def test_boundary_force_from_residual_excludes_neumann_residual_rows():
    case = make_case(4, 2, 2, grading_ratio=1.)
    ops = case.operators
    rng = np.random.default_rng(18)
    old_traction = rng.normal(size=(3, ops.num_sides))
    increment = rng.normal(size=old_traction.shape)
    z = np.zeros(ops.num_dofs)
    old_residual = assemble_global_residual(
        ops, z, case.boundary, case.boundary, old_traction,
        neumann_increment=.4*case.reference_neumann,
    )
    total = assemble_global_residual(
        ops, z, case.boundary, case.boundary, increment,
        neumann_increment=.2*case.reference_neumann, committed_residual=old_residual,
    )
    boundary = np.isin(ops.side_faces, ops.boundary_faces)
    expected = np.linalg.norm((old_traction+increment)[:, boundary].sum(axis=1))
    assert boundary_force_imbalance(ops, total) == pytest.approx(expected, abs=1e-12)
