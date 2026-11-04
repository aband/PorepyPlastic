"""Cartesian-component Dirichlet/Neumann faces, including roller supports."""

import numpy as np
import pytest

from material.tensor import strain
from prototype.jacobian import RetainedAnalyticalJacobian
from prototype.loading import LoadStepController
from prototype.operators import RetainedFaceOperators
from prototype.residual import evaluate_global_residual
from prototype.state import RetainedState, assemble_state_residual
from prototype.tests.test_spatial import affine, gradient, setup
from utils.load_schedule import LoadStepError


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("scale", [1e-4, 0.02], ids=["elastic", "plastic"])
def test_component_boundary_affine_patch_and_analytical_derivative(dim: int, scale: float) -> None:
    problem, old_ops = setup(dim, rotated=False)
    g, bc = problem.grid, problem.bc
    boundary = g.get_all_boundary_faces()
    bc.is_dir[:, boundary] = False
    bc.is_neu[:, boundary] = True
    axes = np.argmax(np.abs(g.face_normals[:dim, boundary]), axis=0)
    bc.is_dir[axes, boundary] = True
    bc.is_neu[axes, boundary] = False
    ops = RetainedFaceOperators(g, bc, mu=old_ops.mu, lam=old_ops.lam)
    grad = gradient(dim, scale)
    z, values = affine(ops, grad)
    eps = np.zeros((3, 3)); eps[:dim, :dim] = (grad + grad.T) / 2
    point = problem.material_points[0]
    returned, _ = point.evaluate_from(point.committed, strain(eps))
    sigma = returned.stress.to_numpy()
    sides = np.searchsorted(ops.side_faces, ops.neumann_faces)
    tractions = (sigma[:dim, :dim] @ ops.side_normals[:, sides]) * ops.side_areas[sides]
    tractions[ops.dirichlet_mask[:, ops.neumann_faces]] = 0.0
    old = RetainedState.zeros(ops)
    trial = evaluate_global_residual(ops, problem.material_points, old.tpsa, z, values,
                                    neumann_increment=tractions)
    np.testing.assert_allclose(trial.residual, 0.0, atol=2e-14)
    jac = RetainedAnalyticalJacobian(ops, problem.material_points, old.tpsa)
    v = np.random.default_rng(12).normal(size=ops.num_dofs)
    h = 1e-7
    plus = evaluate_global_residual(ops, problem.material_points, old.tpsa, z+h*v, values,
                                   neumann_increment=tractions)
    minus = evaluate_global_residual(ops, problem.material_points, old.tpsa, z-h*v, values,
                                    neumann_increment=tractions)
    np.testing.assert_allclose(jac(z, trial) @ v, (plus.residual-minus.residual)/(2*h), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(jac.face_jacobian(z, trial) @ v,
                              ((plus.traction_increment-minus.traction_increment)/(2*h)).ravel("F"),
                              rtol=1e-7, atol=1e-8)
    if scale > 1e-3:
        assert all(s.alpha > 0 for s in trial.material_trial.material_states)
        assert all(p.trial is None and p.committed.alpha == 0.0 for p in problem.material_points)
        return
    controller = LoadStepController(ops, problem.material_points, old, values,
        reference_neumann=tractions, max_iterations=2, atol=(1e-11,)*4, rtol=(1e-10,)*4)
    for factor in (0.5, 1.0, 0.25):
        record = controller.advance(factor)
        assert record.state.tpsa.face_displacement is not None
        np.testing.assert_allclose(record.state.tpsa.x, factor*z[ops.cell_slice], atol=1e-12)
        np.testing.assert_allclose(record.state.tpsa.face_displacement, factor*values, atol=1e-12)
        np.testing.assert_allclose(assemble_state_residual(ops, record.state), 0.0, atol=1e-11)
    assert all(p.trial is None and p.committed.alpha == 0.0 for p in problem.material_points)


def test_component_boundary_failure_preserves_accepted_state() -> None:
    from prototype.strip_footing import make_case
    case = make_case(1, growth=2.0)
    ops = case.operators
    controller = LoadStepController(ops, case.material_points, RetainedState.zeros(ops),
                                    case.bc_values, max_iterations=0)
    with pytest.raises(LoadStepError):
        controller.advance(1.0)
    assert controller.load_factor == 0 and not controller.steps
    np.testing.assert_array_equal(controller.state.tpsa.x, 0.0)
    assert all(s.alpha == 0 for s in controller.state.tpsa.material_states)
