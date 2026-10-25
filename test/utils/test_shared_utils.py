"""Shared problem preparation and solver independence."""

from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.typing import NDArray
import pytest

from alternatives.plane_strain import PlaneStrainTpsa
from prototype.runner import prepare_retained_solver
from utils.problem import PlaneStrainProblem


def test_shared_imports_do_not_load_solver_implementations() -> None:
    # Use a fresh interpreter: collection has already imported solver modules.
    script = """
import importlib
import sys

for module in (
    'material.tangent', 'utils.problem', 'utils.state', 'utils.material_evaluation',
    'utils.reconstruction', 'utils.transfer', 'utils.load_schedule',
    'utils.newton_utils', 'utils.visualization', 'utils.formulation',
):
    importlib.import_module(module)
assert not any(name.split('.')[0] in ('alternatives', 'prototype') for name in sys.modules)
from prototype.runner import prepare_retained_solver
assert not any(name.split('.')[0] == 'alternatives' for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("jacobian", ["analytical", "finite-difference"])
def test_retained_problem_without_tpsa_matrices_matches_existing_setup(jacobian: str) -> None:
    gradient = np.array([[0.004, 0.0], [0.0, 0.0]])
    problem = PlaneStrainProblem(cells_per_axis=2, displacement_gradient=gradient)
    baseline = PlaneStrainTpsa(cells_per_axis=2, displacement_gradient=gradient)
    problem.prepare_simulation()
    baseline.prepare_simulation()
    for name in ("matrices", "face_discretization", "rhs_matrix", "div", "accum", "d"):
        assert not hasattr(problem, name)
        assert hasattr(baseline, name)
    np.testing.assert_array_equal(problem.bc.is_dir, baseline.bc.is_dir)
    np.testing.assert_array_equal(problem.bc_values, baseline.bc_values)
    actual = prepare_retained_solver(problem, jacobian).controller.run([0.1, 0.5, 1.0, 0.98])
    expected = prepare_retained_solver(baseline, jacobian).controller.run([0.1, 0.5, 1.0, 0.98])
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a.state.tpsa.x, b.state.tpsa.x)
        np.testing.assert_array_equal(a.state.tpsa.epsilon, b.state.tpsa.epsilon)
        np.testing.assert_array_equal(a.state.side_traction, b.state.side_traction)
        np.testing.assert_array_equal(a.newton.residual_norms, b.newton.residual_norms)
        for left, right in zip(a.state.tpsa.material_states, b.state.tpsa.material_states, strict=True):
            assert left.alpha == right.alpha
            np.testing.assert_array_equal(left.stress.to_numpy(), right.stress.to_numpy())
    peak = np.array([s.alpha for s in actual[-2].state.tpsa.material_states])
    assert np.all(peak > 0)
    np.testing.assert_allclose([s.alpha for s in actual[-1].state.tpsa.material_states], peak, atol=1e-12)
    assert all(point.committed.alpha == 0 and point.trial is None for point in problem.material_points)


def test_shared_problem_prepares_boundary_callback_once_and_refreshes_histories() -> None:
    calls = 0

    def displacement(centers: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return np.vstack((centers[0] * centers[1] * 0.001, np.zeros(centers.shape[1])))

    problem = PlaneStrainProblem(cells_per_axis=2, boundary_displacement=displacement)
    problem.prepare_simulation()
    assert calls == 1
    boundary = problem.grid.get_all_boundary_faces()
    np.testing.assert_array_equal(
        problem.bc_values[0, boundary],
        problem.grid.face_centers[0, boundary] * problem.grid.face_centers[1, boundary] * 0.001,
    )
    np.testing.assert_array_equal(problem.bc_values[:, np.setdiff1d(np.arange(problem.grid.num_faces), boundary)], 0)
    assert np.shares_memory(problem.bound_vec, problem.bc_values)
    with pytest.raises(ValueError, match="Affine reference"):
        problem.reference_strain()
    old_points = problem.material_points
    old_points[0].committed.alpha = 0.1
    assert old_points[1].committed.alpha == 0
    problem.prepare_simulation()
    assert calls == 2 and problem.x is None
    assert problem.material_points[0] is not old_points[0]
    assert all(point.committed.alpha == 0 for point in problem.material_points)
