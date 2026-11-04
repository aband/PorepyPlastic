"""Exact bending, elastic Schur recovery, convergence and exported stress fields."""

import csv
import json

import numpy as np
from PIL import Image
import pytest

from prototype.cantilever import make_case
from prototype.elastic_bending import (
    _cell_system,
    _face_variances,
    _solve_banded_system,
    make_elastic_cantilever,
    make_pure_bending_case,
    pure_bending_fields,
    relative_l2,
    section_data,
    solve_elastic,
)
from prototype.elastic_bending_plots import export_results
from prototype.newton import solve_newton
from prototype.state import RetainedState
from utils.newton_utils import _solve_linear_correction


def test_exact_face_averages_reconstruct_three_dimensional_bending_stress():
    case = make_pure_bending_case((6, 4, 4))
    ops, grid = case.operators, case.operators.grid
    faces, _ = pure_bending_fields(grid.face_centers, case.curvature,
                                  variances=_face_variances(grid))
    z = ops.pack(np.zeros(ops.cell_size), faces[:, ops.free_faces])
    eps = ops.strain_increment(z, case.boundary, np.zeros_like(case.boundary))
    expected = np.zeros_like(eps)
    expected[0, 0] = -case.curvature*grid.cell_centers[2]
    expected[1, 1] = expected[2, 2] = -case.poisson*expected[0, 0]
    np.testing.assert_allclose(eps, expected, atol=2e-18)
    stress = 2*ops.mu*eps + ops.lam*np.eye(3)[:, :, None]*np.einsum("iic->c", eps)
    _, exact = pure_bending_fields(grid.cell_centers, case.curvature)
    np.testing.assert_allclose(stress, exact, atol=5e-13)


@pytest.mark.parametrize("problem", ["pure", "cantilever"])
def test_condensed_elastic_solution_matches_full_prototype_newton(problem):
    shape = (4, 2, 2)
    case = make_pure_bending_case(shape) if problem == "pure" else make_elastic_cantilever(shape)
    ops = case.operators
    points = make_case(*shape).material_points  # Same paper material in both geometries.
    committed = RetainedState.zeros(ops).tpsa
    boundary, neumann = case.boundary.copy(), case.neumann.copy()
    result = solve_elastic(case)
    full = solve_newton(ops, points, committed, case.boundary,
                       neumann_increment=case.neumann, max_iterations=2,
                       atol=(1e-7, 1e-9, 1e-9, 1e-7), rtol=(1e-9,)*4)
    assert full.iterations == 1
    expected = ops.pack(result.x, result.face_displacement[:, ops.free_faces])
    # The slender cantilever is ill-conditioned. Compare each physical block
    # in its own units; tiny transverse rotations suffer cancellation noise.
    for block in (ops.u_slice, ops.r_slice, ops.p_slice, ops.face_slice):
        assert np.linalg.norm((full.z-expected)[block]) < 2e-7*np.linalg.norm(expected[block])
    trial = full.evaluation.material_trial
    np.testing.assert_allclose(trial.epsilon, result.strain, rtol=2e-7, atol=2e-12)
    stresses = np.stack([state.stress.to_numpy() for state in trial.material_states], axis=2)
    np.testing.assert_allclose(stresses, result.stress, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(full.evaluation.traction_increment, result.side_traction,
                               rtol=2e-7, atol=2e-7)
    assert all(state.alpha == 0 for state in trial.material_states)
    assert all(state.alpha == 0 for state in committed.material_states)
    np.testing.assert_array_equal(case.boundary, boundary)
    np.testing.assert_array_equal(case.neumann, neumann)
    assert result.metrics["retained_residual_max"] < 1e-7


def test_simultaneous_mesh_refinement_reduces_exact_stress_and_displacement_errors():
    results = [solve_elastic(make_pure_bending_case((4*n, 2*n, 2*n))) for n in (1, 2, 4)]
    for name in ("stress_relative_l2", "displacement_relative_l2", "interior_face_moment_relative_l2"):
        errors = [r.metrics[name] for r in results]
        assert np.all(np.diff(errors) < 0), (name, errors)
        assert errors[-1] < .15
    assert results[-1].metrics["force_imbalance_N"] < 1e-8
    assert results[-1].metrics["moment_imbalance_Nmm"] < 1e-8


def test_elastic_load_scaling_and_reference_tip_units():
    a = solve_elastic(make_elastic_cantilever((8, 2, 4), traction=.03))
    b = solve_elastic(make_elastic_cantilever((8, 2, 4), traction=.06))
    np.testing.assert_allclose(b.x, 2*a.x, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b.stress, 2*a.stress, rtol=1e-10, atol=1e-10)
    assert b.metrics["reference_tip_mm"] == pytest.approx(1.142857142857)
    np.testing.assert_allclose(b.case.neumann.sum(axis=1), [0., 0., .6])


@pytest.mark.parametrize("shape", [(4, 2, 2), (4, 8, 10)])
def test_banded_factorization_preserves_the_elastic_system(shape):
    case = make_elastic_cantilever(shape)
    matrix, rhs = _cell_system(case)
    original_data, original_rhs = matrix.data.copy(), rhs.copy()
    x = _solve_banded_system(case, matrix, rhs)
    direct = _solve_linear_correction(matrix, -rhs, 0)
    ops = case.operators
    # The four-cell-long beam is exceptionally ill-conditioned; different LU
    # orderings agree to about six digits. Also check the unscaled equations.
    for block in (ops.u_slice, ops.r_slice, ops.p_slice):
        assert np.linalg.norm((x-direct)[block]) < 1e-6*np.linalg.norm(direct[block])
    np.testing.assert_allclose(matrix @ x, rhs, atol=2e-7)
    np.testing.assert_array_equal(matrix.data, original_data)
    np.testing.assert_array_equal(rhs, original_rhs)


def test_moment_reference_accounts_for_midpoint_quadrature():
    case = make_pure_bending_case((4, 2, 4))
    ops, grid = case.operators, case.operators.grid
    _, stress = pure_bending_fields(grid.cell_centers, case.curvature)
    _, face_stress = pure_bending_fields(grid.face_centers[:, ops.side_faces], case.curvature)
    traction = np.einsum("ijs,js->is", face_stress, ops.side_normals)*ops.side_areas
    fs, cs = section_data(case, stress, traction)
    np.testing.assert_allclose(fs[:, 1]/fs[:, 2], 1-1/case.shape[2]**2)
    np.testing.assert_allclose(cs[:, 1], cs[:, 3])
    np.testing.assert_allclose(cs[:, 3]/cs[:, 2], 1-1/case.shape[2]**2)


def test_l2_uses_volume_weights_and_tensor_frobenius_norm():
    reference = np.ones((3, 3, 2))
    actual = reference.copy()
    actual[0, 1, 1] += 2
    actual[1, 0, 1] += 2
    assert relative_l2(actual, reference, np.array([1., 3.])) == pytest.approx(np.sqrt(24/36))


@pytest.mark.parametrize("case", [make_elastic_cantilever(traction=.6), make_pure_bending_case(curvature=.01)])
def test_elastic_study_rejects_loading_that_requires_plasticity(case):
    with pytest.raises(ValueError, match="reaches yield"):
        solve_elastic(case)


def test_export_contains_verified_fields_norms_and_pngs(tmp_path):
    results = [("exact_3d", solve_elastic(make_pure_bending_case((4*n, 2*n, 2*n)))) for n in (1, 2)]
    results.append(("axial_2x2", solve_elastic(make_elastic_cantilever((4, 2, 2)))))
    results.append(("cross_section_at_4", solve_elastic(make_elastic_cantilever((4, 4, 4)))))
    paths = export_results(results, tmp_path)
    summary = json.loads(paths["summary"].read_text())
    assert summary["results"][1]["observed_stress_order"] > 0
    assert summary["results"][-1]["paper_elastic_tip_mm"] == pytest.approx(1.13227363)
    with paths["csv"].open() as stream:
        assert len(list(csv.DictReader(stream))) == 4
    with np.load(paths["pure_bending_fields"]) as fields:
        np.testing.assert_allclose(fields["stress"], results[1][1].stress)
        np.testing.assert_array_equal(fields["shape"], (8, 4, 4))
    with np.load(paths["cantilever_fields"]) as fields:
        np.testing.assert_array_equal(fields["shape"], (4, 2, 2))
    with np.load(paths["cantilever_cross_section_fields"]) as fields:
        np.testing.assert_array_equal(fields["shape"], (4, 4, 4))
    for path in paths.values():
        if path.suffix == ".png":
            with Image.open(path) as im:
                im.verify()
