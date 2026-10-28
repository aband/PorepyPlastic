"""Mixed traction boundaries and the paper cantilever's units/history/plots."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import porepy as pp
from PIL import Image
import pytest
import scipy.sparse as sps

from material.tensor import strain
from prototype.cantilever import Grading, load_factors, make_case, mean_tip_displacement, run_case
from prototype.cantilever_plots import DATA, export_results
from prototype.jacobian import RetainedAnalyticalJacobian
from prototype.loading import LoadStepController
from prototype.operators import RetainedFaceOperators
from prototype.residual import evaluate_global_residual
from prototype.state import RetainedState, assemble_state_residual
from prototype.tests.test_spatial import ATOL, affine, gradient, setup
from utils.load_schedule import LoadStepError


@pytest.mark.parametrize("dim", [2, 3])
def test_mixed_neumann_affine_solution_and_load_history(dim: int) -> None:
    problem, old_ops = setup(dim, rotated=False)
    grid = problem.grid
    bf = grid.get_all_boundary_faces()
    fixed = bf[np.isclose(grid.face_centers[0, bf], 0.)]
    ops = RetainedFaceOperators(grid, pp.BoundaryConditionVectorial(grid, fixed, "dir"), mu=old_ops.mu, lam=old_ops.lam)
    g = gradient(dim, 1e-4)
    z, boundary = affine(ops, g)
    eps = np.zeros((3, 3)); eps[:dim, :dim] = (g + g.T) / 2
    stress = problem.material_points[0].material.elastic_tensor.double_contract(strain(eps)).to_numpy()
    sides = np.array([np.flatnonzero(ops.side_faces == f)[0] for f in ops.neumann_faces])
    neumann = stress[:dim, :dim] @ ops.side_normals[:, sides] * ops.side_areas[sides]
    initial = RetainedState.zeros(ops)
    source = neumann.copy()
    controller = LoadStepController(ops, problem.material_points, initial, boundary,
                                    reference_neumann=source, atol=ATOL)
    source[:] = 999.
    for factor in (.3, 1., .5, -.1):
        result = controller.advance(factor)
        current = result.state
        np.testing.assert_allclose(current.tpsa.x, factor * z[ops.cell_slice], atol=1e-10)
        assert current.tpsa.face_displacement is not None
        np.testing.assert_allclose(current.tpsa.face_displacement, factor * boundary, atol=1e-10)
        np.testing.assert_array_equal(current.neumann_values, factor * neumann)
        np.testing.assert_allclose(assemble_state_residual(ops, current), 0., atol=1e-10)
    restarted = LoadStepController(ops, problem.material_points, controller.state, boundary,
        reference_neumann=neumann, initial_load_factor=-.1, atol=ATOL)
    a, b = controller.advance(.1).state, restarted.advance(.1).state
    np.testing.assert_array_equal(a.tpsa.x, b.tpsa.x)
    snapshot = controller.state
    snapshot.neumann_values[:] = 42.
    np.testing.assert_array_equal(controller.state.neumann_values, .1 * neumann)


@pytest.mark.parametrize("dim", [2, 3])
def test_mixed_plastic_jacobian_includes_neumann_rows(dim: int) -> None:
    problem, old_ops = setup(dim, rotated=False)
    g = problem.grid
    bf = g.get_all_boundary_faces()
    bc = pp.BoundaryConditionVectorial(g, bf[np.isclose(g.face_centers[0, bf], 0.)], "dir")
    ops = RetainedFaceOperators(g, bc, mu=old_ops.mu, lam=old_ops.lam)
    old = RetainedState.zeros(ops).tpsa
    z, boundary = affine(ops, gradient(dim, .02))
    neumann = np.full((dim, ops.neumann_faces.size), .01)
    ev = evaluate_global_residual(ops, problem.material_points, old, z, boundary, neumann_increment=neumann)
    jac = RetainedAnalyticalJacobian(ops, problem.material_points, old)
    J = jac(z, ev)
    direction = np.random.default_rng(15).normal(size=ops.num_dofs)
    h = 1e-7
    plus = evaluate_global_residual(ops, problem.material_points, old, z+h*direction, boundary, neumann_increment=neumann)
    minus = evaluate_global_residual(ops, problem.material_points, old, z-h*direction, boundary, neumann_increment=neumann)
    np.testing.assert_allclose(J @ direction, (plus.residual-minus.residual)/(2*h), atol=2e-7, rtol=2e-6)
    neumann_rows = ops.face_dofs[:, ops.neumann_faces].ravel("F")
    assert jac.P[neumann_rows].nnz > 0


def test_neumann_failed_step_preserves_loads_and_history() -> None:
    case = make_case(2, 2, 2)
    initial = RetainedState.zeros(case.operators)
    controller = LoadStepController(case.operators, case.material_points, initial, case.boundary,
                                    reference_neumann=case.reference_neumann, max_iterations=0)
    with pytest.raises(LoadStepError):
        controller.advance(1.)
    assert not controller.steps and controller.load_factor == 0.
    np.testing.assert_array_equal(controller.state.neumann_values, 0.)
    assert all(p.alpha == 0. for p in controller.state.tpsa.material_states)


def test_cantilever_geometry_load_units_and_clamp_rank() -> None:
    case = make_case(4, 2, 2)
    grid, ops = case.grid, case.operators
    np.testing.assert_allclose(np.ptp(grid.nodes, axis=1), [100., 10., 1.])
    np.testing.assert_allclose(grid.face_areas[case.tip_faces].sum(), 10.)
    np.testing.assert_allclose(case.reference_neumann.sum(axis=1), [0., 0., 6.])
    assert ops.dirichlet_faces.size == 4
    assert np.allclose(grid.face_centers[0, ops.dirichlet_faces], 0.)
    widths = np.diff(case.axes[0])
    assert widths[-1] / widths[0] == pytest.approx(8.)
    trace = np.vstack([np.hstack([np.eye(3), ops.rotation_matrix(grid.face_centers[:, f])])
                       for f in ops.dirichlet_faces])
    assert np.linalg.matrix_rank(trace) == 6
    with pytest.raises(ValueError, match="clamp"):
        make_case(10, 1, 6)
    factors = load_factors()
    assert factors.shape == (40,) and factors[9] == 1. and factors[29] == -1. and factors[-1] == 0.
    assert np.max(abs(np.diff(np.r_[0., factors]))) == pytest.approx(.1)


def test_paper_combined_hardening_convention() -> None:
    point = deepcopy(make_case(2, 2, 2).material_points[0])
    returned, _ = point.update(strain(np.diag([.01, -.002, -.003])))
    eps_p = returned.plastic_strain.to_numpy()
    gamma = np.linalg.norm(eps_p)
    assert gamma == pytest.approx(np.sqrt(1.5) * returned.alpha)
    np.testing.assert_allclose(returned.backstress.to_numpy(), (2./3.) * (1.-.5) * 7000. * eps_p)
    stress = returned.stress.to_numpy()
    shifted = stress - np.eye(3)*np.trace(stress)/3. - returned.backstress.to_numpy()
    paper_yield = np.linalg.norm(shifted) - .5 * (2./3.) * 7000. * gamma - np.sqrt(2./3.) * 240.
    assert abs(paper_yield) < 1e-8


@pytest.mark.parametrize("grading", ["clamp", "both-ends"])
def test_paper_reference_and_png_outputs(tmp_path: Path, grading: Grading) -> None:
    reference = np.loadtxt(DATA / "cantilever_paper_figure3.csv", delimiter=",", skiprows=1)
    assert reference.shape == (41, 3)
    np.testing.assert_allclose(reference[1:, 1], .6 * load_factors(), atol=1e-9)
    assert reference[10, 2] == pytest.approx(12.744, abs=.01)
    assert reference[30, 2] == pytest.approx(-12.536, abs=.01)
    result = run_case(make_case(4, 2, 2, grading_ratio=8., grading=grading), steps=4)
    paths = export_results(result, [result], tmp_path)
    pngs = [p for p in paths.values() if p.suffix == ".png"]
    assert len(pngs) == 8
    for p in pngs:
        with Image.open(p) as im:
            im.verify()
    summary = json.loads(paths["summary"].read_text())
    assert summary["boundary_conditions"].startswith("x=0")
    assert summary["maximum_force_imbalance_N"] < 2e-5
    assert summary["paper"]["series"] == "Abaqus reference (black curve)"
    assert summary["grading"] == grading
    np.testing.assert_array_equal(summary["x_coordinates_mm"], result.case.axes[0])
    assert mean_tip_displacement(result.case, result.state_A) == result.response[1, 3]


@pytest.mark.parametrize("nx", [3, 4, 20, 21])
def test_both_end_grading_is_symmetric_and_preserves_geometry_and_load(nx: int) -> None:
    case = make_case(nx, 2, 2, grading="both-ends")
    x = case.axes[0]
    widths = np.diff(x)
    np.testing.assert_allclose(x + x[::-1], 100., atol=2e-14)
    np.testing.assert_allclose(widths, widths[::-1], atol=2e-14)
    assert x[0] == 0. and x[-1] == pytest.approx(100.)
    assert np.all(widths > 0.)
    assert widths.max() / widths.min() == pytest.approx(8.)
    half = widths[:(nx + 1)//2]
    assert np.all(np.diff(half) > 0.)
    np.testing.assert_allclose(half[1:] / half[:-1], (half[1:] / half[:-1])[0])
    assert case.grid.cell_volumes.sum() == pytest.approx(1000.)
    np.testing.assert_allclose(case.reference_neumann.sum(axis=1), [0., 0., 6.])
    assert case.operators.dirichlet_faces.size == 4


def test_grading_selection_uniform_limit_and_existing_default() -> None:
    original = make_case(5, 2, 2)
    explicit = make_case(5, 2, 2, grading="clamp")
    np.testing.assert_array_equal(original.axes[0], explicit.axes[0])
    for nx in (2, 5, 6):
        uniform = make_case(nx, 2, 2, grading="both-ends", grading_ratio=1.)
        np.testing.assert_allclose(np.diff(uniform.axes[0]), 100. / nx)
    with pytest.raises(ValueError, match="at least three"):
        make_case(2, 2, 2, grading="both-ends")
    with pytest.raises(ValueError, match="grading must be"):
        make_case(4, 2, 2, grading="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize("dim", [2, 3])
def test_mixed_elastic_schur_matches_porepy(dim: int) -> None:
    problem, old_ops = setup(dim, rotated=False, heterogeneous=True)
    grid = problem.grid
    bf = grid.get_all_boundary_faces()
    bc = pp.BoundaryConditionVectorial(grid, bf[np.isclose(grid.face_centers[0, bf], 0.)], "dir")
    ops = RetainedFaceOperators(grid, bc, mu=old_ops.mu, lam=old_ops.lam)
    assert ops.mu is not None and ops.lam is not None
    data: dict[str, Any] = {
        pp.PARAMETERS: {"mechanics": {"fourth_order_tensor": pp.FourthOrderTensor(ops.mu, ops.lam), "bc": bc}},
        pp.DISCRETIZATION_MATRICES: {"mechanics": {}},
    }
    pp.Tpsa("mechanics").discretize(grid, data)
    m = data[pp.DISCRETIZATION_MATRICES]["mechanics"]
    nc, nf, nr = grid.num_cells, grid.num_faces, ops.rotation_dim
    flux = sps.block_array([
        [m["stress"], m["stress_rotation"], m["stress_total_pressure"]],
        [m["rotation_displacement"], m["rotation_rotation"], sps.csr_array((nr * nf, nc))],
        [m["solid_mass_displacement"], sps.csr_array((nf, nr * nc)), m["solid_mass_total_pressure"]],
    ], format="csr")
    div = sps.block_diag([grid.divergence(dim=dim), grid.divergence(dim=nr), grid.divergence(dim=1)])
    accum = sps.diags_array(np.r_[np.zeros(dim * nc), np.repeat(grid.cell_volumes / ops.mu, nr), grid.cell_volumes / ops.lam])
    expected = sps.csr_array(div @ flux - accum).toarray()
    jac = RetainedAnalyticalJacobian(ops, problem.material_points, RetainedState.zeros(ops).tpsa)
    L, k = jac.L.toarray(), ops.cell_size
    actual = L[:k, :k] - L[:k, k:] @ np.linalg.solve(L[k:, k:], L[k:, :k])
    np.testing.assert_allclose(actual, expected, atol=5e-13, rtol=3e-13)
