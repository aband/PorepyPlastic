"""Coupled TPSA residual: elastic reference, plastic history, sources, and isolation."""

from copy import deepcopy
from typing import TypeAlias

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.grid import Grid

from coupling.plane_strain import (
    PlaneStrainTpsa, _assemble_matrices, _discretize_get_matrices,
)
from coupling.residual import (
    TpsaOperators, TpsaState, assemble_global_residual,
    evaluate_global_residual,
)
from coupling.transfer import CellToFaceTransfer
from tensor import strain

FloatArray: TypeAlias = NDArray[np.float64]
Setup: TypeAlias = tuple[PlaneStrainTpsa, TpsaOperators, CellToFaceTransfer, TpsaState]


@pytest.fixture
def setup() -> Setup:
    case = PlaneStrainTpsa(cells_per_axis=3)
    case.prepare_simulation()
    return case, TpsaOperators(case), CellToFaceTransfer(case.grid), TpsaState.zeros(case.grid)


def affine_candidate(case: PlaneStrainTpsa, gradient: FloatArray) -> tuple[FloatArray, FloatArray]:
    nc = case.grid.num_cells
    u = (gradient @ case.grid.cell_centers[:2]).ravel(order="F")
    r = np.full(nc, case.material.isotropic_shear_modulus * (gradient[0, 1] - gradient[1, 0]))
    p = np.full(nc, case.material.lame_parameter * np.trace(gradient))
    boundary = np.zeros_like(case.bc_values)
    bf = case.grid.get_all_boundary_faces()
    boundary[:, bf] = gradient @ case.grid.face_centers[:2, bf]
    return np.concatenate((u, r, p)), boundary


def original_elastic_residual(
    case: PlaneStrainTpsa, x: FloatArray, boundary: FloatArray, sources: FloatArray,
) -> FloatArray:
    """Reference the original mixed flux system, independently of residual helpers."""
    flux = case.face_discretization @ x + case.rhs_matrix @ boundary.ravel(order="F")
    return np.asarray(case.div @ flux - case.accum @ x - sources, dtype=np.float64)


def assert_blocks(actual: FloatArray, expected: FloatArray, nc: int) -> None:
    assert actual.shape == expected.shape == (4 * nc,)
    # Momentum and auxiliary blocks have different physical units and scales.
    np.testing.assert_allclose(actual[:2 * nc], expected[:2 * nc], rtol=1e-10, atol=1e-5)
    np.testing.assert_allclose(actual[2 * nc:], expected[2 * nc:], rtol=1e-10, atol=1e-14)


def assert_state_equal(actual: TpsaState, expected: TpsaState) -> None:
    for name in ("x", "bc_values", "epsilon", "traction"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for a, b in zip(actual.material_states, expected.material_states, strict=True):
        assert a.alpha == b.alpha
        for name in ("stress", "plastic_strain", "backstress"):
            np.testing.assert_array_equal(getattr(a, name).to_numpy(), getattr(b, name).to_numpy())


@pytest.mark.parametrize("preloaded", [False, True], ids=["virgin", "elastic-history"])
def test_elastic_candidate_matches_original_coupled_system(setup: Setup, preloaded: bool) -> None:
    case, operators, transfer, committed = setup
    nc = case.grid.num_cells
    if preloaded:
        old_x, old_boundary = affine_candidate(case, np.array([[1e-5, 2e-5], [-1e-5, 3e-5]]))
        committed = evaluate_global_residual(
            operators, transfer, case.material_points, committed, old_x, old_boundary,
        ).trial
    rng = np.random.default_rng(42)
    x = rng.normal(size=4 * nc)
    x[:2 * nc] *= 1e-5
    x[2 * nc:] *= 1e6
    boundary = np.zeros_like(case.bc_values)
    bf = operators.boundary_faces
    xf, yf = case.grid.face_centers[:2, bf]
    boundary[:, bf] = 1e-5 * np.array([xf * xf + xf * yf, yf * yf - xf * yf])
    sources = rng.normal(size=4 * nc)
    sources[:2 * nc] *= 1e5
    sources[2 * nc:] *= 1e-5
    result = evaluate_global_residual(
        operators, transfer, case.material_points, committed, x, boundary, sources=sources,
    )
    np.testing.assert_array_equal(result.stress_correction, 0.0)
    assert_blocks(result.residual, original_elastic_residual(case, x, boundary, sources), nc)
    assert case.x is None


@pytest.mark.parametrize("cells_per_axis", [1, 3])
def test_original_elastic_solution_has_zero_residual(cells_per_axis: int) -> None:
    case = PlaneStrainTpsa(
        cells_per_axis=cells_per_axis,
        displacement_gradient=np.array([[1e-4, 2e-4], [-3e-4, 4e-4]]),
    )
    x = case.solve()
    result = evaluate_global_residual(
        TpsaOperators(case), CellToFaceTransfer(case.grid), case.material_points,
        TpsaState.zeros(case.grid), x, case.bc_values,
    )
    assert_blocks(result.residual, np.zeros_like(x), case.grid.num_cells)
    np.testing.assert_array_equal(result.trial.x, x)
    np.testing.assert_allclose(
        result.trial.traction,
        case.reference_stress()[:2, :2] @ case.grid.face_normals[:2], rtol=1e-10, atol=1e-5,
    )


@pytest.mark.parametrize("field", ["r", "p"])
def test_auxiliary_imbalances_remain_in_the_coupled_residual(
    setup: Setup, field: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, operators, transfer, committed = setup
    nc = case.grid.num_cells
    x, boundary = affine_candidate(case, np.array([[1e-4, 2e-4], [-1e-4, 1e-4]]))
    block = slice(2 * nc, 3 * nc) if field == "r" else slice(3 * nc, 4 * nc)
    x[block.start] += 1e6
    saved_x = x.copy()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("The global residual must retain the supplied r and p.")

    monkeypatch.setattr(operators, "solve_auxiliary", forbidden)
    result = evaluate_global_residual(operators, transfer, case.material_points, committed, x, boundary)
    expected = original_elastic_residual(case, x, boundary, np.zeros_like(x))
    assert_blocks(result.residual, expected, nc)
    assert np.linalg.norm(result.residual[block]) > 1e-8
    np.testing.assert_array_equal(result.trial.x, saved_x)
    np.testing.assert_array_equal(x, saved_x)
    assert result.residual.size == 4 * nc  # No displacement rows removed on Dirichlet cells.


def test_integrated_sources_are_subtracted_once_without_volume_scaling(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    x, boundary = affine_candidate(case, np.array([[0.004, 0.001], [0.0, 0.0]]))
    unloaded = evaluate_global_residual(operators, transfer, case.material_points, committed, x, boundary)
    nc = case.grid.num_cells
    sources = np.arange(1, 4 * nc + 1, dtype=float)
    sources[:2 * nc] *= 1e4
    sources[2 * nc:] *= 1e-6
    saved_sources = sources.copy()
    sources.flags.writeable = False
    loaded = evaluate_global_residual(
        operators, transfer, case.material_points, committed, x, boundary, sources=sources,
    )
    assert_blocks(loaded.residual, unloaded.residual - sources, nc)
    assert_state_equal(loaded.trial, unloaded.trial)
    np.testing.assert_array_equal(sources, saved_sources)


def test_low_level_assembly_uses_supplied_traction(setup: Setup) -> None:
    case, operators, _, _ = setup
    nc, nf = case.grid.num_cells, case.grid.num_faces
    rng = np.random.default_rng(12)
    x = rng.normal(size=4 * nc)
    x[:2 * nc] *= 1e-4
    x[2 * nc:] *= 1e7
    boundary = case.bc_values.copy()
    flux = case.face_discretization @ x + case.rhs_matrix @ boundary.ravel(order="F")
    traction = rng.normal(size=(2, nf)) * 1e6
    flux[:2 * nf] = traction.ravel(order="F")
    sources = np.asarray(case.div @ flux - case.accum @ x, dtype=np.float64)
    result = assemble_global_residual(operators, x, boundary, traction, sources=sources)
    assert_blocks(result, np.zeros(4 * nc), nc)
    # The supplied nonlinear traction is intentionally different from the elastic one.
    assert np.linalg.norm(original_elastic_residual(case, x, boundary, sources)[:2 * nc]) > 1e6


@pytest.mark.parametrize("factor", [1.0, 0.99, 1.2], ids=["hold", "unload", "reload"])
def test_homogeneous_plastic_patch_remains_in_equilibrium(setup: Setup, factor: float) -> None:
    case, operators, transfer, virgin = setup
    gradient = np.array([[0.004, 0.001], [0.0, -0.001]])
    x_n, g_n = affine_candidate(case, gradient)
    first = evaluate_global_residual(operators, transfer, case.material_points, virgin, x_n, g_n)
    assert_blocks(first.residual, np.zeros_like(x_n), case.grid.num_cells)
    committed = first.trial
    saved = committed.copy()
    assert committed.material_states[0].alpha > 0.0
    x, boundary = affine_candidate(case, factor * gradient)
    result = evaluate_global_residual(operators, transfer, case.material_points, committed, x, boundary)
    assert_blocks(result.residual, np.zeros_like(x), case.grid.num_cells)
    stress = result.trial.material_states[0].stress.to_numpy()[:2, :2]
    np.testing.assert_allclose(result.trial.traction, stress @ case.grid.face_normals[:2], atol=1e-5)
    if factor <= 1.0:
        np.testing.assert_allclose(result.stress_correction, 0.0, atol=1e-5)
    else:
        assert result.trial.material_states[0].alpha > committed.material_states[0].alpha
    assert_state_equal(committed, saved)


@pytest.mark.parametrize("biased", [False, True])
def test_nonuniform_plasticity_changes_momentum_but_retains_auxiliary_equations(
    setup: Setup, biased: bool,
) -> None:
    case, operators, transfer, committed = setup
    if biased:
        def rule(grid: Grid, face: int, cells: NDArray[np.int64]) -> FloatArray:
            return np.array([0.25, 0.75])
        transfer = CellToFaceTransfer.from_rule(case.grid, rule)
    nc, nf = case.grid.num_cells, case.grid.num_faces
    rng = np.random.default_rng(7)
    x = rng.normal(size=4 * nc)
    x[:2 * nc] *= 0.004
    x[2 * nc:] *= 1e8
    result = evaluate_global_residual(operators, transfer, case.material_points, committed, x, case.bc_values)
    assert max(history.alpha for history in result.trial.material_states) > 0.0
    assert np.ptp(result.stress_correction[0, 0]) > 1e6
    correction_force = np.zeros((2, nf))
    incidence = case.grid.cell_faces.tocsr()
    for face in range(nf):
        cells = np.sort(incidence.indices[incidence.indptr[face]:incidence.indptr[face + 1]])
        weights = np.ones(1) if cells.size == 1 else np.array([0.25, 0.75]) if biased else np.full(2, 0.5)
        correction = np.zeros((2, 2))
        for weight, cell in zip(weights, cells, strict=True):
            correction += weight * result.stress_correction[:2, :2, cell]
        correction_force[:, face] = correction @ case.grid.face_normals[:2, face]
    expected = original_elastic_residual(case, x, case.bc_values, np.zeros(4 * nc))
    correction_balance = case.grid.divergence(dim=2) @ correction_force.ravel(order="F")
    assert np.linalg.norm(correction_balance) > 1e6
    expected[:2 * nc] -= correction_balance
    assert_blocks(result.residual, expected, nc)


def test_trials_are_independent_and_no_candidate_is_committed(setup: Setup) -> None:
    case, operators, transfer, virgin = setup
    x_n, g_n = affine_candidate(case, np.array([[0.004, 0.0], [0.0, 0.0]]))
    committed = evaluate_global_residual(operators, transfer, case.material_points, virgin, x_n, g_n).trial
    saved = committed.copy()
    x, boundary = affine_candidate(case, np.array([[0.00396, 0.0], [0.0, 0.0]]))
    arrays = (x, boundary, committed.x, committed.bc_values, committed.epsilon, committed.traction)
    originals = [value.copy() for value in arrays]
    for value in arrays:
        value.flags.writeable = False
    for point in case.material_points:
        point.update(strain(np.diag([1e-4, 0.0, 0.0])))
    saved_points = deepcopy(case.material_points)
    first = evaluate_global_residual(operators, transfer, case.material_points, committed, x, boundary)
    first_saved = deepcopy(first)
    rejected_x, rejected_g = affine_candidate(case, np.array([[0.008, 0.003], [0.0, 0.0]]))
    rejected = evaluate_global_residual(
        operators, transfer, case.material_points, committed, rejected_x, rejected_g,
    )
    assert rejected.trial.material_states[0].alpha > committed.material_states[0].alpha
    repeated = evaluate_global_residual(operators, transfer, case.material_points, committed, x, boundary)
    for result in (first, repeated):
        assert_state_equal(result.trial, first_saved.trial)
        np.testing.assert_array_equal(result.residual, first_saved.residual)
        np.testing.assert_array_equal(result.stress_correction, first_saved.stress_correction)
    first.trial.x[:] = 0.0
    first.trial.epsilon[:] = 0.0
    first.trial.traction[:] = 0.0
    first.trial.material_states[0].alpha = -1.0
    first.residual[:] = 1.0
    first.stress_correction[:] = 1.0
    assert_state_equal(repeated.trial, first_saved.trial)
    assert_state_equal(committed, saved)
    for value, original in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(value, original)
    for point, original_point in zip(case.material_points, saved_points, strict=True):
        assert point.committed.alpha == original_point.committed.alpha == 0.0
        assert point.trial is not None and original_point.trial is not None
        np.testing.assert_array_equal(point.trial.stress.to_numpy(), original_point.trial.stress.to_numpy())


def test_boundary_storage_is_canonical_and_ignores_unused_entries(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    x, boundary = affine_candidate(case, np.array([[1e-4, 0.0], [0.0, 0.0]]))
    expected = evaluate_global_residual(operators, transfer, case.material_points, committed, x, boundary)
    interior = np.setdiff1d(np.arange(case.grid.num_faces), operators.boundary_faces)
    boundary[:, interior] = np.nan
    committed.bc_values[:, interior] = np.inf
    actual = evaluate_global_residual(operators, transfer, case.material_points, committed, x, boundary)
    np.testing.assert_array_equal(actual.residual, expected.residual)
    assert_state_equal(actual.trial, expected.trial)
    np.testing.assert_array_equal(actual.trial.bc_values[:, interior], 0.0)
    assert np.all(np.isnan(boundary[:, interior]))
    assert np.all(np.isinf(committed.bc_values[:, interior]))


def test_local_failure_cannot_partially_commit_global_state(
    setup: Setup, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, operators, transfer, committed = setup
    saved = committed.copy()
    x, boundary = affine_candidate(case, np.array([[0.004, 0.0], [0.0, 0.0]]))

    def fail(**kwargs: object) -> None:
        raise RuntimeError("Injected return-map failure")

    monkeypatch.setattr(case.material_points[1].model, "radial_return_map", fail)
    with pytest.raises(RuntimeError, match="Injected return-map failure"):
        evaluate_global_residual(operators, transfer, case.material_points, committed, x, boundary)
    assert_state_equal(committed, saved)
    assert all(point.trial is None and point.committed.alpha == 0.0 for point in case.material_points)


@pytest.mark.parametrize("field", ["x", "traction", "sources"])
@pytest.mark.parametrize("defect", ["shape", "nonfinite"])
def test_low_level_assembly_rejects_invalid_arrays(setup: Setup, field: str, defect: str) -> None:
    case, operators, _, committed = setup
    arrays = {"x": committed.x.copy(), "traction": committed.traction.copy(), "sources": np.zeros_like(committed.x)}
    arrays[field] = arrays[field][:-1] if defect == "shape" else np.full_like(arrays[field], np.nan)
    with pytest.raises(ValueError, match=field):
        assemble_global_residual(
            operators, arrays["x"], case.bc_values, arrays["traction"], sources=arrays["sources"],
        )


def test_failed_global_assembly_leaves_committed_state_unchanged(setup: Setup) -> None:
    case, operators, transfer, committed = setup
    saved = committed.copy()
    with pytest.raises(ValueError, match="sources"):
        evaluate_global_residual(
            operators, transfer, case.material_points, committed, committed.x, case.bc_values,
            sources=np.array([np.inf]),
        )
    assert_state_equal(committed, saved)
    assert all(point.trial is None for point in case.material_points)


@pytest.mark.parametrize("field", ["u", "r", "p", "all"])
def test_elastic_directional_difference_matches_original_tpsa_operator(
    setup: Setup, field: str,
) -> None:
    """Check residual sensitivity without using or implementing a plastic Jacobian."""
    case, operators, transfer, committed = setup
    nc = case.grid.num_cells
    x, boundary = affine_candidate(case, np.array([[1e-4, 2e-4], [-1e-4, 1e-4]]))
    rng = np.random.default_rng(23)
    direction = np.zeros_like(x)
    blocks = {"u": slice(0, 2 * nc), "r": slice(2 * nc, 3 * nc), "p": slice(3 * nc, 4 * nc)}
    for name, block in blocks.items():
        if field in (name, "all"):
            direction[block] = rng.normal(size=direction[block].size) * (1e-4 if name == "u" else 1e7)
    step = 0.01  # Dimensionless: field units are carried by direction.
    plus = evaluate_global_residual(
        operators, transfer, case.material_points, committed, x + step * direction, boundary,
    )
    minus = evaluate_global_residual(
        operators, transfer, case.material_points, committed, x - step * direction, boundary,
    )
    # The finite difference is valid on the common elastic branch at fixed load/history.
    for result in (plus, minus):
        np.testing.assert_array_equal(result.stress_correction, 0.0)
        assert all(history.alpha == 0.0 for history in result.trial.material_states)
    actual = (plus.residual - minus.residual) / (2 * step)
    expected = np.asarray(
        (case.div @ case.face_discretization - case.accum) @ direction, dtype=np.float64,
    )
    assert_blocks(actual, expected, nc)
    assert np.linalg.norm(actual[:2 * nc]) > 0.0


def test_global_force_balance_equals_boundary_reactions_minus_body_force(setup: Setup) -> None:
    """Interior forces cancel in the domain sum even with nonuniform plasticity."""
    case, operators, _, committed = setup
    nc = case.grid.num_cells

    def biased_weights(grid: Grid, face: int, cells: NDArray[np.int64]) -> FloatArray:
        return np.array([0.2, 0.8])

    transfer = CellToFaceTransfer.from_rule(case.grid, biased_weights)
    rng = np.random.default_rng(8)
    x = rng.normal(size=4 * nc)
    x[:2 * nc] *= 0.003
    x[2 * nc:] *= 1e8
    # Start with body-force density and integrate exactly once, outside the evaluator.
    body_force = np.array([[2e5], [-3e5]]) * case.grid.cell_volumes
    sources = np.zeros(4 * nc)
    sources[:2 * nc] = body_force.ravel(order="F")
    result = evaluate_global_residual(
        operators, transfer, case.material_points, committed, x, case.bc_values, sources=sources,
    )
    assert max(history.alpha for history in result.trial.material_states) > 0.0
    assert np.ptp(result.stress_correction[0, 0]) > 1e6
    boundary_force = np.zeros(2)
    incidence = case.grid.cell_faces.tocsr()
    for face in operators.boundary_faces:
        sign = incidence.data[incidence.indptr[face]]
        boundary_force += sign * result.trial.traction[:, face]
    summed_momentum = result.residual[:2 * nc].reshape((2, nc), order="F").sum(axis=1)
    np.testing.assert_allclose(
        summed_momentum, boundary_force - body_force.sum(axis=1), rtol=1e-12, atol=1e-5,
    )


@pytest.mark.parametrize("auxiliary_sources", [False, True], ids=["body-force", "all-sources"])
def test_loaded_elastic_equilibrium_on_rotated_nonuniform_grid(auxiliary_sources: bool) -> None:
    """Use an independent mixed linear solve with nonzero prescribed loads."""
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.prepare_simulation()
    grid = case.grid
    # Unequal cell volumes and face measures expose unintended source/flux rescaling.
    grid.nodes[0] = 2.0 * grid.nodes[0] ** 2
    grid.nodes[1] = 0.75 * grid.nodes[1] ** 2
    angle = np.pi / 6
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    grid.nodes[:2] = rotation @ grid.nodes[:2]
    grid.compute_geometry()
    assert np.ptp(grid.cell_volumes) > 0.1
    case.matrices = _discretize_get_matrices(grid, case.d)
    case.face_discretization, case.rhs_matrix, case.div, case.accum = _assemble_matrices(
        case.matrices, grid, case.d,
    )
    _, boundary = affine_candidate(case, np.array([[1e-4, 2e-5], [-3e-5, 5e-5]]))
    nc = grid.num_cells
    sources = np.zeros(4 * nc)
    sources[:2 * nc] = (np.array([[2e5], [-1e5]]) * grid.cell_volumes).ravel(order="F")
    if auxiliary_sources:
        sources[2 * nc:3 * nc] = 2e-6 * grid.cell_volumes
        sources[3 * nc:] = -3e-6 * grid.cell_volumes
    matrix = case.div @ case.face_discretization - case.accum
    rhs = sources - case.div @ case.rhs_matrix @ boundary.ravel(order="F")
    x = np.asarray(sps.linalg.spsolve(matrix, rhs), dtype=np.float64)
    result = evaluate_global_residual(
        TpsaOperators(case), CellToFaceTransfer(grid), case.material_points,
        TpsaState.zeros(grid), x, boundary, sources=sources,
    )
    np.testing.assert_array_equal(result.stress_correction, 0.0)
    assert_blocks(result.residual, np.zeros_like(x), nc)
    # Omitting the nonzero physical load would leave a measurable imbalance.
    assert np.linalg.norm(original_elastic_residual(case, x, boundary, np.zeros_like(x))) > 1e4
    assert case.x is None
