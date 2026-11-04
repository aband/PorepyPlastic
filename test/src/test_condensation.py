"""Static condensation of the Newton system onto free-face unknowns."""

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

from core.condensation import CondensedElasticSystem
from core.jacobian import AnalyticalJacobian
from core.linear_solvers import CondensedDirectSolver, DirectSolver
from core.loading import LoadStepController
from core.newton import solve_newton
from core.residual import LoadIncrement, ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory, SolutionState
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import HardeningParameters, isotropic_hardening_K, kinematic_hardening_H
from utils.newton_utils import BacktrackingOptions, _solve_linear_correction
from utils.problem import PlaneStrainProblem


def _points(num_cells: int) -> list[MaterialPoint]:
    """Soft material with a low yield stress: modest strains give plastic cells."""
    points = []
    for _ in range(num_cells):
        material = _materialProperty()
        material.young_modulus, material.poisson_ratio = 12.0, 0.25
        points.append(MaterialPoint(
            material, HardeningParameters(0.05, 0.05, 1.0, 0.4, 0.0),
            vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H,
        ))
    return points


def _assembler(grid: pp.Grid, *, mixed: bool = False) -> ResidualAssembler:
    grid.compute_geometry()
    boundary = grid.get_all_boundary_faces()
    bc = pp.BoundaryConditionVectorial(grid, boundary[::2] if mixed else boundary, "dir")
    return ResidualAssembler(grid, bc, _points(grid.num_cells))


def _tensor_grid(dim: int) -> pp.Grid:
    x, y, z = np.array([0.0, 0.3, 1.0]), np.array([0.0, 0.4, 1.3]), np.array([0.0, 0.2, 0.8])
    return pp.TensorGrid(x, y, z if dim == 3 else None)


def _rotated_cell(dim: int) -> pp.Grid:
    """One cell, rotated so that its face normals do not cancel exactly in floating point."""
    unit = np.array([0.0, 1.0])
    grid = pp.TensorGrid(unit, unit, unit if dim == 3 else None)
    c, s = np.cos(0.37), np.sin(0.37)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if dim == 3:
        rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ rotation
    grid.nodes[:] = rotation @ grid.nodes
    return grid


def _distorted_quads() -> pp.Grid:
    grid = pp.CartGrid(np.array([4, 4]), np.array([1.0, 1.0]))
    inner = np.all((grid.nodes[:2] > 1e-9) & (grid.nodes[:2] < 1.0 - 1e-9), axis=0)
    grid.nodes[:2, inner] += 0.04 * np.random.default_rng(0).uniform(-1.0, 1.0, (2, inner.sum()))
    return grid


CASES = {
    "2d-dirichlet": lambda: _assembler(_tensor_grid(2)),
    "2d-mixed": lambda: _assembler(_tensor_grid(2), mixed=True),
    "3d-dirichlet": lambda: _assembler(_tensor_grid(3)),
    "3d-mixed": lambda: _assembler(_tensor_grid(3), mixed=True),
    "2d-distorted": lambda: _assembler(_distorted_quads()),
}
SINGLE_CELL = {
    "2d-single": lambda: _assembler(_rotated_cell(2)),
    "3d-single": lambda: _assembler(_rotated_cell(3)),
}


@pytest.fixture(params=list(CASES), ids=list(CASES))
def assembler(request: pytest.FixtureRequest) -> ResidualAssembler:
    return CASES[request.param]()


def _loads(assembler: ResidualAssembler, scale: float = 0.02) -> LoadIncrement:
    layout, rng = assembler.layout, np.random.default_rng(1)
    return LoadIncrement(
        scale * rng.normal(size=(layout.dim, layout.dirichlet_faces.size)),
        0.5 * scale * rng.normal(size=(layout.dim, assembler.neumann_faces.size)),
        np.zeros(layout.cell_size),
    )


def _plastic_evaluation(assembler: ResidualAssembler) -> ResidualEvaluation:
    """An iterate with plastic cells, so the Jacobian carries the plastic face block."""
    z = 0.02 * np.random.default_rng(2).normal(size=assembler.layout.size)
    evaluation = assembler.evaluate(z, _loads(assembler), MaterialHistory.zeros(assembler.layout))
    assert evaluation.material_trial is not None
    assert np.any(evaluation.material_trial.return_map.dgamma > 0.0)
    return evaluation


def _relative(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.abs(actual - expected).max() / max(np.abs(expected).max(), 1e-300))


def test_cell_inverse_is_exact(assembler: ResidualAssembler) -> None:
    jacobian = AnalyticalJacobian(assembler)
    cells = assembler.layout.cell_slice
    system = jacobian.condensed_elastic()
    identity = system.A_cc_inv @ jacobian.L[cells, cells]
    np.testing.assert_allclose(identity.toarray(), np.eye(assembler.layout.cell_size), atol=1e-13)


def test_plastic_term_lives_in_face_block(assembler: ResidualAssembler) -> None:
    jacobian = AnalyticalJacobian(assembler)
    old, evaluation = MaterialHistory.zeros(assembler.layout), _plastic_evaluation(assembler)
    layout = assembler.layout
    cells, faces = layout.cell_slice, layout.face_slice
    difference = sps.csr_array(jacobian.evaluate(old, evaluation) - jacobian.L)
    block = jacobian.plastic_face_block(old, evaluation)
    assert block is not None
    scale = np.abs(block.data).max()
    # Strain never depends on cell unknowns: cell columns are exactly empty.
    assert difference[:, cells].count_nonzero() == 0
    # Cell rows vanish analytically (eq. 21); evaluate() keeps P's momentum-row
    # round-off, which is nonzero on distorted grids.
    assert np.abs(difference[cells].data).max(initial=0.0) <= 1e-12 * scale
    np.testing.assert_allclose(difference[faces, faces].toarray(), block.toarray(), atol=1e-12 * scale)


def test_plastic_face_block_is_none_when_elastic(assembler: ResidualAssembler) -> None:
    jacobian = AnalyticalJacobian(assembler)
    layout = assembler.layout
    evaluation = assembler.evaluate(
        np.zeros(layout.size), _loads(assembler, scale=1e-6), MaterialHistory.zeros(layout),
    )
    assert jacobian.plastic_face_block(MaterialHistory.zeros(layout), evaluation) is None


def test_plastic_face_block_requires_material_trial(assembler: ResidualAssembler) -> None:
    jacobian = AnalyticalJacobian(assembler)
    layout = assembler.layout
    evaluation = assembler.evaluate_elastic(np.zeros(layout.size), _loads(assembler))
    with pytest.raises(ValueError, match="material trial"):
        jacobian.plastic_face_block(MaterialHistory.zeros(layout), evaluation)


@pytest.mark.parametrize("plastic", [False, True], ids=["elastic", "plastic"])
def test_condensed_solve_matches_full_lu(assembler: ResidualAssembler, plastic: bool) -> None:
    jacobian = AnalyticalJacobian(assembler)
    layout = assembler.layout
    old = MaterialHistory.zeros(layout)
    if plastic:
        evaluation = _plastic_evaluation(assembler)
    else:
        evaluation = assembler.evaluate(np.zeros(layout.size), _loads(assembler, scale=1e-6), old)
    residual = np.random.default_rng(3).normal(size=layout.size)

    expected = _solve_linear_correction(jacobian.evaluate(old, evaluation), residual, 0)
    solver = CondensedDirectSolver()
    solver.bind(assembler, jacobian)
    actual = solver.solve(old, evaluation, residual, iteration=0).correction
    for block in (layout.u_slice, layout.r_slice, layout.p_slice, layout.face_slice):
        assert _relative(actual[block], expected[block]) < 1e-10


def test_reduce_and_recover_invert_the_full_system(assembler: ResidualAssembler) -> None:
    """J x = b holds blockwise after reduce -> face solve -> recover."""
    jacobian = AnalyticalJacobian(assembler)
    system, layout = jacobian.condensed_elastic(), assembler.layout
    rhs = np.random.default_rng(4).normal(size=layout.size)
    b_f, b_c = system.reduce(rhs)
    x_f = _solve_linear_correction(system.face_matrix(None), -b_f, 0)
    x = system.recover(x_f, b_c)
    np.testing.assert_allclose(jacobian.L @ x, rhs, rtol=1e-9, atol=1e-9 * np.abs(rhs).max())


def test_condensed_solver_in_newton(assembler: ResidualAssembler) -> None:
    old, loads = MaterialHistory.zeros(assembler.layout), _loads(assembler)
    direct = solve_newton(assembler, loads, old, linear_solver=DirectSolver())
    condensed = solve_newton(assembler, loads, old, linear_solver=CondensedDirectSolver())
    assert condensed.iterations == direct.iterations
    assert condensed.evaluations == direct.evaluations
    assert _relative(condensed.z, direct.z) < 1e-10
    np.testing.assert_array_equal(condensed.linear_iterations, 0)


def test_condensed_solver_in_controller() -> None:
    """Load, unload and line search: same Newton path as the direct solver."""
    problem = PlaneStrainProblem(
        cells_per_axis=8,
        boundary_displacement=lambda x: np.vstack(
            (6e-3 * x[0] ** 2 * x[1] + 1e-3 * x[1], -4e-3 * x[0] * x[1]),
        ),
    )
    problem.hardening_parameters.sigma_u, problem.hardening_parameters.delta = 400e6, 30.0
    problem.prepare_simulation()
    assembler = ResidualAssembler(problem.grid, problem.bc, problem.material_points)
    layout = assembler.layout
    reference = problem.bc_values[:, layout.dirichlet_faces]

    controllers = []
    for solver in (DirectSolver(), CondensedDirectSolver()):
        state = SolutionState.zeros(
            layout, num_sides=assembler.operators.num_sides,
            num_neumann_faces=assembler.neumann_faces.size,
        )
        controller = LoadStepController(
            assembler, state, reference, line_search=BacktrackingOptions(), linear_solver=solver,
        )
        controller.run([0.5, 1.0, 0.6])
        controllers.append(controller)
    direct, condensed = controllers

    assert [r.newton.iterations for r in condensed.steps] == [r.newton.iterations for r in direct.steps]
    assert np.any(direct.state.material.alpha > 0.0)
    for name in ("x", "face_displacement", "side_traction"):
        assert _relative(getattr(condensed.state, name), getattr(direct.state, name)) < 1e-9
    for name in ("plastic_strain", "backstress", "alpha"):
        assert _relative(getattr(condensed.state.material, name), getattr(direct.state.material, name)) < 1e-9


@pytest.mark.parametrize("make", list(SINGLE_CELL.values()), ids=list(SINGLE_CELL))
def test_no_free_faces(make: object) -> None:
    """One all-Dirichlet cell: empty face system, correction is A_cc^-1 b_c."""
    assembler = make()  # type: ignore[operator]
    layout = assembler.layout
    assert layout.free_faces.size == 0

    jacobian = AnalyticalJacobian(assembler)  # rotated cell: round-off in P must be tolerated
    system = jacobian.condensed_elastic()
    assert system.num_face_dofs == 0

    old, loads = MaterialHistory.zeros(layout), _loads(assembler)
    direct = solve_newton(assembler, loads, old, linear_solver=DirectSolver())
    condensed = solve_newton(assembler, loads, old, linear_solver=CondensedDirectSolver())
    assert condensed.iterations == direct.iterations
    assert _relative(condensed.z, direct.z) < 1e-12


def test_condensation_is_cached_and_shared(assembler: ResidualAssembler) -> None:
    jacobian = AnalyticalJacobian(assembler)
    first, second = CondensedDirectSolver(), CondensedDirectSolver()
    first.bind(assembler, jacobian)
    second.bind(assembler, jacobian)
    assert jacobian.condensed_elastic() is jacobian.condensed_elastic()
    assert first._system is second._system is jacobian.condensed_elastic()


def test_cached_matrices_are_read_only(assembler: ResidualAssembler) -> None:
    system = AnalyticalJacobian(assembler).condensed_elastic()
    assert system.face_matrix(None) is system.S_elastic
    for matrix in (system.A_cc_inv, system.A_cf, system.A_fc, system.S_elastic):
        for array in (matrix.data, matrix.indices, matrix.indptr):
            assert not array.flags.writeable
    with pytest.raises(ValueError):
        system.S_elastic.data[0] = 1.0
    # Matrix-vector products and the plastic sum still work on read-only storage.
    vector = np.ones(system.num_face_dofs)
    assert np.all(np.isfinite(system.S_elastic @ vector))
    assert system.face_matrix(system.S_elastic) is not system.S_elastic


@pytest.mark.parametrize("violation", ["rotation_offdiagonal", "pressure_to_momentum_cols", "zero_diagonal"])
def test_structure_violation_is_rejected(violation: str) -> None:
    assembler = _assembler(_tensor_grid(2), mixed=True)
    layout = assembler.layout
    L = sps.lil_array(AnalyticalJacobian(assembler).L)
    r0, p0, u0 = layout.r_slice.start, layout.p_slice.start, layout.u_slice.start
    if violation == "rotation_offdiagonal":
        L[r0, r0 + 1] = 1.0        # r-r block no longer diagonal
    elif violation == "pressure_to_momentum_cols":
        L[p0, u0] = 1.0            # pressure row couples to u: N D^-1 N no longer vanishes
    else:
        L[p0, p0] = 0.0            # singular diagonal
    with pytest.raises(ValueError):
        CondensedElasticSystem(sps.csr_array(L), layout)


def test_cell_row_guard_rejects_structural_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stress-correction map that no longer cancels per cell must be caught."""
    assembler = _assembler(_tensor_grid(2), mixed=True)
    original = AnalyticalJacobian._build_stress_correction_matrix

    def broken(self: AnalyticalJacobian) -> sps.csr_array:
        matrix = sps.csr_array(original(self), copy=True)
        rows = slice(matrix.indptr[0], matrix.indptr[assembler.layout.dim])  # first side only
        matrix.data[rows] *= 2.0
        return matrix

    monkeypatch.setattr(AnalyticalJacobian, "_build_stress_correction_matrix", broken)
    with pytest.raises(ValueError, match="cell rows"):
        AnalyticalJacobian(assembler)
