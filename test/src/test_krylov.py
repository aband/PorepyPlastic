"""PETSc Krylov solvers for the Newton correction (core.krylov)."""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps

pytest.importorskip("petsc4py")
from petsc4py import PETSc  # noqa: E402

import porepy as pp  # noqa: E402

from core.jacobian import AnalyticalJacobian  # noqa: E402
from core.krylov import KrylovOptions, KrylovSolver, KrylovSolveStats  # noqa: E402
from core.layout import FloatArray  # noqa: E402
from core.linear_solvers import CondensedDirectSolver, DirectSolver, LinearSolver  # noqa: E402
from core.loading import LoadStepController  # noqa: E402
from core.newton import solve_newton  # noqa: E402
from core.residual import LoadIncrement, ResidualAssembler, ResidualEvaluation  # noqa: E402
from core.state import MaterialHistory, SolutionState  # noqa: E402
from material.J2 import _materialProperty, vonMisesModel  # noqa: E402
from material.material_state import MaterialPoint  # noqa: E402
from material.scalar_hardening import (  # noqa: E402
    HardeningParameters, isotropic_hardening_K, kinematic_hardening_H,
)
from utils.load_schedule import AdaptiveLoadOptions  # noqa: E402
from utils.newton_utils import BacktrackingOptions, NewtonDivergenceError  # noqa: E402
from utils.problem import PlaneStrainProblem  # noqa: E402


# ---------------------------------------------------------------- problems
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


def _distorted_quads() -> pp.Grid:
    grid = pp.CartGrid(np.array([4, 4]), np.array([1.0, 1.0]))
    inner = np.all((grid.nodes[:2] > 1e-9) & (grid.nodes[:2] < 1.0 - 1e-9), axis=0)
    grid.nodes[:2, inner] += 0.04 * np.random.default_rng(0).uniform(-1.0, 1.0, (2, inner.sum()))
    return grid


def _rotated_cell(dim: int) -> pp.Grid:
    unit = np.array([0.0, 1.0])
    grid = pp.TensorGrid(unit, unit, unit if dim == 3 else None)
    c, s = np.cos(0.37), np.sin(0.37)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if dim == 3:
        rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ rotation
    grid.nodes[:] = rotation @ grid.nodes
    return grid


CASES: dict[str, Callable[[], ResidualAssembler]] = {
    "2d-mixed": lambda: _assembler(_tensor_grid(2), mixed=True),
    "3d-dirichlet": lambda: _assembler(_tensor_grid(3)),
    "3d-mixed": lambda: _assembler(_tensor_grid(3), mixed=True),
    "2d-distorted": lambda: _assembler(_distorted_quads()),
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
    z = 0.02 * np.random.default_rng(2).normal(size=assembler.layout.size)
    evaluation = assembler.evaluate(z, _loads(assembler), MaterialHistory.zeros(assembler.layout))
    assert evaluation.material_trial is not None
    assert np.any(evaluation.material_trial.return_map.dgamma > 0.0)
    return evaluation


def _relative(actual: FloatArray, expected: FloatArray) -> float:
    return float(np.abs(actual - expected).max() / max(np.abs(expected).max(), 1e-300))


def _plane_strain(cells: int = 8) -> tuple[ResidualAssembler, FloatArray]:
    problem = PlaneStrainProblem(
        cells_per_axis=cells,
        boundary_displacement=lambda x: np.vstack(
            (6e-3 * x[0] ** 2 * x[1] + 1e-3 * x[1], -4e-3 * x[0] * x[1]),
        ),
    )
    problem.hardening_parameters.sigma_u, problem.hardening_parameters.delta = 400e6, 30.0
    problem.prepare_simulation()
    assembler = ResidualAssembler(problem.grid, problem.bc, problem.material_points)
    return assembler, problem.bc_values[:, assembler.layout.dirichlet_faces]


def _controller(
    assembler: ResidualAssembler, reference: FloatArray, solver: LinearSolver,
) -> LoadStepController:
    layout = assembler.layout
    state = SolutionState.zeros(
        layout, num_sides=assembler.operators.num_sides,
        num_neumann_faces=assembler.neumann_faces.size,
    )
    return LoadStepController(
        assembler, state, reference, line_search=BacktrackingOptions(), linear_solver=solver,
    )


# ---------------------------------------------------------------- accuracy
@pytest.mark.parametrize("options", [
    KrylovOptions(rtol=1e-10),
    KrylovOptions(preconditioner="ilu", rtol=1e-10),
    KrylovOptions(system="full", preconditioner="ilu", ilu_levels=1, rtol=1e-10),
], ids=["condensed-gamg", "condensed-ilu0", "full-ilu1"])
def test_single_solve_matches_direct(assembler: ResidualAssembler, options: KrylovOptions) -> None:
    jacobian, layout = AnalyticalJacobian(assembler), assembler.layout
    old, evaluation = MaterialHistory.zeros(layout), _plastic_evaluation(assembler)
    residual = np.random.default_rng(3).normal(size=layout.size)

    direct = CondensedDirectSolver()
    direct.bind(assembler, jacobian)
    expected = direct.solve(old, evaluation, residual, iteration=0).correction

    with KrylovSolver(options) as solver:
        solver.bind(assembler, jacobian)
        result = solver.solve(old, evaluation, residual, iteration=0)
        assert result.iterations > 0
        assert solver.stats[-1].reason.startswith("CONVERGED")
    for block in (layout.u_slice, layout.r_slice, layout.p_slice, layout.face_slice):
        assert _relative(result.correction[block], expected[block]) < 1e-7


def test_rtol_bounds_true_residual_with_right_preconditioning() -> None:
    """With left preconditioning PETSc tests the preconditioned norm instead; on
    this system that left a true residual about 6x above rtol."""
    assembler = CASES["3d-mixed"]()
    jacobian, layout = AnalyticalJacobian(assembler), assembler.layout
    old, evaluation = MaterialHistory.zeros(layout), _plastic_evaluation(assembler)
    system = jacobian.condensed_elastic()
    S = system.face_matrix(jacobian.plastic_face_block(old, evaluation))
    b = np.random.default_rng(5).normal(size=S.shape[0])
    rtol = 1e-6
    with KrylovSolver(KrylovOptions(rtol=rtol)) as solver:
        solver.bind(assembler, jacobian)
        x = solver._krylov(S, b, iteration=0, block_size=layout.dim)
    assert np.linalg.norm(S @ x - b) <= rtol * np.linalg.norm(b)


def test_newton_matches_direct(assembler: ResidualAssembler) -> None:
    old, loads = MaterialHistory.zeros(assembler.layout), _loads(assembler)
    direct = solve_newton(assembler, loads, old, linear_solver=DirectSolver())
    with KrylovSolver() as solver:
        krylov = solve_newton(assembler, loads, old, linear_solver=solver)
        assert len(solver.stats) == krylov.iterations
        np.testing.assert_array_equal(krylov.linear_iterations, [s.iterations for s in solver.stats])
    assert krylov.iterations == direct.iterations
    assert np.all(krylov.linear_iterations > 0)
    assert _relative(krylov.z, direct.z) < 1e-6


@pytest.mark.parametrize("options", [
    KrylovOptions(),
    KrylovOptions(preconditioner="ilu"),
], ids=["gamg", "ilu0"])
def test_controller_load_unload_matches_direct(options: KrylovOptions) -> None:
    assembler, reference = _plane_strain()
    direct = _controller(assembler, reference, DirectSolver())
    direct.run([0.5, 1.0, 0.6])
    with KrylovSolver(options) as solver:
        krylov = _controller(assembler, reference, solver)
        krylov.run([0.5, 1.0, 0.6])
    assert [r.newton.iterations for r in krylov.steps] == [r.newton.iterations for r in direct.steps]
    assert np.any(direct.state.material.alpha > 0.0)
    for name in ("x", "face_displacement"):
        assert _relative(getattr(krylov.state, name), getattr(direct.state, name)) < 1e-6
    assert _relative(krylov.state.material.plastic_strain, direct.state.material.plastic_strain) < 1e-6


@pytest.mark.parametrize("dim", [2, 3])
def test_no_free_faces_never_calls_petsc(dim: int, monkeypatch: pytest.MonkeyPatch) -> None:
    assembler = _assembler(_rotated_cell(dim))
    assert assembler.layout.free_faces.size == 0
    solver = KrylovSolver()

    def forbidden(*args: object, **kwargs: object) -> FloatArray:
        raise AssertionError("PETSc must not be called without face unknowns.")

    monkeypatch.setattr(solver, "_krylov", forbidden)
    old, loads = MaterialHistory.zeros(assembler.layout), _loads(assembler)
    direct = solve_newton(assembler, loads, old, linear_solver=DirectSolver())
    krylov = solve_newton(assembler, loads, old, linear_solver=solver)
    assert krylov.iterations == direct.iterations
    assert _relative(krylov.z, direct.z) < 1e-12
    np.testing.assert_array_equal(krylov.linear_iterations, 0)
    assert solver._near_null_space is None
    solver.close()


# ---------------------------------------------------------------- failures
def test_non_convergence_raises_recoverable_error() -> None:
    assembler = CASES["2d-mixed"]()
    solver = KrylovSolver(KrylovOptions(max_iterations=1, rtol=1e-12))
    with pytest.raises(NewtonDivergenceError, match="DIVERGED_MAX_IT"):
        solve_newton(
            assembler, _loads(assembler), MaterialHistory.zeros(assembler.layout),
            linear_solver=solver,
        )
    assert solver.stats[-1].reason == "DIVERGED_MAX_IT"
    assert not solver.stats[-1].fallback
    solver.close()


def test_petsc_setup_error_is_recoverable() -> None:
    """A zero pivot in ILU is a PETSc error, reported as NewtonDivergenceError."""
    assembler = CASES["2d-mixed"]()
    with KrylovSolver(KrylovOptions(preconditioner="ilu")) as solver:
        solver.bind(assembler, AnalyticalJacobian(assembler))
        zero_diagonal = sps.csr_array(np.array([[0.0, 1.0], [1.0, 0.0]]))
        with pytest.raises(NewtonDivergenceError, match="PETSc error"):
            solver._krylov(zero_diagonal, np.ones(2), iteration=0, block_size=None)
        assert solver.stats[-1].reason.startswith("PETSC_ERROR")


def test_fallback_to_direct() -> None:
    assembler = CASES["2d-mixed"]()
    old, loads = MaterialHistory.zeros(assembler.layout), _loads(assembler)
    direct = solve_newton(assembler, loads, old, linear_solver=CondensedDirectSolver())
    with KrylovSolver(KrylovOptions(max_iterations=1, rtol=1e-12, fallback_to_direct=True)) as solver:
        result = solve_newton(assembler, loads, old, linear_solver=solver)
        assert solver.stats and all(stats.fallback for stats in solver.stats)
        assert all(stats.reason == "DIVERGED_MAX_IT" for stats in solver.stats)
    assert result.iterations == direct.iterations
    assert _relative(result.z, direct.z) < 1e-12


class _FailFirstKrylov(KrylovSolver):
    """The first Krylov solve fails as a breakdown would; later ones run normally."""

    failed = False

    def _krylov(self, *args: object, **kwargs: object) -> FloatArray:
        if not self.failed:
            self.failed = True
            raise NewtonDivergenceError("Injected Krylov failure.")
        return super()._krylov(*args, **kwargs)  # type: ignore[arg-type]


def test_failure_is_recovered_by_adaptive_cutback() -> None:
    assembler, reference = _plane_strain(6)
    solver = _FailFirstKrylov()
    with solver:
        controller = _controller(assembler, reference, solver)
        records = controller.run_adaptive([0.5], options=AdaptiveLoadOptions(
            initial_increment=0.5, min_increment=0.125, max_increment=0.5, growth_factor=1.0,
        ))
    assert solver.failed
    assert [record.load_factor for record in records] == pytest.approx([0.25, 0.5])
    assert [attempt.accepted for attempt in controller.attempts] == [False, True, True]
    assert controller.attempts[0].error == "Injected Krylov failure."


# ---------------------------------------------------------------- PETSc objects
def test_petsc_options_are_applied_per_solver_and_cleaned_up() -> None:
    assembler = CASES["2d-mixed"]()
    jacobian = AnalyticalJacobian(assembler)
    with KrylovSolver(KrylovOptions(petsc_options={"ksp_type": "bcgs", "pc_gamg_threshold": "0.02"})) as first, \
            KrylovSolver() as second:
        first.bind(assembler, jacobian)
        second.bind(assembler, jacobian)
        assert first._ksp.getType() == "bcgs"
        assert second._ksp.getType() == "gmres"
        assert not [name for name in PETSc.Options().getAll() if name.startswith("core_krylov")]
        # The overridden solver still converges.
        old, evaluation = MaterialHistory.zeros(assembler.layout), _plastic_evaluation(assembler)
        first.solve(old, evaluation, np.ones(assembler.layout.size), iteration=0)
        assert first.stats[-1].reason.startswith("CONVERGED")


@pytest.mark.parametrize(("case", "modes"), [("2d-mixed", 3), ("3d-mixed", 6)])
def test_rigid_body_near_null_space(case: str, modes: int) -> None:
    assembler = CASES[case]()
    jacobian = AnalyticalJacobian(assembler)
    with KrylovSolver() as gamg, KrylovSolver(KrylovOptions(preconditioner="ilu")) as ilu, \
            KrylovSolver(KrylovOptions(system="full", preconditioner="ilu")) as full:
        for solver in (gamg, ilu, full):
            solver.bind(assembler, jacobian)
        assert len(gamg._near_null_space.getVecs()) == modes
        assert ilu._near_null_space is None
        assert full._near_null_space is None


def test_ksp_reused_and_closed() -> None:
    assembler = CASES["2d-mixed"]()
    jacobian = AnalyticalJacobian(assembler)
    old, evaluation = MaterialHistory.zeros(assembler.layout), _plastic_evaluation(assembler)
    residual = np.ones(assembler.layout.size)
    solver = KrylovSolver()
    solver.bind(assembler, jacobian)
    ksp = solver._ksp
    solver.bind(assembler, jacobian)          # same Jacobian: no rebuild
    solver.solve(old, evaluation, residual, iteration=0)
    solver.solve(old, evaluation, residual, iteration=1)
    assert solver._ksp is ksp
    solver.close()
    assert solver._ksp is None
    with pytest.raises(RuntimeError, match="closed"):
        solver.solve(old, evaluation, residual, iteration=2)
    with pytest.raises(RuntimeError, match="closed"):
        solver.bind(assembler, jacobian)


def test_context_manager_closes() -> None:
    with KrylovSolver() as solver:
        pass
    assert solver._closed


def test_condensation_shared_with_direct_solver() -> None:
    assembler = CASES["2d-mixed"]()
    jacobian = AnalyticalJacobian(assembler)
    direct = CondensedDirectSolver()
    direct.bind(assembler, jacobian)
    with KrylovSolver() as solver:
        solver.bind(assembler, jacobian)
        assert solver._system is direct._system is jacobian.condensed_elastic()


def test_solver_bound_to_one_assembler() -> None:
    first, second = CASES["2d-mixed"](), CASES["2d-distorted"]()
    with KrylovSolver() as solver:
        solver.bind(first, AnalyticalJacobian(first))
        with pytest.raises(ValueError, match="another assembler"):
            solver.bind(second, AnalyticalJacobian(second))


# ---------------------------------------------------------------- options and API
def test_stats_record_each_solve() -> None:
    assembler = CASES["2d-mixed"]()
    with KrylovSolver() as solver:
        result = solve_newton(
            assembler, _loads(assembler), MaterialHistory.zeros(assembler.layout),
            linear_solver=solver,
        )
        assert [stats.iteration for stats in solver.stats] == list(range(result.iterations))
        for stats in solver.stats:
            assert isinstance(stats, KrylovSolveStats)
            assert stats.iterations > 0 and stats.reason.startswith("CONVERGED")
            assert stats.setup_time >= 0.0 and stats.solve_time >= 0.0
            assert np.isfinite(stats.residual_norm) and not stats.fallback


@pytest.mark.parametrize("bad", [
    {"rtol": 0.0},
    {"rtol": 1.0},
    {"rtol": float("nan")},
    {"system": "monolithic"},
    {"method": "cg"},
    {"preconditioner": "hypre"},
    {"ilu_levels": -1},
    {"ilu_levels": True},
    {"max_iterations": 0},
    {"restart": 1.5},
    {"method": "fgmres", "right_preconditioning": False},
    {"petsc_options": {"-ksp_type": "cg"}},
    {"petsc_options": {"ksp_rtol": 1e-3}},
])
def test_invalid_options_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        KrylovOptions(**bad)  # type: ignore[arg-type]


def test_petsc_options_are_copied() -> None:
    options = {"pc_gamg_threshold": "0.01"}
    frozen = KrylovOptions(petsc_options=options)
    options["ksp_type"] = "bcgs"
    assert dict(frozen.petsc_options) == {"pc_gamg_threshold": "0.01"}
    with pytest.raises(TypeError):
        frozen.petsc_options["ksp_type"] = "bcgs"  # type: ignore[index]


def test_solver_api() -> None:
    with pytest.raises(ValueError, match="KrylovOptions"):
        KrylovSolver(0.5)  # type: ignore[arg-type]
    with KrylovSolver() as solver:
        assert isinstance(solver, LinearSolver)


def test_missing_petsc_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "petsc4py", None)
    with pytest.raises(ImportError, match="requires petsc4py"):
        KrylovSolver()


def test_importing_module_does_not_import_petsc() -> None:
    """PETSc is optional: core.krylov imports cleanly and loads PETSc only on use."""
    code = "import sys, core.krylov; print('petsc4py' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parents[2],   # repository root
    )
    assert result.stdout.strip() == "False"
