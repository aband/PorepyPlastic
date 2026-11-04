"""PETSc Krylov solvers for the Newton correction (core.krylov)."""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import scipy.sparse as sps

pytest.importorskip("petsc4py")
from petsc4py import PETSc  # noqa: E402

import porepy as pp  # noqa: E402

from core.jacobian import AnalyticalJacobian  # noqa: E402
from core.krylov import PREDICTOR_ITERATION, KrylovOptions, KrylovSolver, KrylovSolveStats  # noqa: E402
from core.layout import FloatArray  # noqa: E402
from core.linear_solvers import (  # noqa: E402
    CondensedDirectSolver, DirectSolver, ElasticPredictor, LinearSolver,
)
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
    assembler: ResidualAssembler, reference: FloatArray, solver: LinearSolver | None,
    **kwargs: object,
) -> LoadStepController:
    layout = assembler.layout
    state = SolutionState.zeros(
        layout, num_sides=assembler.operators.num_sides,
        num_neumann_faces=assembler.neumann_faces.size,
    )
    return LoadStepController(
        assembler, state, reference, line_search=BacktrackingOptions(), linear_solver=solver,
        **kwargs,  # type: ignore[arg-type]
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
    with KrylovSolver(KrylovOptions(rtol=1e-8)) as solver:
        krylov = solve_newton(assembler, loads, old, linear_solver=solver)
        assert len(solver.stats) == krylov.iterations
        np.testing.assert_array_equal(krylov.linear_iterations, [s.iterations for s in solver.stats])
    assert krylov.iterations == direct.iterations
    assert np.all(krylov.linear_iterations > 0)
    assert _relative(krylov.z, direct.z) < 1e-6


@pytest.mark.parametrize("options", [
    KrylovOptions(rtol=1e-8),
    KrylovOptions(preconditioner="ilu", rtol=1e-8),
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
    {"rtol_max": 1.0},
    {"rtol_max": 0.0},
    {"forcing": "newton"},
    {"forcing": "eisenstat-walker", "rtol": 0.1, "rtol_max": 0.01},
    {"ew_gamma": 0.0},
    {"ew_gamma": 1.5},
    {"ew_alpha": 1.0},
    {"ew_alpha": 2.5},
    {"ew_alpha": True},
    {"matrix_update": "copy"},
    {"matrix_update": "in_place", "system": "full"},
    {"refresh_interpolation": "always"},
    {"refresh_interpolation": "step"},
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


# ---------------------------------------------------------------- tolerance (Phase 4a)
def test_default_tolerance_is_loose_but_converges_to_direct() -> None:
    """Newton decides convergence: rtol 1e-3 changes the path, not the accepted state."""
    assert KrylovOptions().rtol == 1e-3 and KrylovOptions().forcing == "fixed"
    assembler, reference = _plane_strain()
    direct = _controller(assembler, reference, DirectSolver())
    direct.run([0.5, 1.0, 0.6])
    with KrylovSolver() as solver:
        krylov = _controller(assembler, reference, solver)
        krylov.run([0.5, 1.0, 0.6])
        assert {stats.rtol for stats in solver.stats} == {1e-3}
    for name in ("x", "face_displacement"):
        assert _relative(getattr(krylov.state, name), getattr(direct.state, name)) < 1e-5
    assert _relative(krylov.state.material.plastic_strain, direct.state.material.plastic_strain) < 1e-5


def _ew(eta_previous: float, ratio: float, options: KrylovOptions) -> float:
    eta = options.ew_gamma * ratio ** options.ew_alpha
    guard = options.ew_gamma * eta_previous ** options.ew_alpha
    if guard > 0.1:
        eta = max(eta, guard)
    return float(min(max(eta, options.rtol), options.rtol_max))


def test_eisenstat_walker_sequence() -> None:
    options = KrylovOptions(forcing="eisenstat-walker", rtol=1e-6, rtol_max=1e-2)
    with KrylovSolver(options) as solver:
        norms = [1.0, 0.5, 1e-3, 1e-9, 1e-30]
        etas = [solver._forcing(norm) for norm in norms]
    expected = [options.rtol_max]
    for previous, current in zip(norms, norms[1:]):
        expected.append(_ew(expected[-1], current / previous, options))
    assert etas == pytest.approx(expected, rel=1e-14)
    assert etas[0] == 1e-2           # first correction of a step
    assert etas[1] == 1e-2           # slow decrease: clamped at rtol_max
    assert 1e-6 < etas[2] < 1e-2     # fast decrease: tighter
    assert etas[-1] == 1e-6          # clamped at the lower bound rtol


def test_eisenstat_walker_safeguard() -> None:
    """With a large previous eta, the sequence may not drop by more than the safeguard."""
    options = KrylovOptions(forcing="eisenstat-walker", rtol=1e-6, rtol_max=0.9)
    with KrylovSolver(options) as solver:
        first, second = solver._forcing(1.0), solver._forcing(1e-4)
    guard = options.ew_gamma * first ** options.ew_alpha
    assert guard > 0.1
    assert second == pytest.approx(guard)


def test_begin_step_restarts_forcing_and_fixed_ignores_history() -> None:
    with KrylovSolver(KrylovOptions(forcing="eisenstat-walker", rtol=1e-6)) as solver:
        solver._forcing(1.0)
        assert solver._forcing(1e-6) < 1e-2
        solver.begin_step()
        # Without the reset, the ratio 1e-12 / 1e-6 would give the lower bound instead.
        assert solver._forcing(1e-12) == 1e-2
    with KrylovSolver(KrylovOptions(rtol=1e-4)) as fixed:
        assert [fixed._forcing(n) for n in (1.0, 1e-3, 1e-9)] == [1e-4] * 3


@pytest.mark.parametrize("system", ["condensed", "full"])
def test_eisenstat_walker_in_newton(system: str) -> None:
    assembler = CASES["3d-mixed"]()
    old, loads = MaterialHistory.zeros(assembler.layout), _loads(assembler)
    direct = solve_newton(assembler, loads, old, linear_solver=DirectSolver())
    options = KrylovOptions(
        system=system,  # type: ignore[arg-type]
        preconditioner="gamg" if system == "condensed" else "ilu",
        forcing="eisenstat-walker", rtol=1e-8, rtol_max=1e-2,
    )
    with KrylovSolver(options) as solver:
        result = solve_newton(assembler, loads, old, linear_solver=solver)
        tolerances = [stats.rtol for stats in solver.stats]
    assert tolerances[0] == 1e-2                         # begin_step restarted the sequence
    assert all(1e-8 <= eta <= 1e-2 for eta in tolerances)
    assert tolerances[-1] < tolerances[0]                # tightens as Newton converges
    # Inexact steps change the path; the accepted iterate meets the same Newton test,
    # whose absolute tolerance (1e-5) is loose for this soft material.
    assert np.all(result.residual_norms[-1] <= result.thresholds)
    assert result.iterations <= direct.iterations + 1
    assert _relative(result.z, direct.z) < 1e-4


# ---------------------------------------------------------------- in-place matrix (Phase 4b)
def test_in_place_matches_rebuild(assembler: ResidualAssembler) -> None:
    jacobian, layout = AnalyticalJacobian(assembler), assembler.layout
    old, evaluation = MaterialHistory.zeros(layout), _plastic_evaluation(assembler)
    residual = np.random.default_rng(3).normal(size=layout.size)
    corrections = []
    updates: tuple[Literal["rebuild", "in_place"], ...] = ("rebuild", "in_place")
    for update in updates:
        with KrylovSolver(KrylovOptions(rtol=1e-10, matrix_update=update)) as solver:
            solver.bind(assembler, jacobian)
            corrections.append(solver.solve(old, evaluation, residual, iteration=0).correction)
    for block in (layout.u_slice, layout.r_slice, layout.p_slice, layout.face_slice):
        assert _relative(corrections[1][block], corrections[0][block]) < 1e-7


def test_in_place_keeps_one_matrix_with_fixed_pattern() -> None:
    assembler = CASES["2d-distorted"]()
    jacobian, layout = AnalyticalJacobian(assembler), assembler.layout
    old = MaterialHistory.zeros(layout)
    elastic = assembler.evaluate(np.zeros(layout.size), LoadIncrement(
        np.zeros((layout.dim, layout.dirichlet_faces.size)),
        np.zeros((layout.dim, assembler.neumann_faces.size)), np.zeros(layout.cell_size),
    ), old)
    plastic = _plastic_evaluation(assembler)
    residual = np.ones(layout.size)
    with KrylovSolver(KrylovOptions(matrix_update="in_place", rtol=1e-10)) as solver:
        solver.bind(assembler, jacobian)
        solver.solve(old, elastic, residual, iteration=0)
        mat = solver._mat
        indptr, indices, _ = mat.getValuesCSR()
        solver.solve(old, plastic, residual, iteration=1)     # different plastic set
        solver.begin_step()                                   # "never": no refresh
        solver.solve(old, plastic, residual, iteration=0)
        assert solver._mat is mat
        new_indptr, new_indices, values = mat.getValuesCSR()
        np.testing.assert_array_equal(new_indptr, indptr)
        np.testing.assert_array_equal(new_indices, indices)
        pattern = jacobian.fixed_face_pattern()
        np.testing.assert_allclose(values, jacobian.fixed_face_values(old, plastic), rtol=1e-15)
        assert all(stats.reason.startswith("CONVERGED") for stats in solver.stats)


def test_refresh_interpolation_replaces_matrix_each_step() -> None:
    assembler = CASES["2d-mixed"]()
    jacobian = AnalyticalJacobian(assembler)
    old, evaluation = MaterialHistory.zeros(assembler.layout), _plastic_evaluation(assembler)
    residual = np.ones(assembler.layout.size)
    options = KrylovOptions(matrix_update="in_place", refresh_interpolation="step")
    with KrylovSolver(options) as solver:
        solver.bind(assembler, jacobian)
        solver.solve(old, evaluation, residual, iteration=0)
        first = solver._mat
        solver.solve(old, evaluation, residual, iteration=1)
        assert solver._mat is first
        solver.begin_step()
        assert solver._mat is None
        solver.solve(old, evaluation, residual, iteration=0)
        assert solver._mat is not None and solver._mat is not first


@pytest.mark.parametrize("preconditioner", ["gamg", "ilu"])
def test_in_place_controller_matches_direct(preconditioner: str) -> None:
    assembler, reference = _plane_strain()
    direct = _controller(assembler, reference, DirectSolver())
    direct.run([0.5, 1.0, 0.6])
    options = KrylovOptions(
        preconditioner=preconditioner, matrix_update="in_place", rtol=1e-8,  # type: ignore[arg-type]
    )
    with KrylovSolver(options) as solver:
        krylov = _controller(assembler, reference, solver)
        krylov.run([0.5, 1.0, 0.6])
    assert [r.newton.iterations for r in krylov.steps] == [r.newton.iterations for r in direct.steps]
    for name in ("x", "face_displacement"):
        assert _relative(getattr(krylov.state, name), getattr(direct.state, name)) < 1e-6


def _prolongation(solver: KrylovSolver) -> tuple[int, FloatArray]:
    """Handle and values of GAMG's finest prolongation operator."""
    interpolation = solver._ksp.getPC().getMGInterpolation(1)
    return interpolation.handle, np.array(interpolation.getValuesCSR()[2], copy=True)


@pytest.mark.parametrize(("update", "petsc_options", "reused"), [
    ("in_place", {}, True),
    ("in_place", {"pc_gamg_reuse_interpolation": "false"}, False),   # user options win
    ("rebuild", {}, False),
], ids=["in-place", "in-place-user-off", "rebuild"])
def test_gamg_interpolation_reuse(update: str, petsc_options: dict[str, str], reused: bool) -> None:
    """In-place updates keep GAMG's prolongation although the matrix values change."""
    assembler, reference = _plane_strain(12)
    jacobian, layout = AnalyticalJacobian(assembler), assembler.layout
    old = MaterialHistory.zeros(layout)
    no_load = np.zeros((layout.dim, 0)), np.zeros(layout.cell_size)
    elastic = assembler.evaluate(np.zeros(layout.size), LoadIncrement(0.0 * reference, *no_load), old)
    plastic = assembler.evaluate(np.zeros(layout.size), LoadIncrement(reference, *no_load), old)
    assert plastic.material_trial is not None
    assert np.any(plastic.material_trial.return_map.dgamma > 0.0)
    options = KrylovOptions(matrix_update=update, petsc_options=petsc_options)  # type: ignore[arg-type]
    with KrylovSolver(options) as solver:
        solver.bind(assembler, jacobian)
        solver.solve(old, elastic, np.ones(layout.size), iteration=0)
        before = _prolongation(solver)
        solver.solve(old, plastic, np.ones(layout.size), iteration=1)
        after = _prolongation(solver)
    assert (before[0] == after[0]) is reused
    assert (before[1].shape == after[1].shape and np.array_equal(before[1], after[1])) is reused
    assert not [name for name in PETSc.Options().getAll() if name.startswith("core_krylov")]


# ---------------------------------------------------------------- elastic predictor (Phase 4c)
@pytest.mark.parametrize("options", [
    KrylovOptions(rtol=1e-10),
    KrylovOptions(preconditioner="ilu", rtol=1e-10),
    KrylovOptions(system="full", preconditioner="ilu", rtol=1e-10),
    KrylovOptions(matrix_update="in_place", rtol=1e-10),
], ids=["condensed-gamg", "condensed-ilu", "full-ilu", "in-place"])
def test_solve_elastic_matches_direct(assembler: ResidualAssembler, options: KrylovOptions) -> None:
    jacobian = AnalyticalJacobian(assembler)
    rhs = np.random.default_rng(7).normal(size=assembler.layout.size)
    expected = sps.linalg.spsolve(sps.csc_matrix(jacobian.L), rhs)
    with KrylovSolver(options) as solver:
        solver.bind(assembler, jacobian)
        result = solver.solve_elastic(rhs)
        assert solver.stats[-1].iteration == PREDICTOR_ITERATION
        assert result.iterations == solver.stats[-1].iterations > 0
    for block in (assembler.layout.u_slice, assembler.layout.face_slice):
        assert _relative(result.correction[block], expected[block]) < 1e-7


def test_elastic_ksp_is_built_once_and_separate() -> None:
    assembler = CASES["3d-mixed"]()
    jacobian = AnalyticalJacobian(assembler)
    old, evaluation = MaterialHistory.zeros(assembler.layout), _plastic_evaluation(assembler)
    rhs = np.ones(assembler.layout.size)
    with KrylovSolver(KrylovOptions(matrix_update="in_place")) as solver:
        solver.bind(assembler, jacobian)
        solver.solve_elastic(rhs)
        elastic_ksp, elastic_mat = solver._elastic_ksp, solver._elastic_mat
        solver.solve(old, evaluation, rhs, iteration=0)
        solver.solve_elastic(2.0 * rhs)
        assert solver._elastic_ksp is elastic_ksp and solver._elastic_mat is elastic_mat
        # The Newton KSP keeps its in-place operator (and with it the reusable PC).
        operator, _ = solver._ksp.getOperators()
        assert operator.handle == solver._mat.handle
        assert solver._elastic_ksp is not solver._ksp
        assert elastic_ksp.getOptionsPrefix() != solver._ksp.getOptionsPrefix()
        # The elastic preconditioner is set up once; the second solve only iterates.
        first, second = (s for s in solver.stats if s.iteration == PREDICTOR_ITERATION)
        assert second.setup_time < first.setup_time
    assert solver._elastic_ksp is None and solver._elastic_mat is None and solver._mat is None


@pytest.mark.parametrize("dim", [2, 3])
def test_solve_elastic_without_free_faces_never_calls_petsc(dim: int) -> None:
    assembler = _assembler(_rotated_cell(dim))
    jacobian = AnalyticalJacobian(assembler)
    rhs = np.random.default_rng(8).normal(size=assembler.layout.size)
    with KrylovSolver() as solver:
        solver.bind(assembler, jacobian)
        result = solver.solve_elastic(rhs)
        assert solver._elastic_ksp is None and not solver.stats
    expected = sps.linalg.spsolve(sps.csc_matrix(jacobian.L), rhs)
    assert _relative(result.correction, expected) < 1e-12


def test_solve_elastic_failure_and_fallback() -> None:
    assembler = CASES["2d-mixed"]()
    jacobian = AnalyticalJacobian(assembler)
    rhs = np.ones(assembler.layout.size)
    expected = sps.linalg.spsolve(sps.csc_matrix(jacobian.L), rhs)
    with KrylovSolver(KrylovOptions(max_iterations=1, rtol=1e-12)) as failing:
        failing.bind(assembler, jacobian)
        with pytest.raises(NewtonDivergenceError, match="elastic predictor"):
            failing.solve_elastic(rhs)
    options = KrylovOptions(max_iterations=1, rtol=1e-12, fallback_to_direct=True)
    with KrylovSolver(options) as solver:
        solver.bind(assembler, jacobian)
        result = solver.solve_elastic(rhs)
        assert solver.stats[-1].fallback
    assert _relative(result.correction, expected) < 1e-12


def test_controller_with_krylov_predictor() -> None:
    assembler, reference = _plane_strain()
    plain = _controller(assembler, reference, DirectSolver())
    plain.run([0.5, 1.0, 0.6])
    with KrylovSolver(KrylovOptions(rtol=1e-8)) as solver:
        predicted = _controller(assembler, reference, solver, predictor="elastic")
        predicted.run([0.5, 1.0, 0.6])
        predictor_solves = [s for s in solver.stats if s.iteration == PREDICTOR_ITERATION]
    assert len(predictor_solves) == 3                       # one per load step
    assert all(record.predictor_time > 0.0 for record in predicted.steps)
    for name in ("x", "face_displacement"):
        assert _relative(getattr(predicted.state, name), getattr(plain.state, name)) < 1e-6


def test_krylov_solver_is_elastic_predictor() -> None:
    with KrylovSolver() as solver:
        assert isinstance(solver, ElasticPredictor)
    with pytest.raises(RuntimeError, match="closed"):
        solver.solve_elastic(np.zeros(1))
