"""Pluggable linear solvers in core.newton and core.loading."""

import numpy as np
import pytest

from core.layout import FloatArray
import scipy.sparse as sps

from core.jacobian import AnalyticalJacobian
from core.linear_solvers import (
    CondensedDirectSolver,
    DirectSolver,
    ElasticPredictor,
    LinearSolveResult,
    LinearSolver,
    _BoundSolver,
)
from core.loading import LoadStepController
from core.newton import solve_newton
from core.residual import LoadIncrement, ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory, SolutionState
from utils.load_schedule import AdaptiveLoadOptions, LoadStepError
from utils.newton_utils import NewtonDivergenceError
from utils.problem import PlaneStrainProblem


def _boundary(x: np.ndarray) -> np.ndarray:
    return np.vstack((6e-3 * x[0] ** 2 * x[1] + 1e-3 * x[1], -4e-3 * x[0] * x[1]))


def _setup(cells: int = 6) -> tuple[ResidualAssembler, np.ndarray]:
    """Small plane-strain problem that yields under the reference load."""
    problem = PlaneStrainProblem(cells_per_axis=cells, boundary_displacement=_boundary)
    problem.hardening_parameters.sigma_u, problem.hardening_parameters.delta = 400e6, 30.0
    problem.prepare_simulation()
    assembler = ResidualAssembler(problem.grid, problem.bc, problem.material_points)
    return assembler, problem.bc_values[:, assembler.layout.dirichlet_faces]


def _loads(assembler: ResidualAssembler, reference: np.ndarray, factor: float = 0.5) -> LoadIncrement:
    layout = assembler.layout
    return LoadIncrement(factor * reference, np.zeros((layout.dim, 0)), np.zeros(layout.cell_size))


def _controller(
    assembler: ResidualAssembler, reference: np.ndarray, linear_solver: object = None,
) -> LoadStepController:
    layout = assembler.layout
    state = SolutionState.zeros(
        layout, num_sides=assembler.operators.num_sides, num_neumann_faces=assembler.neumann_faces.size,
    )
    return LoadStepController(assembler, state, reference, linear_solver=linear_solver)  # type: ignore[arg-type]


class _CountingSolver(DirectSolver):
    """Direct solves that count calls and report a fixed iteration number."""

    def __init__(self, iterations: int = 0) -> None:
        self.steps = 0
        self.solves = 0
        self.iterations = iterations

    def begin_step(self) -> None:
        self.steps += 1

    def solve(
        self, committed: MaterialHistory, evaluation: ResidualEvaluation,
        residual: FloatArray, *, iteration: int,
    ) -> LinearSolveResult:
        self.solves += 1
        result = super().solve(committed, evaluation, residual, iteration=iteration)
        return LinearSolveResult(result.correction, self.iterations, result.elapsed)


class _FailOnce(DirectSolver):
    """First solve fails as a linear-solver breakdown would."""

    def __init__(self) -> None:
        self.failed = False

    def solve(
        self, committed: MaterialHistory, evaluation: ResidualEvaluation,
        residual: FloatArray, *, iteration: int,
    ) -> LinearSolveResult:
        if not self.failed:
            self.failed = True
            raise NewtonDivergenceError("Injected linear-solver failure.")
        return super().solve(committed, evaluation, residual, iteration=iteration)


class _FixedCorrection(_BoundSolver):
    def __init__(self, correction: FloatArray) -> None:
        self.correction = correction

    def solve(
        self, committed: MaterialHistory, evaluation: ResidualEvaluation,
        residual: FloatArray, *, iteration: int,
    ) -> LinearSolveResult:
        return LinearSolveResult(self.correction)


def test_direct_solver_matches_default_path() -> None:
    assembler, reference = _setup()
    loads, old = _loads(assembler, reference), MaterialHistory.zeros(assembler.layout)
    default = solve_newton(assembler, loads, old)
    direct = solve_newton(assembler, loads, old, linear_solver=DirectSolver())
    assert default.iterations > 1
    assert np.any(default.evaluation.material_trial.return_map.dgamma > 0.0)  # type: ignore[union-attr]
    assert direct.iterations == default.iterations
    assert direct.evaluations == default.evaluations
    np.testing.assert_array_equal(direct.z, default.z)
    np.testing.assert_array_equal(direct.residual_norms, default.residual_norms)
    np.testing.assert_array_equal(direct.step_length, default.step_length)


def test_controller_with_direct_solver_is_identical() -> None:
    assembler, reference = _setup()
    default = _controller(assembler, reference)
    direct = _controller(assembler, reference, DirectSolver())
    default.run([0.5, 1.0])
    direct.run([0.5, 1.0])
    for name in ("x", "face_displacement", "side_traction"):
        np.testing.assert_array_equal(getattr(direct.state, name), getattr(default.state, name))
    for name in ("epsilon", "stress", "plastic_strain", "backstress", "alpha"):
        np.testing.assert_array_equal(
            getattr(direct.state.material, name), getattr(default.state.material, name),
        )


@pytest.mark.parametrize("linear_solver", [None, "direct"])
def test_diagnostics_have_one_entry_per_correction(linear_solver: str | None) -> None:
    assembler, reference = _setup()
    result = solve_newton(
        assembler, _loads(assembler, reference), MaterialHistory.zeros(assembler.layout),
        linear_solver=None if linear_solver is None else DirectSolver(),
    )
    assert result.linear_iterations.dtype == np.int64
    assert result.linear_iterations.shape == result.linear_times.shape == (result.iterations,)
    np.testing.assert_array_equal(result.linear_iterations, 0)
    assert np.all(result.linear_times >= 0.0)


def test_solver_iterations_are_reported() -> None:
    assembler, reference = _setup()
    solver = _CountingSolver(iterations=7)
    result = solve_newton(
        assembler, _loads(assembler, reference), MaterialHistory.zeros(assembler.layout),
        linear_solver=solver,
    )
    assert solver.solves == result.iterations > 0
    np.testing.assert_array_equal(result.linear_iterations, 7)


def test_begin_step_called_once_per_load_step() -> None:
    assembler, reference = _setup()
    solver = _CountingSolver()
    controller = _controller(assembler, reference, solver)
    records = controller.run([0.5, 1.0])
    assert solver.steps == 2
    assert solver.solves == sum(record.newton.iterations for record in records)


def test_linear_failure_is_recovered_by_adaptive_cutback() -> None:
    assembler, reference = _setup()
    solver = _FailOnce()
    controller = _controller(assembler, reference, solver)
    records = controller.run_adaptive([0.5], options=AdaptiveLoadOptions(
        initial_increment=0.5, min_increment=0.125, max_increment=0.5, growth_factor=1.0,
    ))
    assert solver.failed
    assert [record.load_factor for record in records] == pytest.approx([0.25, 0.5])
    assert [attempt.accepted for attempt in controller.attempts] == [False, True, True]
    assert controller.attempts[0].error == "Injected linear-solver failure."
    assert controller.load_factor == 0.5


@pytest.mark.parametrize(("correction", "error"), [
    ("nan", NewtonDivergenceError),
    ("short", RuntimeError),
])
def test_invalid_corrections_are_rejected(correction: str, error: type[Exception]) -> None:
    assembler, reference = _setup()
    size = assembler.layout.size
    values = np.full(size, np.nan) if correction == "nan" else np.zeros(size - 1)
    with pytest.raises(error):
        solve_newton(
            assembler, _loads(assembler, reference), MaterialHistory.zeros(assembler.layout),
            linear_solver=_FixedCorrection(values),
        )


def test_non_solver_is_rejected_by_newton() -> None:
    assembler, reference = _setup(2)
    with pytest.raises(ValueError, match="LinearSolver"):
        solve_newton(
            assembler, _loads(assembler, reference), MaterialHistory.zeros(assembler.layout),
            linear_solver=0.5,  # type: ignore[arg-type]
        )


def test_non_solver_is_rejected_by_controller() -> None:
    assembler, reference = _setup(2)
    with pytest.raises(ValueError, match="LinearSolver"):
        _controller(assembler, reference, 0.5)


def test_direct_solver_implements_protocol() -> None:
    assert isinstance(DirectSolver(), LinearSolver)
    assert not isinstance(0.5, LinearSolver)


def test_solver_is_bound_to_one_assembler() -> None:
    assembler, reference = _setup(2)
    other, other_reference = _setup(3)
    solver = DirectSolver()
    _controller(assembler, reference, solver)
    with pytest.raises(ValueError, match="another assembler"):
        _controller(other, other_reference, solver)
    with pytest.raises(ValueError, match="another assembler"):
        solve_newton(
            other, _loads(other, other_reference), MaterialHistory.zeros(other.layout),
            linear_solver=solver,
        )


def test_bind_rejects_jacobian_of_another_assembler() -> None:
    from core.jacobian import AnalyticalJacobian

    assembler, _ = _setup(2)
    other, _ = _setup(3)
    with pytest.raises(ValueError, match="same|supplied"):
        DirectSolver().bind(assembler, AnalyticalJacobian(other))


def test_unbound_solver_cannot_solve() -> None:
    assembler, reference = _setup(2)
    evaluation = assembler.evaluate(
        np.zeros(assembler.layout.size), _loads(assembler, reference),
        MaterialHistory.zeros(assembler.layout),
    )
    with pytest.raises(RuntimeError, match="bound"):
        DirectSolver().solve(
            MaterialHistory.zeros(assembler.layout), evaluation,
            evaluation.residual, iteration=0,
        )


# ---------------------------------------------------------------- elastic predictor
@pytest.mark.parametrize("kind", [DirectSolver, CondensedDirectSolver])
def test_direct_solve_elastic_matches_full_lu(kind: type[DirectSolver]) -> None:
    assembler, _ = _setup()
    jacobian = AnalyticalJacobian(assembler)
    solver = kind()
    assert isinstance(solver, ElasticPredictor)
    solver.bind(assembler, jacobian)
    rhs = np.random.default_rng(0).normal(size=assembler.layout.size)
    expected = sps.linalg.spsolve(sps.csc_matrix(jacobian.L), rhs)
    first = solver.solve_elastic(rhs)
    factor = solver._elastic_factor
    second = solver.solve_elastic(2.0 * rhs)
    assert factor is not None and solver._elastic_factor is factor     # factored once
    for actual, scale in ((first.correction, 1.0), (second.correction, 2.0)):
        np.testing.assert_allclose(actual, scale * expected, rtol=1e-8, atol=1e-12 * np.abs(expected).max())


def test_direct_elastic_factor_is_dropped_on_new_jacobian() -> None:
    assembler, _ = _setup()
    solver, jacobian = DirectSolver(), AnalyticalJacobian(assembler)
    solver.bind(assembler, jacobian)
    solver.solve_elastic(np.ones(assembler.layout.size))
    assert solver._elastic_factor is not None
    solver.bind(assembler, jacobian)                      # same Jacobian: kept
    assert solver._elastic_factor is not None
    solver.bind(assembler, AnalyticalJacobian(assembler))
    assert solver._elastic_factor is None


def _predictor_controller(
    assembler: ResidualAssembler, reference: np.ndarray, linear_solver: object = None, **kwargs: object,
) -> LoadStepController:
    layout = assembler.layout
    state = SolutionState.zeros(
        layout, num_sides=assembler.operators.num_sides, num_neumann_faces=assembler.neumann_faces.size,
    )
    return LoadStepController(
        assembler, state, reference, linear_solver=linear_solver, **kwargs,  # type: ignore[arg-type]
    )


def test_elastic_predictor_solves_elastic_steps_without_corrections() -> None:
    """While every cell stays elastic, the predictor is the solution of the step."""
    assembler, reference = _setup()
    elastic = 0.02 * reference                            # below yield on the whole path
    controller = _predictor_controller(assembler, elastic, predictor="elastic")
    records = controller.run([0.5, 1.0, -0.4])
    assert np.all(controller.state.material.alpha == 0.0)
    assert [record.newton.iterations for record in records] == [0, 0, 0]
    zero = _predictor_controller(assembler, elastic)
    zero.run([0.5, 1.0, -0.4])
    np.testing.assert_allclose(controller.state.x, zero.state.x, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("linear_solver", [None, "direct", "condensed"])
def test_elastic_predictor_matches_zero_start_on_plastic_path(linear_solver: str | None) -> None:
    assembler, reference = _setup()
    solver = {None: None, "direct": DirectSolver(), "condensed": CondensedDirectSolver()}[linear_solver]
    predicted = _predictor_controller(assembler, reference, solver, predictor="elastic")
    predicted.run([0.5, 1.0, 0.6])
    zero = _predictor_controller(assembler, reference)
    zero.run([0.5, 1.0, 0.6])
    assert np.any(predicted.state.material.alpha > 0.0)
    assert all(record.predictor_time > 0.0 for record in predicted.steps)
    assert all(record.predictor_time == 0.0 for record in zero.steps)
    for name in ("x", "face_displacement"):
        actual, expected = getattr(predicted.state, name), getattr(zero.state, name)
        assert np.abs(actual - expected).max() <= 1e-6 * np.abs(expected).max()
    assert sum(r.newton.iterations for r in predicted.steps) <= sum(r.newton.iterations for r in zero.steps)


class _CountingPredictor(DirectSolver):
    def __init__(self) -> None:
        self.predictions = 0

    def solve_elastic(self, rhs: FloatArray) -> LinearSolveResult:
        self.predictions += 1
        return super().solve_elastic(rhs)


def test_predictor_runs_once_per_step_and_attempt() -> None:
    assembler, reference = _setup()
    solver = _CountingPredictor()
    controller = _predictor_controller(assembler, reference, solver, predictor="elastic")
    controller.run([0.5, 1.0])
    assert solver.predictions == 2
    _predictor_controller(assembler, reference, solver)     # predictor="zero": never used
    assert solver.predictions == 2


class _NoPredictor(_BoundSolver):
    def solve(
        self, committed: MaterialHistory, evaluation: ResidualEvaluation,
        residual: FloatArray, *, iteration: int,
    ) -> LinearSolveResult:
        raise AssertionError("not used")


class _BadPredictor(DirectSolver):
    def solve_elastic(self, rhs: FloatArray) -> LinearSolveResult:
        return LinearSolveResult(np.full(rhs.size, np.nan))


def test_predictor_validation_and_failure() -> None:
    assembler, reference = _setup()
    with pytest.raises(ValueError, match="predictor must be"):
        _predictor_controller(assembler, reference, predictor="linear")
    with pytest.raises(ValueError, match="solve_elastic"):
        _predictor_controller(assembler, reference, _NoPredictor(), predictor="elastic")
    controller = _predictor_controller(assembler, reference, _BadPredictor(), predictor="elastic")
    before = controller.state
    with pytest.raises(LoadStepError, match="non-finite starting guess"):
        controller.advance(0.5)
    assert controller.load_factor == 0.0 and not controller.steps
    assert [attempt.accepted for attempt in controller.attempts] == [False]
    np.testing.assert_array_equal(controller.state.x, before.x)
