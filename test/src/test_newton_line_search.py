"""Backtracking line search in core.newton and core.loading."""

import numpy as np
import pytest

import core.newton as core_newton
from core.loading import LoadStepController
from core.newton import solve_newton
from core.residual import LoadIncrement, ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory, SolutionState
from utils.load_schedule import LoadStepError
from utils.newton_utils import BacktrackingOptions, LocalUpdateError, NewtonConvergenceError
from utils.problem import PlaneStrainProblem

# Unloading after plastic loading: full Newton steps cycle on this path.
PATH = [0.25, 0.5, 0.75, 1.0, 0.6]


def _boundary(x: np.ndarray) -> np.ndarray:
    return np.vstack((6e-3 * x[0] ** 2 * x[1] + 1e-3 * x[1], -4e-3 * x[0] * x[1]))


def _setup(cells: int = 12) -> tuple[ResidualAssembler, np.ndarray]:
    problem = PlaneStrainProblem(cells_per_axis=cells, boundary_displacement=_boundary)
    problem.hardening_parameters.sigma_u, problem.hardening_parameters.delta = 400e6, 30.0
    problem.prepare_simulation()
    assembler = ResidualAssembler(problem.grid, problem.bc, problem.material_points)
    return assembler, problem.bc_values[:, assembler.layout.dirichlet_faces]


def _controller(assembler: ResidualAssembler, reference: np.ndarray, **options: object) -> LoadStepController:
    layout = assembler.layout
    state = SolutionState.zeros(
        layout, num_sides=assembler.operators.num_sides, num_neumann_faces=assembler.neumann_faces.size,
    )
    return LoadStepController(assembler, state, reference, **options)  # type: ignore[arg-type]


def _first_step(assembler: ResidualAssembler, reference: np.ndarray) -> LoadIncrement:
    layout = assembler.layout
    return LoadIncrement(0.25 * reference, np.zeros((layout.dim, 0)), np.zeros(layout.cell_size))


def test_full_step_path_is_unchanged_when_every_full_step_is_accepted() -> None:
    assembler, reference = _setup()
    loads, old = _first_step(assembler, reference), MaterialHistory.zeros(assembler.layout)
    full = solve_newton(assembler, loads, old)
    searched = solve_newton(assembler, loads, old, line_search=BacktrackingOptions())
    assert searched.iterations == full.iterations
    np.testing.assert_array_equal(searched.z, full.z)
    np.testing.assert_array_equal(searched.step_length, 1.0)
    np.testing.assert_array_equal(full.step_length, 1.0)
    assert searched.evaluations == full.evaluations == full.iterations + 1


def test_line_search_converges_where_full_steps_cycle() -> None:
    assembler, reference = _setup()
    full = _controller(assembler, reference)
    with pytest.raises(LoadStepError) as caught:
        full.run(PATH)
    assert caught.value.load_factor == 0.6
    assert isinstance(caught.value.__cause__, NewtonConvergenceError)

    searched = _controller(assembler, reference, line_search=BacktrackingOptions())
    records = searched.run(PATH)
    assert searched.load_factor == 0.6
    lengths = records[-1].newton.step_length
    assert np.all((lengths > 0.0) & (lengths <= 1.0)) and np.any(lengths < 1.0)


def test_exhausted_search_raises_recoverable_convergence_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assembler, reference = _setup(4)
    solve = core_newton._solve_linear_correction
    # An ascent direction: no step length can satisfy the Armijo condition.
    monkeypatch.setattr(core_newton, "_solve_linear_correction", lambda *args: -solve(*args))
    with pytest.raises(NewtonConvergenceError, match="Line search failed"):
        solve_newton(
            assembler, _first_step(assembler, reference), MaterialHistory.zeros(assembler.layout),
            line_search=BacktrackingOptions(max_backtracks=3),
        )


def test_local_failure_in_a_trial_shortens_the_step(monkeypatch: pytest.MonkeyPatch) -> None:
    assembler, reference = _setup(4)
    evaluate, calls = assembler.evaluate, [0]

    def fail_first_trial(*args: object, **kwargs: object) -> ResidualEvaluation:
        calls[0] += 1
        if calls[0] == 2:  # the full-step trial of the first correction
            raise LocalUpdateError("Return map failed in cell 0 (injected).")
        return evaluate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(assembler, "evaluate", fail_first_trial)
    result = solve_newton(
        assembler, _first_step(assembler, reference), MaterialHistory.zeros(assembler.layout),
        line_search=BacktrackingOptions(factor=0.5),
    )
    assert result.step_length[0] == 0.5


def test_local_failure_without_line_search_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    assembler, reference = _setup(4)
    evaluate, calls = assembler.evaluate, [0]

    def fail_first_trial(*args: object, **kwargs: object) -> ResidualEvaluation:
        calls[0] += 1
        if calls[0] == 2:
            raise LocalUpdateError("Return map failed in cell 0 (injected).")
        return evaluate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(assembler, "evaluate", fail_first_trial)
    with pytest.raises(LocalUpdateError):
        solve_newton(assembler, _first_step(assembler, reference), MaterialHistory.zeros(assembler.layout))


@pytest.mark.parametrize("options", [{"factor": 1.0}, {"factor": 0.0}, {"armijo": 0.5}, {"max_backtracks": -1}])
def test_invalid_options_are_rejected(options: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BacktrackingOptions(**options)  # type: ignore[arg-type]


def test_invalid_line_search_type_is_rejected() -> None:
    assembler, reference = _setup(2)
    with pytest.raises(ValueError, match="BacktrackingOptions"):
        solve_newton(
            assembler, _first_step(assembler, reference), MaterialHistory.zeros(assembler.layout),
            line_search=0.5,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="BacktrackingOptions"):
        _controller(assembler, reference, line_search=0.5)
