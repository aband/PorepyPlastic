from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from core.jacobian import AnalyticalJacobian
from core.layout import FloatArray
from core.newton import BlockTolerances, NewtonResult, solve_newton
from core.residual import ResidualAssembler, LoadIncrement
from core.state import SolutionState
from utils.load_schedule import(
    LoadAttempt,
    LoadStepError,
    _LoadStepSchedule,
)

from utils.newton_utils import(
    BacktrackingOptions,
    NewtonConvergenceError,
    _residual_block_norms,
    _validate_newton_options,
)

from core.linear_solvers import LinearSolver

@dataclass(eq=False)
class LoadStepResult:
    load_factor: float
    state: SolutionState
    newton: NewtonResult
    residual: FloatArray # Total residual of the accepted state

class LoadStepController(_LoadStepSchedule[LoadStepResult]):
    """Proportional loading with fixed geometry and material parameters."""
    def __init__(
        self,
        assembler: ResidualAssembler,
        committed: SolutionState,
        reference_dirichlet: FloatArray,
        *,
        reference_neumann: FloatArray | None = None,
        reference_cell_source: FloatArray | None = None,
        initial_load_factor: float = 0.0,
        max_iterations: int = 20,
        atol: BlockTolerances = (1e-5, 1e-12, 1e-12, 1e-5),
        rtol: BlockTolerances = (1e-8, 1e-8, 1e-8, 1e-8),
        line_search: BacktrackingOptions | None = None,
        linear_solver: LinearSolver | None = None,
    ) -> None:
        self._assembler = assembler
        layout = assembler.layout

        factor = float(initial_load_factor)
        if not np.isfinite(factor):
            raise ValueError("Initial load factor must be finite.")

        _validate_newton_options(max_iterations, atol, rtol, blocks=4)
        if line_search is not None and not isinstance(line_search, BacktrackingOptions):
            raise ValueError("line_search must be BacktrackingOptions or None.")
        if linear_solver is not None and not isinstance(linear_solver, LinearSolver):
            raise ValueError("linear_solver must implement LinearSolver or None.")

        self._committed = committed.copy()
        for name, shape in (
            ("x", (layout.cell_size,)),
            ("face_displacement", (layout.dim, layout.num_faces)),
            ("side_traction", (layout.dim, assembler.operators.num_sides)),
            ("neumann_values", (layout.dim, assembler.neumann_faces.size)),
            ("cell_source", (layout.cell_size,)),
        ):
            if getattr(self._committed, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")

        if (
            self._committed.material.dim != layout.dim
            or self._committed.material.num_cells != layout.num_cells
        ):
            raise ValueError("Material history does not match the layout.")

        # Total stored strain must match the total face-displacement field.
        reconstructed = assembler.operators.green_gauss_reconstruction(
            self._committed.face_displacement
        )
        stored = self._committed.material.epsilon

        if not np.allclose(stored, reconstructed, rtol=1e-8, atol=1e-12):
            max_error = float(np.max(np.abs(stored - reconstructed)))
            raise ValueError(
                "Committed total strain does not match face displacement: "
                f"maximum absolute difference = {max_error:.3e}."
            )

        reference = LoadIncrement(
            np.asarray(reference_dirichlet),
            np.zeros((layout.dim, assembler.neumann_faces.size))
            if reference_neumann is None
            else np.asarray(reference_neumann),
            np.zeros(layout.cell_size)
            if reference_cell_source is None
            else np.asarray(reference_cell_source),
        )
        assembler._validate_inputs(np.zeros(layout.size), reference)

        self._reference = LoadIncrement(
            np.array(
                reference.dirichlet_increment, dtype=np.float64, copy=True,
            ),
            np.array(
                reference.neumann_increment, dtype=np.float64, copy=True,
            ),
            np.array(
                reference.cell_source_increment, dtype=np.float64, copy=True,
            ),
        )

        self._jacobian = AnalyticalJacobian(assembler)

        if linear_solver is not None:
            linear_solver.bind(assembler, self._jacobian)
        self._linear_solver = linear_solver

        self._committed_residual = assemble_state_residual(
            assembler, self._committed,
        )
        self._blocks = (
            layout.u_slice,
            layout.r_slice,
            layout.p_slice,
            layout.face_slice,
        )

        self._load_factor = factor
        self._max_iterations = max_iterations
        self._atol, self._rtol = deepcopy(atol), deepcopy(rtol)
        self._line_search = line_search
        self._steps: list[LoadStepResult] = []
        self._attempts: list[LoadAttempt] = []

    @property
    def state(self) -> SolutionState:
        return self._committed.copy()

    @property
    def load_factor(self) -> float:
        return self._load_factor

    @property
    def steps(self) -> tuple[LoadStepResult, ...]:
        return tuple(deepcopy(self._steps))

    @property
    def attempts(self) -> tuple[LoadAttempt, ...]:
        return tuple(self._attempts)

    def advance(self, load_factor: float) -> LoadStepResult:
        factor = float(load_factor)
        if not np.isfinite(factor):
            raise ValueError("Target load factor must be finite.")

        assembler = self._assembler
        layout = assembler.layout
        old = self._committed
        start = self._load_factor

        try:
            # Absolute targets and increments from the accepted state.
            target_d = factor * self._reference.dirichlet_increment
            target_t = factor * self._reference.neumann_increment
            target_b = factor * self._reference.cell_source_increment

            loads = LoadIncrement(
                target_d
                - old.face_displacement[:, layout.dirichlet_faces],
                target_t - old.neumann_values,
                target_b - old.cell_source,
            )

            newton = solve_newton(
                assembler,
                loads,
                old.material,
                jacobian=self._jacobian,
                committed_residual=self._committed_residual,
                max_iterations=self._max_iterations,
                atol=self._atol,
                rtol=self._rtol,
                line_search=self._line_search,
                linear_solver=self._linear_solver,
            )

            evaluation = newton.evaluation
            trial = evaluation.material_trial
            if trial is None:
                raise RuntimeError(
                    "Converged evaluation has no material trial."
                )

            faces = old.face_displacement + evaluation.face_increment
            faces[:, layout.dirichlet_faces] = target_d

            candidate = SolutionState(
                x=old.x + newton.z[layout.cell_slice],
                face_displacement=faces,
                side_traction=(
                    old.side_traction + evaluation.traction_increment
                ),
                material=trial.history,
                neumann_values=target_t,
                cell_source=target_b,
            )

            # Verify the accumulated state; retain this residual for reuse.
            residual = assemble_state_residual(assembler, candidate)
            norms = _residual_block_norms(
                residual, layout.size, self._blocks,
            )
            if np.any(norms > newton.thresholds):
                raise NewtonConvergenceError(
                    "Accumulated state exceeds Newton residual thresholds."
                )

            record = LoadStepResult(factor, candidate, newton, residual)
            returned = deepcopy(record)

        except Exception as error:
            self._attempts.append(
                LoadAttempt(start, factor, None, str(error))
            )
            raise LoadStepError(
                len(self._steps) + 1, factor, error,
            ) from error

        # Acceptance occurs only after all calculations and checks succeed.
        self._steps.append(record)
        self._attempts.append(
            LoadAttempt(start, factor, newton.iterations)
        )
        self._committed, self._committed_residual, self._load_factor = (
            candidate,
            residual,
            factor,
        )
        return returned

def assemble_state_residual(
    assembler: ResidualAssembler,
    state: SolutionState,
) -> FloatArray:
    """Return total-state residual [R_u, R_r, R_p, R_face].

    Assumes state belongs to this assembler's layout.
    Does not update material history or modify the state.
    """
    cell_residual = (
        assembler._cell_flux_balance(
            state.face_displacement,
            state.side_traction,
        )
        - assembler.accumulation @ state.x
        - state.cell_source
    )

    face_residual = assembler._free_face_balance(
        state.side_traction,
        state.neumann_values,
    )

    residual = assembler.layout.pack(
        cell_residual,
        face_residual,
    )

    if not np.all(np.isfinite(residual)):
        raise ValueError("Total-state residual must remain finite.")

    return residual