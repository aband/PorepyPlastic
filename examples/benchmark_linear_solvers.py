"""Benchmark linear solvers for the core retained-face Newton solver.

Run from the repository root:

    python -m examples.benchmark_linear_solvers                   # default cases
    python -m examples.benchmark_linear_solvers --case 2d:96 \\
        --solver krylov:preconditioner=gamg,rtol=1e-6 \\
        --solver krylov:preconditioner=gamg,rtol=1e-3
    python -m examples.benchmark_linear_solvers --case 2d:24 --solver direct --solver krylov
    python -m examples.benchmark_linear_solvers --case 3d:12 \\
        --solver krylov:preconditioner=ilu,petsc.pc_factor_levels=1 --json results/bench.json

Cases are "<dim>d:<cells per axis>" on the unit square/cube with all faces
Dirichlet and a fixed load path (loading, then partial unloading).

Solver specs are "<kind>[:key=value,...]":
    default            the controller's built-in path (linear_solver=None)
    direct             core.linear_solvers.DirectSolver
    condensed-direct   core.linear_solvers.CondensedDirectSolver
    krylov             core.krylov.KrylovSolver; keys are KrylovOptions fields,
                       and "petsc.<name>=<value>" goes into petsc_options.
New KrylovOptions fields are accepted automatically.

--controller key=value is forwarded to LoadStepController (for example a
future predictor=elastic option). Every run is compared with a reference run
(by default the first solver), using the relative max-norm difference of the
final x, face displacement and plastic strain.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import porepy as pp

from core.layout import FloatArray
from core.linear_solvers import CondensedDirectSolver, DirectSolver, LinearSolver
from core.loading import LoadStepController
from core.residual import ResidualAssembler
from core.state import SolutionState
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import HardeningParameters, isotropic_hardening_K, kinematic_hardening_H
from utils.load_schedule import LoadStepError
from utils.newton_utils import BacktrackingOptions
from utils.problem import PlaneStrainProblem

DEFAULT_CASES = ("2d:96", "3d:12", "3d:16")
DEFAULT_SOLVERS = {2: "krylov:preconditioner=gamg", 3: "krylov:preconditioner=ilu"}
DEFAULT_PATH = (0.25, 0.5, 0.75, 1.0, 0.6)

# Keyword arguments the benchmark sets itself; --controller must not override them.
_RESERVED_CONTROLLER_KEYS = frozenset(
    {"assembler", "committed", "reference_dirichlet", "line_search", "linear_solver"}
)


# --------------------------------------------------------------------------- cases
def _boundary_2d(x: FloatArray) -> FloatArray:
    return np.vstack((6e-3 * x[0] ** 2 * x[1] + 1e-3 * x[1], -4e-3 * x[0] * x[1]))


def _boundary_3d(x: FloatArray) -> FloatArray:
    return np.vstack((
        6e-3 * x[0] ** 2 * x[1] + 1e-3 * x[1],
        -4e-3 * x[0] * x[1],
        2e-3 * x[0] * x[2],
    ))


@dataclass(frozen=True)
class Case:
    name: str
    dim: int
    cells: int


def parse_case(text: str) -> Case:
    """Parse "<dim>d:<cells per axis>", e.g. "2d:96" or "3d:12"."""
    head, sep, tail = text.strip().lower().partition(":")
    if not sep or head not in ("2d", "3d"):
        raise ValueError(f"Case must look like '2d:96' or '3d:12'; got {text!r}.")
    try:
        cells = int(tail)
    except ValueError:
        raise ValueError(f"Cells per axis must be an integer; got {tail!r}.") from None
    if cells < 1:
        raise ValueError("Cells per axis must be positive.")
    return Case(f"{head}:{cells}", int(head[0]), cells)


def build_problem(case: Case) -> tuple[ResidualAssembler, FloatArray]:
    """Assembler and reference Dirichlet values on the Dirichlet faces."""
    if case.dim == 2:
        problem = PlaneStrainProblem(cells_per_axis=case.cells, boundary_displacement=_boundary_2d)
        problem.hardening_parameters.sigma_u = 400e6
        problem.hardening_parameters.delta = 30.0
        problem.prepare_simulation()
        assembler = ResidualAssembler(problem.grid, problem.bc, problem.material_points)
        return assembler, problem.bc_values[:, assembler.layout.dirichlet_faces]

    grid = pp.CartGrid(np.array([case.cells] * 3), np.array([1.0, 1.0, 1.0]))
    grid.compute_geometry()
    boundary = grid.get_all_boundary_faces()
    bc = pp.BoundaryConditionVectorial(grid, boundary, "dir")
    material = _materialProperty()
    material.young_modulus, material.poisson_ratio = 210e9, 0.3
    hardening = HardeningParameters(sigma_y=250e6, sigma_u=400e6, H_bar=1e9, theta=0.5, delta=30.0)
    points = [
        MaterialPoint(material, hardening, vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H)
        for _ in range(grid.num_cells)
    ]
    assembler = ResidualAssembler(grid, bc, points)
    return assembler, _boundary_3d(grid.face_centers)[:, assembler.layout.dirichlet_faces]


# ------------------------------------------------------------------------- options
def parse_value(text: str) -> Any:
    """'true'/'false' -> bool, then int, then float, otherwise the string itself."""
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text.strip()


def parse_pairs(text: str) -> dict[str, str]:
    """'a=1,b=x' -> {'a': '1', 'b': 'x'} (values unparsed)."""
    pairs: dict[str, str] = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"Expected key=value; got {item!r}.")
        if key.strip() in pairs:
            raise ValueError(f"Option {key.strip()!r} given twice.")
        pairs[key.strip()] = value.strip()
    return pairs


@dataclass(frozen=True)
class SolverSpec:
    label: str
    kind: str
    options: tuple[tuple[str, str], ...] = ()


_KINDS = ("default", "direct", "condensed-direct", "krylov")


def parse_solver(text: str) -> SolverSpec:
    kind, _, rest = text.strip().partition(":")
    kind = kind.strip().lower()
    if kind not in _KINDS:
        raise ValueError(f"Solver kind must be one of {_KINDS}; got {kind!r}.")
    options = parse_pairs(rest)
    if options and kind != "krylov":
        raise ValueError(f"Solver {kind!r} takes no options.")
    return SolverSpec(text.strip(), kind, tuple(options.items()))


def krylov_options(spec: SolverSpec) -> Any:
    """Build KrylovOptions from a spec; unknown keys list the accepted ones."""
    from core.krylov import KrylovOptions

    names = {f.name for f in dataclasses.fields(KrylovOptions)} - {"petsc_options"}
    values: dict[str, Any] = {}
    petsc: dict[str, str] = {}
    for key, value in spec.options:
        if key.startswith("petsc."):
            petsc[key.removeprefix("petsc.")] = value
        elif key in names:
            values[key] = parse_value(value)
        else:
            raise ValueError(
                f"Unknown KrylovOptions field {key!r}; accepted: {sorted(names)} or petsc.<name>."
            )
    return KrylovOptions(**values, petsc_options=petsc)


def make_solver(spec: SolverSpec) -> LinearSolver | None:
    if spec.kind == "default":
        return None
    if spec.kind == "direct":
        return DirectSolver()
    if spec.kind == "condensed-direct":
        return CondensedDirectSolver()
    from core.krylov import KrylovSolver

    return KrylovSolver(krylov_options(spec))


def controller_options(pairs: Sequence[str]) -> dict[str, Any]:
    """Parse --controller key=value items against LoadStepController's keywords."""
    accepted = set(inspect.signature(LoadStepController.__init__).parameters) - {"self"}
    options: dict[str, Any] = {}
    for item in pairs:
        for key, value in parse_pairs(item).items():
            if key in _RESERVED_CONTROLLER_KEYS:
                raise ValueError(f"Controller option {key!r} is set by the benchmark.")
            if key not in accepted:
                raise ValueError(
                    f"Unknown LoadStepController option {key!r}; "
                    f"accepted: {sorted(accepted - _RESERVED_CONTROLLER_KEYS)}."
                )
            if key in options:
                raise ValueError(f"Controller option {key!r} given twice.")
            options[key] = parse_value(value)
    return options


# ----------------------------------------------------------------------------- run
@dataclass
class RunRecord:
    case: str
    solver: str
    size: int
    status: str                     # "ok" or "failed"
    setup_time: float               # controller construction (Jacobian, condensation, bind)
    run_time: float                 # load path
    newton_iterations: list[int]    # per accepted step
    backtracks: int
    residual_evaluations: int
    linear_time: float              # sum of NewtonResult.linear_times
    krylov_iterations: int
    max_krylov_iterations: int
    pc_setup_time: float | None     # KrylovSolver only
    krylov_solve_time: float | None
    fallbacks: int | None
    difference: dict[str, float] | None = None   # relative difference to the reference
    error: str | None = None
    final_state: SolutionState | None = dataclasses.field(default=None, repr=False)

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data.pop("final_state")
        return data


def _close(solver: object) -> None:
    close = getattr(solver, "close", None)
    if callable(close):
        close()


def run_once(
    case: Case,
    assembler: ResidualAssembler,
    reference_dirichlet: FloatArray,
    spec: SolverSpec,
    path: Sequence[float],
    controller_kwargs: dict[str, Any],
) -> RunRecord:
    """One load path with a fresh solver and controller; failures become records."""
    layout = assembler.layout
    solver = make_solver(spec)
    try:
        initial = SolutionState.zeros(
            layout, num_sides=assembler.operators.num_sides,
            num_neumann_faces=assembler.neumann_faces.size,
        )
        start = time.perf_counter()
        controller = LoadStepController(
            assembler, initial, reference_dirichlet,
            line_search=BacktrackingOptions(), linear_solver=solver, **controller_kwargs,
        )
        setup = time.perf_counter() - start

        status, error = "ok", None
        start = time.perf_counter()
        try:
            controller.run(path)
        except LoadStepError as failure:
            status, error = "failed", str(failure)
        run_time = time.perf_counter() - start

        steps = controller.steps
        newton = [step.newton for step in steps]
        linear = np.concatenate([n.linear_iterations for n in newton]) if newton else np.zeros(0, int)
        stats = getattr(solver, "stats", None)
        return RunRecord(
            case=case.name,
            solver=spec.label,
            size=layout.size,
            status=status,
            setup_time=setup,
            run_time=run_time,
            newton_iterations=[int(n.iterations) for n in newton],
            backtracks=int(sum(np.count_nonzero(n.step_length < 1.0) for n in newton)),
            residual_evaluations=int(sum(n.evaluations for n in newton)),
            linear_time=float(sum(n.linear_times.sum() for n in newton)),
            krylov_iterations=int(linear.sum()),
            max_krylov_iterations=int(linear.max(initial=0)),
            pc_setup_time=None if stats is None else float(sum(s.setup_time for s in stats)),
            krylov_solve_time=None if stats is None else float(sum(s.solve_time for s in stats)),
            fallbacks=None if stats is None else int(sum(s.fallback for s in stats)),
            error=error,
            final_state=controller.state if status == "ok" else None,
        )
    finally:
        _close(solver)


def relative_difference(state: SolutionState, reference: SolutionState) -> dict[str, float]:
    """Relative max-norm differences of the fields that define the final state."""
    fields: dict[str, Callable[[SolutionState], FloatArray]] = {
        "x": lambda s: s.x,
        "face_displacement": lambda s: s.face_displacement,
        "plastic_strain": lambda s: s.material.plastic_strain,
    }
    result = {}
    for name, get in fields.items():
        actual, expected = get(state), get(reference)
        scale = float(np.abs(expected).max(initial=0.0))
        result[name] = float(np.abs(actual - expected).max(initial=0.0) / max(scale, 1e-300))
    return result


def benchmark_case(
    case: Case,
    specs: Sequence[SolverSpec],
    reference: SolverSpec | None,
    *,
    path: Sequence[float],
    controller_kwargs: dict[str, Any],
    repeat: int = 1,
    report: Callable[[RunRecord], None] | None = None,
) -> list[RunRecord]:
    """Run every solver on one case; the best of `repeat` runs is kept per solver."""
    assembler, reference_dirichlet = build_problem(case)
    order = list(specs)
    if reference is not None and reference not in order:
        order.insert(0, reference)

    records: list[RunRecord] = []
    reference_state: SolutionState | None = None
    for spec in order:
        runs = [
            run_once(case, assembler, reference_dirichlet, spec, path, controller_kwargs)
            for _ in range(repeat)
        ]
        ok = [r for r in runs if r.status == "ok"]
        record = min(ok, key=lambda r: r.setup_time + r.run_time) if ok else runs[0]
        if spec == reference:
            reference_state = record.final_state
        if record.final_state is not None and reference_state is not None:
            record.difference = relative_difference(record.final_state, reference_state)
        record.final_state = None   # release memory; states are not reported
        records.append(record)
        if report is not None:
            report(record)
    return records


# -------------------------------------------------------------------------- output
HEADER = (
    f"{'case':7s} {'solver':44s} {'status':6s} {'setup':>6s} {'run':>7s} {'linear':>7s} "
    f"{'PC set':>6s} {'Krylov':>6s} {'Newton':>6s} {'bt':>3s} {'Kry its':>7s} {'max':>4s} "
    f"{'diff x':>8s} {'diff εp':>8s}  Newton per step"
)


def format_record(r: RunRecord) -> str:
    def seconds(value: float | None) -> str:
        return f"{value:6.2f}" if value is not None else f"{'-':>6s}"

    diff_x = f"{r.difference['x']:8.1e}" if r.difference else f"{'-':>8s}"
    diff_p = f"{r.difference['plastic_strain']:8.1e}" if r.difference else f"{'-':>8s}"
    line = (
        f"{r.case:7s} {r.solver[:44]:44s} {r.status:6s} {r.setup_time:6.2f} {r.run_time:7.2f} "
        f"{r.linear_time:7.2f} {seconds(r.pc_setup_time)} {seconds(r.krylov_solve_time)} "
        f"{sum(r.newton_iterations):6d} {r.backtracks:3d} {r.krylov_iterations:7d} "
        f"{r.max_krylov_iterations:4d} {diff_x} {diff_p}  {r.newton_iterations}"
    )
    if r.fallbacks:
        line += f"  fallbacks={r.fallbacks}"
    if r.error:
        line += f"\n        error: {r.error[:200]}"
    return line


def main(argv: Sequence[str] | None = None) -> list[RunRecord]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case", action="append", default=None,
                        help=f"'<dim>d:<cells>'; repeatable (default: {' '.join(DEFAULT_CASES)})")
    parser.add_argument("--solver", action="append", default=None,
                        help="solver spec; repeatable (default: GAMG in 2D, ILU(0) in 3D)")
    parser.add_argument("--reference", default=None,
                        help="solver spec for the reference state (default: first solver; 'none' to skip)")
    parser.add_argument("--controller", action="append", default=[],
                        help="LoadStepController key=value[,key=value]; repeatable")
    parser.add_argument("--path", default=",".join(map(str, DEFAULT_PATH)),
                        help="comma-separated load factors")
    parser.add_argument("--repeat", type=int, default=1, help="runs per solver; the fastest is kept")
    parser.add_argument("--json", type=Path, default=None, help="write all records to this JSON file")
    args = parser.parse_args(argv)

    if args.repeat < 1:
        parser.error("--repeat must be positive.")
    try:
        cases = [parse_case(text) for text in (args.case or DEFAULT_CASES)]
        solvers = None if args.solver is None else [parse_solver(text) for text in args.solver]
        reference = (
            None if args.reference is None or args.reference.lower() == "none"
            else parse_solver(args.reference)
        )
        controller_kwargs = controller_options(args.controller)
        path = [float(value) for value in args.path.split(",") if value.strip()]
        # Reject bad Krylov options now, not after earlier runs have finished.
        for spec in [*(solvers or []), *([reference] if reference else [])]:
            if spec.kind == "krylov":
                krylov_options(spec)
    except ValueError as error:
        parser.error(str(error))
    if not path or not all(np.isfinite(path)):
        parser.error("--path must contain finite load factors.")

    print(f"Load path {path}; controller options {controller_kwargs or '{}'}")
    print(HEADER, flush=True)
    records: list[RunRecord] = []
    with warnings.catch_warnings():
        # The unscaled full Jacobian has row sizes spanning ~1e20, so UMFPACK's condition
        # estimate always reports "(almost) singular"; it says nothing about this run.
        warnings.filterwarnings("ignore", message=r"\(almost\) singular matrix")
        for case in cases:
            specs = solvers or [parse_solver(DEFAULT_SOLVERS[case.dim])]
            # Default reference: the first solver; "--reference none" disables it.
            case_reference = reference if args.reference is not None else specs[0]
            records += benchmark_case(
                case, specs, case_reference, path=path, controller_kwargs=controller_kwargs,
                repeat=args.repeat, report=lambda r: print(format_record(r), flush=True),
            )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "path": path,
            "controller": {k: repr(v) for k, v in controller_kwargs.items()},
            "records": [r.to_json() for r in records],
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.json}")
    return records


if __name__ == "__main__":
    main()
