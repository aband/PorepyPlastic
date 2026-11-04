"""Linear-solver benchmark: option parsing, records, reference comparison."""

import json
from pathlib import Path

import numpy as np
import pytest

from examples.benchmark_linear_solvers import (
    Case,
    benchmark_case,
    build_problem,
    controller_options,
    krylov_options,
    main,
    parse_case,
    parse_pairs,
    parse_solver,
    parse_value,
    relative_difference,
    run_once,
)
from core.linear_solvers import CondensedDirectSolver, DirectSolver
from examples.benchmark_linear_solvers import make_solver

TINY = Case("2d:4", 2, 4)
PATH = (0.5, 1.0, 0.6)


# ------------------------------------------------------------------- parsing
@pytest.mark.parametrize(("text", "expected"), [
    ("2d:96", Case("2d:96", 2, 96)),
    (" 3D:12 ", Case("3d:12", 3, 12)),
])
def test_parse_case(text: str, expected: Case) -> None:
    assert parse_case(text) == expected


@pytest.mark.parametrize("text", ["96", "4d:8", "2d:", "2d:x", "2d:0", "2d:-3"])
def test_parse_case_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_case(text)


@pytest.mark.parametrize(("text", "expected"), [
    ("true", True), ("False", False), ("3", 3), ("1e-3", 1e-3), ("0.5", 0.5), ("gamg", "gamg"),
])
def test_parse_value(text: str, expected: object) -> None:
    value = parse_value(text)
    assert value == expected and type(value) is type(expected)


def test_parse_pairs() -> None:
    assert parse_pairs("a=1, b = x,") == {"a": "1", "b": "x"}
    assert parse_pairs("") == {}
    for bad in ("a", "=1", "a=1,a=2"):
        with pytest.raises(ValueError):
            parse_pairs(bad)


def test_parse_solver() -> None:
    spec = parse_solver("krylov:preconditioner=ilu,rtol=1e-3")
    assert (spec.kind, spec.options) == ("krylov", (("preconditioner", "ilu"), ("rtol", "1e-3")))
    assert parse_solver("Direct").kind == "direct"
    with pytest.raises(ValueError, match="kind"):
        parse_solver("lu")
    with pytest.raises(ValueError, match="no options"):
        parse_solver("direct:rtol=1e-3")


def test_krylov_options_from_spec() -> None:
    options = krylov_options(parse_solver(
        "krylov:preconditioner=ilu,ilu_levels=1,rtol=1e-3,right_preconditioning=false,"
        "petsc.pc_factor_levels=2"
    ))
    assert (options.preconditioner, options.ilu_levels, options.rtol) == ("ilu", 1, 1e-3)
    assert options.right_preconditioning is False
    assert dict(options.petsc_options) == {"pc_factor_levels": "2"}
    with pytest.raises(ValueError, match="Unknown KrylovOptions field"):
        krylov_options(parse_solver("krylov:foo=1"))
    with pytest.raises(ValueError, match="rtol"):
        krylov_options(parse_solver("krylov:rtol=2"))


def test_make_solver_kinds() -> None:
    assert make_solver(parse_solver("default")) is None
    assert isinstance(make_solver(parse_solver("direct")), DirectSolver)
    assert isinstance(make_solver(parse_solver("condensed-direct")), CondensedDirectSolver)


def test_controller_options() -> None:
    assert controller_options(["max_iterations=7", "initial_load_factor=0.0"]) == {
        "max_iterations": 7, "initial_load_factor": 0.0,
    }
    with pytest.raises(ValueError, match="set by the benchmark"):
        controller_options(["linear_solver=x"])
    with pytest.raises(ValueError, match="Unknown LoadStepController option"):
        controller_options(["nonexistent=1"])
    with pytest.raises(ValueError, match="twice"):
        controller_options(["max_iterations=7", "max_iterations=8"])


# ----------------------------------------------------------------------- runs
@pytest.mark.parametrize("dim", [2, 3])
def test_build_problem_is_plastic_on_the_path(dim: int) -> None:
    assembler, reference = build_problem(Case(f"{dim}d:3", dim, 3))
    assert reference.shape == (dim, assembler.layout.dirichlet_faces.size)
    record = run_once(Case(f"{dim}d:3", dim, 3), assembler, reference,
                      parse_solver("direct"), PATH, {})
    assert record.status == "ok"
    assert record.final_state is not None
    assert np.any(record.final_state.material.alpha > 0.0), "benchmark path must reach plasticity"


def test_condensed_matches_reference_and_reference_is_inserted() -> None:
    records = benchmark_case(
        TINY, [parse_solver("condensed-direct")], parse_solver("direct"),
        path=PATH, controller_kwargs={},
    )
    assert [r.solver for r in records] == ["direct", "condensed-direct"]
    reference, condensed = records
    assert reference.difference == {"x": 0.0, "face_displacement": 0.0, "plastic_strain": 0.0}
    assert condensed.difference is not None and max(condensed.difference.values()) < 1e-10
    assert condensed.newton_iterations == reference.newton_iterations
    assert len(reference.newton_iterations) == len(PATH)
    assert all(r.final_state is None for r in records)   # released after comparison
    assert reference.krylov_iterations == 0 and reference.pc_setup_time is None


def test_krylov_records_solver_statistics() -> None:
    pytest.importorskip("petsc4py")
    records = benchmark_case(
        TINY, [parse_solver("direct"), parse_solver("krylov:preconditioner=ilu,rtol=1e-8")],
        parse_solver("direct"), path=PATH, controller_kwargs={},
    )
    krylov = records[1]
    assert krylov.status == "ok"
    assert krylov.krylov_iterations > 0 and krylov.max_krylov_iterations > 0
    assert krylov.pc_setup_time is not None and krylov.krylov_solve_time is not None
    assert krylov.fallbacks == 0
    assert krylov.difference is not None and max(krylov.difference.values()) < 1e-5


def test_failure_becomes_record_without_difference() -> None:
    records = benchmark_case(
        TINY, [parse_solver("direct")], None,
        path=PATH, controller_kwargs={"max_iterations": 1},
    )
    (record,) = records
    assert record.status == "failed"
    assert record.error is not None and "Load step 1" in record.error
    assert record.difference is None and record.newton_iterations == []


def test_relative_difference_of_identical_states_is_zero() -> None:
    assembler, reference = build_problem(TINY)
    record = run_once(TINY, assembler, reference, parse_solver("direct"), PATH, {})
    assert record.final_state is not None
    assert set(relative_difference(record.final_state, record.final_state).values()) == {0.0}


def test_main_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "nested" / "bench.json"
    records = main([
        "--case", "2d:4", "--solver", "direct", "--solver", "condensed-direct",
        "--path", "0.5,1.0", "--repeat", "2", "--json", str(output),
    ])
    assert [r.solver for r in records] == ["direct", "condensed-direct"]
    payload = json.loads(output.read_text())
    assert payload["path"] == [0.5, 1.0]
    assert [r["solver"] for r in payload["records"]] == ["direct", "condensed-direct"]
    assert "final_state" not in payload["records"][0]
    printed = capsys.readouterr().out
    assert "condensed-direct" in printed and "Wrote" in printed


def test_main_reference_none_skips_comparison() -> None:
    (record,) = main(["--case", "2d:4", "--solver", "direct", "--path", "1.0", "--reference", "none"])
    assert record.status == "ok" and record.difference is None


@pytest.mark.parametrize("argv", [
    ["--case", "5d:3"],
    ["--solver", "krylov:foo=1"],
    ["--controller", "linear_solver=x"],
    ["--repeat", "0"],
    ["--path", "0.5,nan"],
])
def test_main_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        main(argv)
