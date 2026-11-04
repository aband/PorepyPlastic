"""Reuse the benchmark plots with explicit alternatives/coupled output labels."""

from pathlib import Path

from alternatives.strip_footing import RESPONSE_COLUMNS, StripFootingResult
from prototype.strip_footing_plots import export_results as export_benchmark

PREFIX = "alternatives_strip_footing_coupled_analytical"


def export_results(result: StripFootingResult, folder: Path) -> dict[str, Path]:
    return export_benchmark(
        result, folder, prefix=PREFIX, label="Alternatives coupled TPSA",
        solver_package="alternatives", formulation="coupled cell/strain TPSA",
        response_columns=RESPONSE_COLUMNS,
    )
