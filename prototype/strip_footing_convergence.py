"""Nested mesh and load-increment convergence of the prototype strip footing.

Run from the repository root with python -B -m prototype.strip_footing_convergence.
The existing prototype solver, material, boundary conditions and tolerances are
used unchanged. Final cell fields are interpreted as piecewise constant when
computing volume-weighted L2 differences. These are self-differences, not errors
against an exact elastoplastic field.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from prototype.strip_footing import (
    FINAL_SETTLEMENT, HALF_WIDTH, LIMIT_FACTOR, YIELD_STRESS, make_case, run_case,
)
from prototype.strip_footing_plots import export_results

PREFIX = "prototype_strip_footing_analytical"


def curve_difference(coarse: np.ndarray, fine: np.ndarray) -> dict[str, float]:
    """Compare P/c over the common settlement interval, including cutback nodes.

    Integrate squared piecewise-linear curves exactly on the union of their
    knots. Normalization is by the finer curve's L2 norm over settlement.
    """
    if not np.allclose(coarse[[0, -1], 3], fine[[0, -1], 3], rtol=0, atol=1e-14):
        raise ValueError("Response curves must cover the same settlement interval.")
    x = np.union1d(coarse[:, 3], fine[:, 3])
    reference = np.interp(x, fine[:, 3], fine[:, 5])
    delta = np.interp(x, coarse[:, 3], coarse[:, 5]) - reference

    def squared_integral(values: np.ndarray) -> float:
        return float(np.sum(np.diff(x) * (
            values[:-1]**2 + values[:-1]*values[1:] + values[1:]**2
        ) / 3))

    return {
        "relative_curve_L2": float(np.sqrt(squared_integral(delta) / squared_integral(reference))),
        "max_pressure_over_c_difference": float(np.max(np.abs(delta))),
        "final_pressure_over_c_difference": float(abs(delta[-1])),
    }


def field_differences(coarse: dict, fine: dict) -> dict[str, float]:
    """Prolong nested coarse cell constants to fine cells before taking L2 norms."""
    nx, ny = len(coarse["x_nodes"]) - 1, len(coarse["y_nodes"]) - 1
    fx, fy = len(fine["x_nodes"]) - 1, len(fine["y_nodes"]) - 1
    ratio = fx // nx
    if ratio < 1 or fx != ratio*nx or fy != ratio*ny:
        raise ValueError("Expected nested tensor meshes with an integer refinement ratio.")
    for axis in ("x_nodes", "y_nodes"):
        if not np.allclose(coarse[axis], fine[axis][::ratio], rtol=0, atol=1e-13):
            raise ValueError("Mesh nodes are not nested.")
    ix, iy = np.meshgrid(np.arange(fx), np.arange(fy))
    parent = ((iy // ratio)*nx + ix // ratio).ravel()
    volumes = fine["cell_volumes"]
    errors = {}
    for name in ("displacement", "stress", "strain", "alpha"):
        reference = fine[name].reshape((-1, fx*fy))
        prolonged = coarse[name].reshape((-1, nx*ny))[:, parent]
        absolute = float(np.sqrt(np.sum((prolonged-reference)**2 * volumes)))
        norm = float(np.sqrt(np.sum(reference**2 * volumes)))
        errors[f"{name}_L2"] = absolute
        errors[f"relative_{name}_L2"] = absolute / norm if norm else 0.0 if absolute == 0 else None
    return errors


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("prototype", "material", "utils")
            for p in sorted((root/folder).glob("*.py"))}


def run_one(folder: Path, level: int, steps: int, hashes: dict, *, resume: bool) -> tuple[dict, dict]:
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / f"{PREFIX}_metrics.json"
    fields_path = folder / f"{PREFIX}_fields.npz"
    response_path = folder / f"{PREFIX}_response.csv"
    if resume and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["source_hashes"] != hashes or metadata["level"] != level or metadata["requested_steps"] != steps:
            raise ValueError(f"Cannot reuse a run with different source or settings: {folder}")
        with np.load(fields_path) as saved:
            fields = {name: saved[name].copy() for name in saved.files}
        fields["response"] = np.loadtxt(response_path, delimiter=",", skiprows=1)
        print(f"Reusing completed L{level}, {steps} targets", flush=True)
        return metadata, fields

    start = time.perf_counter()
    case = make_case(refinement_level=level)
    print(f"Starting L{level}: {case.grid.num_cells} cells, {case.operators.num_dofs} DOFs, {steps} targets", flush=True)
    result = run_case(case, steps=steps, adaptive=True, progress=True)
    solve_seconds = time.perf_counter() - start
    state = result.state.tpsa
    stress = np.stack([p.stress.to_numpy() for p in state.material_states], axis=2)
    plastic_strain = np.stack([p.plastic_strain.to_numpy() for p in state.material_states], axis=2)
    deviator = stress - np.einsum("iik->k", stress)[None, None, :] * np.eye(3)[:, :, None] / 3
    max_vm = float(np.sqrt(1.5*np.sum(deviator**2, axis=(0, 1))).max())
    trace_error = float(np.abs(np.einsum("iik->k", plastic_strain)).max())
    mask = case.operators.dirichlet_mask
    boundary_error = float(np.abs(state.face_displacement[mask]-case.bc_values[mask]).max())
    residual_ratio = max(float(np.max(row[8:12]/step.newton.thresholds))
                         for row, step in zip(result.response[1:], result.steps, strict=True))
    if max_vm > YIELD_STRESS*(1+1e-8) or trace_error > 1e-10 or boundary_error > 1e-13 or residual_ratio > 1+1e-7:
        raise RuntimeError("A converged run failed material, boundary or residual validation.")
    fields = {
        "x_nodes": case.x_nodes, "y_nodes": case.y_nodes,
        "cell_volumes": case.grid.cell_volumes,
        "displacement": state.x[:2*case.grid.num_cells].reshape((2, -1), order="F"),
        "stress": stress, "strain": state.epsilon,
        "alpha": np.array([p.alpha for p in state.material_states]),
    }
    if not all(np.all(np.isfinite(a)) for a in fields.values()):
        raise RuntimeError("Nonfinite final field.")
    relative_balance = result.response[1:, 12] / np.maximum(1.0, np.abs(result.response[1:, 6]))
    if relative_balance.max() > 1e-7:
        raise RuntimeError("Boundary force imbalance exceeds 1e-7 of half-footing reaction.")
    metadata = {
        "level": level, "cells": case.grid.num_cells, "dofs": case.operators.num_dofs,
        "cells_under_half_footing": case.cells_under_footing,
        "h_under_footing_m": HALF_WIDTH / case.cells_under_footing,
        "h_max_axis_m": float(np.diff(case.x_nodes).max()),
        "requested_steps": steps, "accepted_steps": len(result.steps),
        "rejected_attempts": sum(not a.accepted for a in result.attempts),
        "total_newton_iterations": int(result.response[:, 7].sum()),
        "setup_and_solve_seconds": solve_seconds,
        "final_pressure_over_c": float(result.response[-1, 5]),
        "relative_difference_from_slip_line_limit": float(result.response[-1, 5]/LIMIT_FACTOR-1),
        "max_force_imbalance_kN_per_m": float(result.response[:, 12].max()),
        "max_relative_force_imbalance": float(relative_balance.max()),
        "max_total_residual_over_threshold": residual_ratio,
        "max_von_mises_kPa": max_vm, "max_plastic_trace": trace_error,
        "max_dirichlet_error_m": boundary_error,
        "max_alpha": float(fields["alpha"].max()),
        "plastic_area_m2": float(case.grid.cell_volumes[fields["alpha"] > 1e-12].sum()),
        "source_hashes": hashes,
    }
    export_results(result, folder)
    np.savez_compressed(fields_path, **fields)
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    fields["response"] = result.response.copy()
    print(f"Completed L{level}/{steps}: P/c={metadata['final_pressure_over_c']:.8f}, {solve_seconds:.1f} s", flush=True)
    return metadata, fields


def plot_study(output: Path, spatial: list[tuple[dict, dict]], temporal: list[tuple[dict, dict]], differences: list[dict]) -> None:
    def save(fig: Figure, name: str) -> None:
        FigureCanvasAgg(fig)
        fig.savefig(output/f"{PREFIX}_{name}.png", dpi=180)
        fig.clear()

    fig = Figure(figsize=(12, 4.5), layout="constrained")
    for ax, runs, title in zip(fig.subplots(1, 2), (spatial, temporal),
                               ("Mesh refinement (28 target increments)", "Load-increment refinement (finest mesh)")):
        for m, fields in runs:
            response = fields["response"]
            ax.plot(response[:, 3], response[:, 5], label=f"L{m['level']}: {m['cells']} cells, {m['requested_steps']} targets")
        ax.axhline(LIMIT_FACTOR, color="black", ls="--", label=r"Slip-line limit $2+\pi$")
        ax.set(xlabel=r"Settlement $u/B$", ylabel=r"Average footing pressure $P/c$", title=title)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Prototype strip footing — mesh and load-increment study")
    save(fig, "convergence_response")

    fig = Figure(figsize=(12, 4.5), layout="constrained")
    ax, bx = fig.subplots(1, 2)
    ax.plot([m["h_under_footing_m"] for m, _ in spatial], [m["final_pressure_over_c"] for m, _ in spatial], "o-")
    ax.axhline(LIMIT_FACTOR, color="black", ls="--", label=r"$2+\pi$")
    ax.set(xscale="log", xlabel="Cell width under footing [m]", ylabel=r"Final $P/c$", title="Endpoint response; fixed target increment")
    ax.invert_xaxis()
    ax.grid(alpha=.25)
    ax.legend()
    for name in ("displacement", "stress", "strain", "alpha"):
        bx.loglog([row["h_coarse_m"] for row in differences], [row[f"relative_{name}_L2"] for row in differences], "o-", label=name)
    bx.set(xlabel="Coarser cell width under footing [m]", ylabel="Relative volume-weighted L2 difference", title="Adjacent meshes; piecewise-constant fields")
    bx.invert_xaxis()
    bx.grid(alpha=.25)
    bx.legend()
    save(fig, "convergence_metrics")


def run(output: Path, *, resume: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    # Three spatial levels at fixed load resolution; three load resolutions on L2.
    schedule = [(0, 28), (1, 28), (2, 14), (2, 28), (2, 56)]
    status_path = output/f"{PREFIX}_study_status.json"
    status = {"status": "running", "schedule": schedule, "completed": []}
    runs = {}
    try:
        for level, steps in schedule:
            runs[level, steps] = run_one(output/f"level{level}_steps{steps}", level, steps, hashes, resume=resume)
            status["completed"].append([level, steps])
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            gc.collect()
    except Exception as error:
        status.update(status="failed", error=str(error))
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        raise

    spatial = [runs[level, 28] for level in (0, 1, 2)]
    temporal = [runs[2, steps] for steps in (14, 28, 56)]
    rows = [{k: v for k, v in metadata.items() if k != "source_hashes"} for metadata, _ in runs.values()]
    write_csv(output/f"{PREFIX}_runs.csv", rows)
    mesh_rows, load_rows = [], []
    for target, pairs in ((mesh_rows, spatial), (load_rows, temporal)):
        for (a, coarse), (b, fine) in zip(pairs[:-1], pairs[1:]):
            target.append({"coarse_level": a["level"], "fine_level": b["level"],
                           "coarse_steps": a["requested_steps"], "fine_steps": b["requested_steps"],
                           "h_coarse_m": a["h_under_footing_m"],
                           **curve_difference(coarse["response"], fine["response"]),
                           **field_differences(coarse, fine)})
    write_csv(output/f"{PREFIX}_mesh_differences.csv", mesh_rows)
    write_csv(output/f"{PREFIX}_load_differences.csv", load_rows)
    rates = {name: float(np.log2(mesh_rows[0][name]/mesh_rows[1][name]))
             for name in ("final_pressure_over_c_difference", "displacement_L2", "stress_L2", "strain_L2", "alpha_L2")
             if mesh_rows[0][name] > 0 and mesh_rows[1][name] > 0}
    plot_study(output, spatial, temporal, mesh_rows)
    summary = {
        "solver_package": "prototype", "jacobian": "analytical", "python": platform.python_version(),
        "study": "Full-domain nested h-refinement and independent load-increment refinement",
        "mesh_levels_cells": [m["cells"] for m, _ in spatial], "spatial_target_steps": 28,
        "load_refinement_steps_on_finest_mesh": [14, 28, 56],
        "final_settlement_m": FINAL_SETTLEMENT, "slip_line_limit_P_over_c": LIMIT_FACTOR,
        "finest_run": rows[-1], "adjacent_mesh_comparisons": mesh_rows,
        "adjacent_load_comparisons": load_rows, "observed_difference_reduction_orders": rates,
        "interpretation": [
            "Three levels show refinement trends, not proof of an asymptotic rate.",
            "Cell-field L2 differences use volume-weighted coarse-to-fine piecewise-constant prolongation; tensors use all 3x3 components.",
            "Curve L2 differences integrate over settlement, with exact integration of squared piecewise-linear interpolants.",
            "2+pi is an infinite-medium ultimate-capacity reference, not an exact finite-domain response at settlement 0.002 m.",
            "Domain size and final settlement are fixed; domain truncation and approach to the limit load are not separately converged.",
            "Adaptive cutbacks are enabled and reported; requested and accepted step counts can differ.",
        ],
        "source_hashes": hashes,
    }
    (output/f"{PREFIX}_convergence_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    status["status"] = "complete"
    status_path.write_text(json.dumps(status, indent=2)+"\n")
    print(json.dumps({"output": str(output), "runs": rows, "orders": rates}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/prototype_strip_footing_convergence"))
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs only when source hashes match")
    args = parser.parse_args()
    run(args.output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
