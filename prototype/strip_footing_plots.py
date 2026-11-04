"""Shared strip-footing plots, defaulting to prototype labels and metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from porepy.viz.exporter import Exporter

from prototype.strip_footing import (
    COHESION, DOMAIN_SIZE, HALF_WIDTH, LIMIT_FACTOR, POISSON_RATIO, RESPONSE_COLUMNS,
    StripFootingResult, WIDTH, YIELD_STRESS, YOUNG_MODULUS,
)
from utils.visualization import _publish_pvd_collections, _solution_data, _tensor_data
from utils.state import as_tpsa_state

if TYPE_CHECKING:
    from alternatives.strip_footing import StripFootingResult as CoupledStripFootingResult

PREFIX = "prototype_strip_footing_analytical"


def export_results(
    result: StripFootingResult | CoupledStripFootingResult, folder: Path, *,
    prefix: str = PREFIX, label: str = "Prototype TPSA", solver_package: str = "prototype",
    formulation: str = "retained-face TPSA", response_columns: tuple[str, ...] = RESPONSE_COLUMNS,
) -> dict[str, Path]:
    """Export numerical results, with the analytical limit as a reference only."""
    folder.mkdir(parents=True, exist_ok=True)
    c, state = result.case, as_tpsa_state(result.state)
    g, ops = c.grid, c.operators
    paths: dict[str, Path] = {}
    # PorePy TensorGrid cell numbering has x varying fastest.
    polygons = [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                for y0, y1 in zip(c.y_nodes[:-1], c.y_nodes[1:])
                for x0, x1 in zip(c.x_nodes[:-1], c.x_nodes[1:])]

    def figure(width: float = 8.0, height: float = 6.0) -> Figure:
        fig = Figure(figsize=(width, height), layout="constrained")
        FigureCanvasAgg(fig)
        return fig

    def save(fig: Figure, name: str) -> None:
        path = folder / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=180)
        fig.clear()
        paths[name] = path

    fig = figure()
    ax = fig.subplots()
    ax.add_collection(PolyCollection(polygons, facecolors="#e3edf0", edgecolors="#81939b", linewidths=0.3))
    ax.plot([0, 0], [-5, 0], color="#286a96", lw=3, label=r"Sides: $u_x=0,\ t_y=0$")
    ax.plot([5, 5], [-5, 0], color="#286a96", lw=3)
    ax.plot([0, 5], [-5, -5], color="#39754d", lw=3, label=r"Bottom: $u_y=0,\ t_x=0$")
    ax.plot([0, HALF_WIDTH], [0, 0], color="#a83835", lw=5, label=r"Footing: $u_y=-u,\ t_x=0$")
    ax.plot([HALF_WIDTH, 5], [0, 0], color="#666666", lw=2, label="Exposed surface: traction free")
    for x in np.linspace(0.06, 0.44, 4):
        ax.annotate("", (x, 0), (x, 0.35), arrowprops={"arrowstyle": "->", "color": "#a83835"})
    ax.set(xlim=(-0.3, 5.3), ylim=(-5.3, 0.6), aspect="equal", xlabel="x [m]", ylabel="y [m]")
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{label} — smooth strip footing, half-model\n"
                 f"E = 10⁷ kPa, ν = 0.48, σᵧ = 848.7 kPa; {g.num_cells} graded rectangular cells")
    save(fig, "setup")

    fig = figure(8, 5)
    ax = fig.subplots()
    ax.plot(result.response[:, 3], result.response[:, 5], "o-", ms=3, label=label)
    ax.axhline(LIMIT_FACTOR, ls="--", color="#555555", label=r"Slip-line limit: $2+\pi$")
    ax.set(xlabel=r"Normalized settlement $u/B$", ylabel=r"Average pressure $P/c$", xlim=(0, c.settlement / WIDTH))
    ax.grid(alpha=0.25); ax.legend()
    difference = float(result.response[-1, 5] / LIMIT_FACTOR - 1)
    fig.suptitle(f"{label} strip footing — displacement-controlled loading\n"
                 f"Final P/c = {result.response[-1, 5]:.4f}; difference from limit = {difference:+.2%}")
    save(fig, "response")

    displacement = state.x[:2 * g.num_cells].reshape((2, g.num_cells), order="F")
    stress = np.stack([s.stress.to_numpy() for s in state.material_states], axis=2)
    alpha = np.array([s.alpha for s in state.material_states])
    fig = figure(13, 4.5)
    axes = fig.subplots(1, 3)
    for ax, values, label in zip(axes, [displacement[1], stress[1, 1], alpha],
                               ["Vertical displacement [m]", "Vertical stress [kPa]", "Accumulated plastic strain"]):
        collection = PolyCollection(polygons, array=values, cmap="viridis", edgecolors="none")
        ax.add_collection(collection)
        ax.plot([0, HALF_WIDTH], [0, 0], "k-", lw=4)
        ax.set(xlim=(0, DOMAIN_SIZE), ylim=(-DOMAIN_SIZE, 0), aspect="equal", xlabel="x [m]", ylabel="y [m]")
        fig.colorbar(collection, ax=ax, shrink=0.8, label=label)
    fig.suptitle(f"{label} strip footing — final fields at u = {c.settlement:g} m")
    save(fig, "fields")

    fig = figure(10, 8)
    axes = fig.subplots(2, 2).ravel()
    for ax, target in zip(axes, [0.00015, 0.00025, 0.00065, 0.0011]):
        index = int(np.argmin([abs(s.load_factor*c.settlement-target) for s in result.steps]))
        record = result.steps[index]
        yielded = np.array([s.alpha > 1e-12 for s in as_tpsa_state(record.state).material_states])
        ax.add_collection(PolyCollection(polygons, facecolors=np.where(yielded[:, None],
                          np.array([[0.25, 0.42, 0.56, 1.0]]), np.array([[0.95, 0.95, 0.95, 1.0]])),
                          edgecolors="#aaaaaa", linewidths=0.15))
        ax.plot([0, HALF_WIDTH], [0, 0], "k-", lw=4)
        ax.set(xlim=(0, 2.5), ylim=(-2.5, 0), aspect="equal", xlabel="x [m]", ylabel="y [m]",
               title=f"u/B = {record.load_factor*c.settlement/WIDTH:.6g}")
    fig.suptitle(f"{label} strip footing — accumulated plastic zones (α > 10⁻¹²)\n"
                 "Nearest computed settlements to the four stages in Figure 7.27")
    save(fig, "plastic_zones")

    path = folder / f"{prefix}_response.csv"
    np.savetxt(path, result.response, delimiter=",", header=",".join(response_columns), comments="")
    paths["csv"] = path
    data = _solution_data(g, state.x) + _tensor_data(g, "strain", state.epsilon) + _tensor_data(g, "stress", stress)
    data += [(g, "alpha", alpha), (g, "yielded", (alpha > 1e-12).astype(float))]
    Exporter(g, Path(prefix), folder_name=folder / "vtu").write_vtu(data)
    _publish_pvd_collections(folder, prefix)
    paths["vtk"] = folder / "pvd" / f"{prefix}.pvd"
    metadata = {
        "solver_package": solver_package, "formulation": formulation, "jacobian": "analytical",
        "reference": {"book": "Computational Methods for Plasticity (de Souza Neto, Peric and Owen, 2008)",
                      "section": "7.5.4", "printed_pages": [252, 253, 254, 255],
                      "pdf_pages": [277, 278, 279, 280], "figures": ["7.24", "7.25", "7.26", "7.27"],
                      "limit_factor_P_over_c": float(LIMIT_FACTOR)},
        "units": {"length": "m", "stress": "kPa", "integrated_force": "kN/m of out-of-plane length"},
        "plane_strain": True, "footing_width_m": WIDTH, "footing_length_m": 5.0,
        "half_domain_m": [5.0, 5.0], "weightless_soil": True,
        "material": {"model": "von Mises J2", "E_kPa": YOUNG_MODULUS, "nu": POISSON_RATIO,
                     "yield_stress_kPa": YIELD_STRESS, "cohesion_kPa": float(COHESION), "hardening": "none"},
        "boundary_conditions": {"x=0 and x=5": "ux=0, ty=0", "y=-5": "uy=0, tx=0",
                                "y=0, 0<x<0.5": "uy=-settlement, tx=0 (smooth rigid footing)",
                                "y=0, 0.5<x<5": "tx=ty=0"},
        "mesh": {"type": "orthogonal graded TensorGrid", "cells": g.num_cells, "dofs": ops.num_dofs,
                 "cells_under_half_footing": c.cells_under_footing, "growth": c.growth,
                 "refinement_level": getattr(c, "refinement_level", 0),
                 "x_nodes": c.x_nodes.tolist(), "y_nodes": c.y_nodes.tolist()},
        "loading": {"final_settlement_m": c.settlement, "requested_uniform_increments": result.requested_steps,
                    "accepted_steps": len(result.steps), "adaptive": result.adaptive,
                    "rejected_attempts": sum(not a.accepted for a in result.attempts),
                    "accepted_settlements_m": result.response[1:, 2].tolist(),
                    "schedule_note": "Book specifies 14 increments and u=0.002 m, not the complete sequence; uniform targets are used here."},
        "results": {"final_pressure_kPa": float(result.response[-1, 4]),
                    "final_pressure_over_c": float(result.response[-1, 5]),
                    "relative_difference_from_limit": difference,
                    "max_force_imbalance_kN_per_m": float(result.response[:, 12].max()),
                    "max_alpha": float(alpha.max())},
        "comparison_notes": "Same material and component boundary conditions as the book. Graded rectangular TPSA cells replace quadratic FEM elements. The analytical line is an infinite-medium limit, not a verified accuracy result for this mesh. Pressure = 2*half_reaction/B. No contact separation or finite-strain geometry update.",
        "files": {k: str(p) for k, p in paths.items()},
    }
    path = folder / f"{prefix}_summary.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    paths["summary"] = path
    return paths
