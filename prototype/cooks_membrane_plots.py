"""Prototype-labelled plots/VTK and published OGS references for Cook's membrane."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from porepy.viz.exporter import Exporter

from prototype.cooks_membrane import (
    CooksMembraneResult, GEOMETRY_URL, POLYGON, PROJECT_URL, RESPONSE_COLUMNS, SOURCE_URL,
)
from utils.visualization import _publish_pvd_collections, _solution_data, _tensor_data

PREFIX = "prototype_cooks_membrane_analytical"
REFERENCE_FILE = Path(__file__).with_name("data") / "cooks_membrane_ogs_reference.json"


def _figure(width: float = 8., height: float = 6.) -> Figure:
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def export_results(results: list[CooksMembraneResult], folder: Path) -> dict[str, Path]:
    """Keep prototype source labels on every file and separate VTK/PVD directories."""
    if not results:
        raise ValueError("At least one Cook's membrane result is required.")
    results = sorted(results, key=lambda r: r.case.refinement)
    folder.mkdir(parents=True, exist_ok=True)
    reference = json.loads(REFERENCE_FILE.read_text())
    paths: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    outlines = np.vstack((POLYGON, POLYGON[0]))
    polygons, vertical, traces = [], [], []
    for result in results:
        case, state = result.case, result.state.tpsa
        g, ops = case.grid, case.operators
        u = state.x[ops.u_slice].reshape((2, g.num_cells), order="F")
        stress = np.stack([s.stress.to_numpy() for s in state.material_states], axis=2)
        trace = np.trace(stress, axis1=0, axis2=1)
        vertices = g.cell_centers[:2].T[:, None, :] + .5*case.spacing*np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])[None, :, :]
        polygons.append(vertices); vertical.append(u[1]); traces.append(trace)
        name = f"{PREFIX}_l{case.refinement}"
        data = _solution_data(g, state.x) + _tensor_data(g, "strain", state.epsilon) + _tensor_data(g, "stress", stress)
        data.append((g, "stress_trace", trace))
        Exporter(g, Path(name), folder_name=folder / "vtu").write_vtu(data)
        _publish_pvd_collections(folder, name)
        paths[f"vtk_l{case.refinement}"] = folder / "pvd" / f"{name}.pvd"
        response_path = folder / f"{name}_response.csv"
        np.savetxt(response_path, result.response, delimiter=",", header=",".join(RESPONSE_COLUMNS), comments="")
        paths[f"response_l{case.refinement}"] = response_path
        final = result.response[-1]
        rows.append({"solver": "prototype", "refinement": case.refinement, "spacing_mm": case.spacing,
                     "cells": g.num_cells, "dofs": ops.num_dofs, "area_mm2": float(g.cell_volumes.sum()),
                     "area_relative_error": float(g.cell_volumes.sum()/1440.-1.),
                     "top_right_uy_mm": float(final[2]), "mean_right_uy_mm": float(final[3]),
                     "relative_difference_from_ogs_bbar_n30": float(final[2]/reference["uy_bbar_mm"][-1]-1.),
                     "newton_corrections": int(result.response[:, 4].sum()),
                     "max_force_imbalance_N_per_mm": float(result.response[:, 9].max()),
                     "max_residual_block_norm": float(result.response[:, 5:9].max()),
                     "max_alpha": float(max(s.alpha for s in state.material_states))})

    def save(figure: Figure, suffix: str) -> None:
        path = folder / f"{PREFIX}_{suffix}.png"
        figure.savefig(path, dpi=180)
        figure.clear()
        paths[suffix] = path

    fig = _figure(10., 6.)
    ax = fig.subplots()
    ax.add_collection(PolyCollection(list(polygons[-1]), facecolors="#d8e8ee", edgecolors="#65818a", linewidths=.15, rasterized=True))
    ax.plot(outlines[:, 0], outlines[:, 1], "k--", lw=1.3, label="Exact Cook polygon")
    ax.plot([0., 0.], [0., 44.], color="#344750", lw=4, label="Clamp: u = 0")
    ax.quiver(np.full(5, 48.), np.linspace(45., 57., 5), np.zeros(5), np.full(5, 4.),
              angles="xy", scale_units="xy", scale=1., color="#007fa3", width=.004)
    ax.scatter([48.], [60.], color="#b02c2c", zorder=5)
    ax.annotate("Measure u_y at (48, 60)", (48., 60.), xytext=(12., 65.), arrowprops={"arrowstyle":"->"})
    ax.text(50., 48., "t_y = 6.25 MPa", rotation=90, va="center", fontsize=9)
    ax.set(xlim=(-4., 56.), ylim=(-4., 70.), aspect="equal", xlabel="x [mm]", ylabel="y [mm]")
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Prototype TPSA — Cook's membrane staircase mesh, h = {results[-1].case.spacing:g} mm\nE = 240.565 MPa, ν = 0.4999; plane strain, unit thickness")
    save(fig, "setup")

    chosen = [0, len(results)-1] if len(results) > 1 else [0]
    fig = _figure(10. if len(chosen) == 2 else 6., 10.)
    axes = fig.subplots(2, len(chosen), squeeze=False)
    for row, (values, label, cmap) in enumerate(((vertical, "Vertical displacement u_y [mm]", "viridis"),
                                                (traces, "Stress trace σ_xx + σ_yy + σ_zz [MPa]", "coolwarm"))):
        low, high = min(float(values[i].min()) for i in chosen), max(float(values[i].max()) for i in chosen)
        norm = (TwoSlopeNorm(vcenter=0., vmin=low, vmax=high) if row == 1 and low < 0. < high
                else Normalize(low, high if high > low else low+1.))
        for col, i in enumerate(chosen):
            ax = axes[row, col]
            collection = PolyCollection(list(polygons[i]), array=values[i], cmap=cmap, norm=norm,
                                        edgecolors="#666666", linewidths=.12, rasterized=True)
            ax.add_collection(collection)
            ax.plot(outlines[:, 0], outlines[:, 1], "k--", lw=.6)
            ax.set(xlim=(-1., 49.), ylim=(-1., 61.), aspect="equal", xlabel="x [mm]", ylabel="y [mm]",
                   title=f"h = {results[i].case.spacing:g} mm; {results[i].case.grid.num_cells} cells")
        fig.colorbar(collection, ax=list(axes[row]), shrink=.85, label=label)
    fig.suptitle("Prototype TPSA — elastic Cook's membrane\nStaircase geometry; dashed outline is the exact polygon")
    save(fig, "fields")

    fig = _figure(8., 5.5); ax = fig.subplots()
    n = np.array(reference["elements_per_side"])
    ax.semilogx(n*n, reference["uy_bbar_mm"], "o--", color="#444444", label="OGS B-bar (published)")
    ax.semilogx(n*n, reference["uy_non_bbar_mm"], "x--", color="#969696", label="OGS without B-bar (published)")
    ax.semilogx([r["cells"] for r in rows], [r["top_right_uy_mm"] for r in rows], "o-", color="#007fa3", label="Prototype TPSA (staircase)")
    for r in rows:
        ax.annotate(f"h={r['spacing_mm']:g}", (r["cells"], r["top_right_uy_mm"]), xytext=(-4, 7),
                    textcoords="offset points", ha="right", fontsize=8)
    ax.set(xlabel="Number of cells / elements (different geometry and DOF counts)",
           ylabel="Top-right vertical displacement [mm]", title="Cook's membrane — prototype mesh refinement")
    ax.grid(True, which="both", alpha=.25); ax.legend(fontsize=9)
    save(fig, "convergence")

    csv_path = folder / f"{PREFIX}_mesh_summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    paths["mesh_summary"] = csv_path
    metadata = {
        "solver_package":"prototype", "formulation":"retained-face TPSA", "jacobian":"analytical",
        "source_url":SOURCE_URL, "ogs_project_url":PROJECT_URL, "ogs_geometry_url":GEOMETRY_URL,
        "material":{"model":"linear elastic", "E_MPa":240.565, "nu":.4999, "inactive_yield_guard_MPa":1e12},
        "plane_strain":True, "thickness_mm":1., "polygon_mm":POLYGON.tolist(), "exact_area_mm2":1440.,
        "boundary_conditions":"left: full displacement clamp; right: (0,6.25) MPa; remaining faces: zero traction",
        "traction_MPa":[0.,6.25], "resultant_N_per_mm":100., "resultant_for_unit_thickness_N":100.,
        "geometry_approximation":"Cartesian cells selected by centre containment; sloped boundaries become staircases. No nonorthogonal TPSA law is used.",
        "corner_measurement":"Linear extrapolation from the two topmost retained right-face displacement values to (48,60) mm",
        "comparison_limit":"OGS values use conforming quadrilateral FEM meshes. Cell counts, geometry errors and corner evaluation differ; the finest OGS value is a numerical reference, not an exact solution.",
        "stress_trace_definition":"sigma_xx + sigma_yy + sigma_zz (3D plane-strain stress)",
        "reference":reference, "cases":rows, "files":{name:str(path) for name,path in paths.items()},
    }
    path = folder / f"{PREFIX}_summary.json"
    path.write_text(json.dumps(metadata, indent=2)+"\n")
    paths["summary"] = path
    return paths
