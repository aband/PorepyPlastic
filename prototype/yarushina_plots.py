"""Plots and numerical error reports for the prototype circular-hole benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from porepy.viz.exporter import Exporter

from prototype.yarushina import CircularHoleResult, RESPONSE_COLUMNS, SOURCE, load_path, plastic_axes, stress_measures
from utils.visualization import _publish_pvd_collections, _solution_data, _tensor_data

PREFIX = "prototype_yarushina_analytical"


def _figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _polygons(result: CircularHoleResult) -> np.ndarray:
    return (result.case.grid.cell_centers[:2].T[:, None, :]+.5*result.case.spacing*
            np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])[None, :, :])


def export_results(results: list[CircularHoleResult], folder: Path) -> dict[str, Path]:
    """Write PNG, CSV, JSON, VTU and PVD, all with an explicit prototype prefix."""
    if not results:
        raise ValueError("At least one result is required.")
    results = sorted(results, key=lambda r: r.case.level)
    if any(r.pressure != results[0].pressure or r.case.nu != results[0].case.nu or r.case.half_width != results[0].case.half_width for r in results):
        raise ValueError("Compare meshes at identical final pressure, nu and outer domain size.")
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    summaries: list[dict[str, Any]] = []
    for result in results:
        c, s = result.case, result.state
        g, ops = c.grid, c.operators
        exact_mean, exact_shear = stress_measures(result.reference_stress)
        mean, shear = stress_measures(result.stress)
        mean_error = 100.*abs(mean-exact_mean)/np.max(abs(exact_mean))
        shear_error = 100.*abs(shear-exact_shear)/np.max(abs(exact_shear))
        alpha = np.array([p.alpha for p in s.tpsa.material_states])
        axis_x, axis_y = plastic_axes(result.pressure)
        exact_plastic = (g.cell_centers[0]/axis_x)**2+(g.cell_centers[1]/axis_y)**2 <= 1. if axis_x else np.zeros(g.num_cells, dtype=bool)
        reference_area = max(0., np.pi*(axis_x*axis_y-1.)) if axis_x else 0.
        mask_mismatch_area = float(np.sum(g.cell_volumes[(alpha > 1e-8) != exact_plastic]))
        name = f"{PREFIX}_l{c.level}"
        data = _solution_data(g, s.tpsa.x)+_tensor_data(g, "strain", s.tpsa.epsilon)+_tensor_data(g, "stress", result.stress)
        for field, values in (("mean_stress", mean), ("maximum_shear", shear),
                              ("analytical_mean_stress", exact_mean), ("analytical_maximum_shear", exact_shear),
                              ("mean_stress_error_percent", mean_error), ("maximum_shear_error_percent", shear_error),
                              ("accumulated_plastic_strain", alpha), ("analytical_plastic_zone", exact_plastic.astype(float))):
            data.append((g, field, values))
        Exporter(g, Path(name), folder_name=folder/"vtu").write_vtu(data)
        _publish_pvd_collections(folder, name)
        paths[f"vtk_l{c.level}"] = folder/"pvd"/f"{name}.pvd"
        response = folder/f"{name}_response.csv"
        np.savetxt(response, result.response, delimiter=",", header=",".join(RESPONSE_COLUMNS), comments="")
        paths[f"response_l{c.level}"] = response
        fields = folder/f"{name}_fields.csv"
        np.savetxt(fields, np.column_stack((g.cell_centers[:2].T, mean, exact_mean, shear, exact_shear, mean_error, shear_error, alpha)),
                   delimiter=",", header="x_over_R,y_over_R,mean_stress_over_k,analytical_mean_stress_over_k,max_shear_over_k,analytical_max_shear_over_k,mean_error_percent,shear_error_percent,alpha", comments="")
        paths[f"fields_l{c.level}"] = fields
        dev = result.stress-np.trace(result.stress, axis1=0, axis2=1)[None, None, :]*np.eye(3)[:, :, None]/3.
        mises = np.sqrt(1.5*np.einsum("ijn,ijn->n", dev, dev))
        sides = np.isin(ops.side_faces, ops.boundary_faces)
        summary = {
            "solver": "prototype", "level": c.level, "spacing_over_R": c.spacing,
            "cells": g.num_cells, "dofs": ops.num_dofs,
            "hole_area": float(4.*c.half_width**2-g.cell_volumes.sum()),
            "hole_area_relative_error": float((4.*c.half_width**2-g.cell_volumes.sum())/np.pi-1.),
            "mean_stress_relative_L2": float(result.response[-1, 8]),
            "maximum_shear_relative_L2": float(result.response[-1, 9]),
            "mean_stress_max_error_percent_eq41": float(mean_error.max()),
            "maximum_shear_max_error_percent_eq41": float(shear_error.max()),
            "numerical_plastic_area": float(result.response[-1, 10]),
            "analytical_plastic_area": float(reference_area),
            "plastic_mask_disagreement_area": mask_mismatch_area,
            "max_alpha": float(alpha.max()),
            "max_yield_excess_over_k": float(max(0., mises.max()-np.sqrt(3.))),
            "max_accumulated_residual_norm": float(result.response[:, 4:8].max()),
            "boundary_force_imbalance": float(np.linalg.norm(s.side_traction[:, sides].sum(axis=1))),
            "load_steps": len(result.response)-1,
            "total_newton_corrections": int(result.response[:, 3].sum()),
            "elapsed_seconds": float(result.response[-1, 11]),
        }
        summaries.append(summary)

    def save(fig: Figure, suffix: str) -> None:
        path = folder/f"{PREFIX}_{suffix}.png"
        fig.savefig(path, dpi=180)
        fig.clear()
        paths[suffix] = path

    finest = results[-1]
    c, g = finest.case, finest.case.grid
    theta = np.linspace(0., 2.*np.pi, 361)
    ax_len, ay_len = plastic_axes(finest.pressure)
    verts = _polygons(finest)

    def outline(ax: Any) -> None:
        ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1., label="Exact circular hole")
        if ax_len:
            ax.plot(ax_len*np.cos(theta), ay_len*np.sin(theta), color="#de7b00", lw=1.4, label="Analytical plastic front")
        ax.set(xlim=(-c.half_width, c.half_width), ylim=(-c.half_width, c.half_width),
               aspect="equal", xlabel="x / R", ylabel="y / R")

    fig = _figure(9., 7.); ax = fig.subplots()
    ax.add_collection(PolyCollection(list(verts), facecolors="#dbe9ed", edgecolors="#5b737b", linewidths=.25))
    outline(ax)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Prototype — circular-hole staircase, h/R={c.spacing:g}\nOuter square: analytical displacement; hole: zero traction")
    save(fig, "setup")

    mean, shear = stress_measures(finest.stress)
    exact_mean, exact_shear = stress_measures(finest.reference_stress)
    fig = _figure(13., 8.); axes = fig.subplots(2, 3)
    for row, (numerical, exact, label) in enumerate(((mean, exact_mean, "Mean stress / k"), (shear, exact_shear, "Maximum shear / k"))):
        lower, upper = min(numerical.min(), exact.min()), max(numerical.max(), exact.max())
        for col, (values, title) in enumerate(((numerical, "Prototype"), (exact, "Analytical"), (100.*abs(numerical-exact)/np.max(abs(exact)), "Error, Eq. (41) [%]"))):
            ax = axes[row, col]
            collection = PolyCollection(list(verts), array=values, cmap="magma" if col == 2 else "viridis", rasterized=True)
            if col != 2:
                collection.set_clim(lower, upper)
            ax.add_collection(collection); outline(ax); ax.set_title(title)
            fig.colorbar(collection, ax=ax, shrink=.8, label=label if col != 2 else "Normalized absolute error [%]")
    fig.suptitle(f"Prototype — Galin/Yarushina stress comparison, P/k={finest.pressure:g}, τ∞/k={load_path(finest.pressure):g}\nJ2, perfect plasticity, ν={c.nu:g}; analytical reference: incompressible limit")
    save(fig, "stress_comparison")

    fig = _figure(10., 5.); axes = fig.subplots(1, 2)
    alpha = np.array([s.alpha for s in finest.state.tpsa.material_states])
    for ax, values, title in zip(axes, (alpha, (alpha > 1e-8).astype(float)), ("Accumulated plastic strain", "Yielded cells (α > 1e−8)")):
        collection = PolyCollection(list(verts), array=values, cmap="viridis", rasterized=True)
        ax.add_collection(collection); outline(ax); ax.set_title(title)
        fig.colorbar(collection, ax=ax, shrink=.8)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Prototype — plastic-zone comparison; orange curve: analytical ellipse")
    save(fig, "plastic_zone")

    fig = _figure(10., 4.5); axes = fig.subplots(1, 2)
    spacing = [r["spacing_over_R"] for r in summaries]
    for key, title in (("mean_stress_relative_L2", "Mean stress"), ("maximum_shear_relative_L2", "Maximum shear")):
        axes[0].loglog(spacing, [r[key] for r in summaries], "o-", label=title)
    axes[0].set(xlabel="h / R", ylabel="Relative discrete L2 error", title="Volume-weighted stress error")
    axes[0].invert_xaxis(); axes[0].legend(); axes[0].grid(True, which="both", alpha=.25)
    for key, title in (("mean_stress_max_error_percent_eq41", "Mean stress"), ("maximum_shear_max_error_percent_eq41", "Maximum shear")):
        axes[1].loglog(spacing, [r[key] for r in summaries], "o-", label=title)
    axes[1].set(xlabel="h / R", ylabel="Maximum normalized error [%]", title="Paper's Eq. (41), maximum over cells")
    axes[1].invert_xaxis(); axes[1].legend(); axes[1].grid(True, which="both", alpha=.25)
    fig.suptitle("Prototype — staircase refinement against the circular analytical solution")
    save(fig, "convergence")

    table = folder/f"{PREFIX}_mesh_summary.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    paths["mesh_summary"] = table
    metadata = {
        "solver_package": "prototype", "jacobian": "analytical", "source": SOURCE,
        "reference_equations": [10, 15, 16, 17, 18, 22, 24, 41],
        "geometry": {"hole_radius": 1., "half_width": c.half_width,
                     "approximation": "Cartesian cells selected by centre radius >= 1; traction-free staircase hole"},
        "material": {"model": "3D J2 in plane strain", "E_over_k": 100., "nu": c.nu,
                     "sigma_y_over_k": float(np.sqrt(3.)), "hardening": 0.},
        "loading": {"final_pressure_over_k": finest.pressure, "final_tau_over_k": load_path(finest.pressure),
                    "path": "tau=0 for P<=1; tau=-0.3*(P-1) for 1<P<=2; p_inside=0",
                    "outer_boundary": "Exact elastic-domain displacement, Eq.(17), at each target; eliminates remote-traction truncation for the circular reference",
                    "hole_boundary": "Zero integrated traction on each actual staircase face"},
        "analytical_axes": {"x": ax_len, "y": ay_len},
        "comparison_limits": ["Galin is exact for this J2 material only at nu=0.5; finite nu introduces constitutive mismatch.",
                              "The discrete domain has a staircase hole, so reported errors include geometry error.",
                              "Outer displacement truncation replaces the paper's remote traction protocol.",
                              "No analytical displacement comparison inside the compressive plastic region is claimed.",
                              "Stress is from constitutive material histories, not reconstructed from numerical face traction.",
                              "Maximum shear is sqrt((sxx-syy)^2/4+sxy^2); mean is (sxx+syy)/2.",
                              "The zero-hardening model has no artificial stabilizing hardening."],
        "cases": summaries, "files": {key: str(path) for key, path in paths.items()},
    }
    summary_path = folder/f"{PREFIX}_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2)+"\n")
    paths["summary"] = summary_path
    return paths
