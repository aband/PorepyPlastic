"""CSV, JSON, PNG and reconstructed fields for prototype elastic bending studies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections.abc import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prototype.elastic_bending import ElasticBendingResult, FloatArray, Shape

Record = tuple[str, ElasticBendingResult]
PREFIX = "prototype_elastic_bending"


def _paper_tip(result: ElasticBendingResult) -> float | None:
    if result.case.kind != "cantilever":
        return None
    data = np.loadtxt(Path(__file__).with_name("data") / "cantilever_paper_figure3.csv", delimiter=",", skiprows=1)
    return float(data[1, 2] * result.case.traction / data[1, 1])


def export_results(records: Sequence[Record], folder: Path, *, plots: bool = True) -> dict[str, Path]:
    """Checkpoint every completed level; write finest fields and figures on request."""
    if not records:
        raise ValueError("At least one completed result is required.")
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"csv": folder/f"{PREFIX}_convergence.csv", "summary": folder/f"{PREFIX}_summary.json"}
    rows = []
    previous: dict[str, dict[str, object]] = {}
    for series, result in records:
        case, m = result.case, result.metrics
        row = {"series": series, "problem": case.kind, "nx": case.shape[0], "ny": case.shape[1],
               "nz": case.shape[2], "traction_MPa": case.traction, "curvature_per_mm": case.curvature, **m}
        row["paper_elastic_tip_mm"] = _paper_tip(result)
        row["paper_tip_relative_error"] = (None if case.kind != "cantilever" else
            abs(m["tip_mm"] - row["paper_elastic_tip_mm"])/abs(row["paper_elastic_tip_mm"]))
        row["observed_stress_order"] = row["observed_displacement_order"] = None
        old = previous.get(series)
        # Orders against an exact solution only, and only for proportional refinement.
        if old is not None and case.kind == "pure_bending":
            ratios = np.array([old[name]/row[name] for name in ("dx_max_mm", "dy_max_mm", "dz_max_mm")])
            if ratios.min() > 1 and np.allclose(ratios, ratios[0]):
                for field, name in (("stress_relative_l2", "observed_stress_order"),
                                    ("displacement_relative_l2", "observed_displacement_order")):
                    if old[field] > 0 and row[field] > 0:
                        row[name] = float(np.log(old[field]/row[field])/np.log(ratios[0]))
        rows.append(row)
        previous[series] = row
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with paths["csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "solver": "Prototype retained TPSA; upstream elastic cell Schur complement with prototype face recovery/residual checks",
        "linear_solve": "Shared direct solver for smaller cases; unit-scaled, bandwidth-ordered SuperLU for cantilever cross-sections with more than 64 cells. Same elastic equations; original residuals are checked.",
        "units": {"length": "mm", "stress": "MPa", "force": "N"},
        "references": {
            "pure_bending": "Exact 3D polynomial displacement and sigma_xx=-E*k*z, all other stresses zero. Left boundary uses exact face-averaged Poisson warping.",
            "cantilever_displacement": "Euler-Bernoulli estimate and digitized first elastic point of Nagler et al. Figure 3 (doi:10.1002/nme.7070). Neither is an exact 3D field reference.",
            "cantilever_stress": "Saint-Venant beam approximation. Interior xx error excludes x<0.1L and x>0.9L; it is an engineering comparison, not an exact 3D error.",
            "moments": "Continuum section equilibrium and midpoint-sampled reference. The exact linear axial stress has a 1/nz^2 midpoint moment deficit.",
        },
        "error_definition": "sqrt(sum_cell(V*||computed-reference||^2) / sum_cell(V*||reference||^2)); tensor norm is Frobenius. Displacement is compared at cell centers. Face moment errors use axial trapezoidal weights.",
        "limitations": ["Axial-only refinement is not full 3D convergence.",
                        "Peak stresses at clamp edges are not used as convergence targets.",
                        "The study is elastic; it does not certify the plastic loading/unloading cycle.",
                        "Small residual and global force balance alone do not demonstrate accurate bending stress."],
        "results": rows,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, allow_nan=False))
    if not plots:
        return paths

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for series in dict.fromkeys(s for s, _ in records):
        selected = [r for s, r in records if s == series]
        if selected[0].case.kind == "pure_bending":
            h = [r.metrics["dx_max_mm"] for r in selected]
            axes[0].loglog(h, [r.metrics["displacement_relative_l2"] for r in selected], "o-", label="Displacement L2")
            axes[0].loglog(h, [r.metrics["stress_relative_l2"] for r in selected], "s-", label="Stress L2")
            axes[0].loglog(h, [r.metrics["interior_face_moment_relative_l2"] for r in selected], "^-", label="Interior moment")
        elif "cross" not in series:
            axes[1].semilogx([r.case.shape[0] for r in selected], [r.metrics["tip_mm"] for r in selected], "o-", label=series)
    cantilever = [r for _, r in records if r.case.kind == "cantilever"]
    if cantilever:
        axes[1].axhline(_paper_tip(cantilever[-1]), color="black", ls="--", label="Digitized paper elastic reference")
        axes[1].axhline(cantilever[-1].metrics["reference_tip_mm"], color="gray", ls=":", label="Euler-Bernoulli estimate")
    axes[0].set(xlabel="Maximum axial cell width [mm]", ylabel="Relative error", title="Exact 3D pure bending: simultaneous refinement")
    axes[1].set(xlabel="Axial cells", ylabel="Mean tip displacement [mm]", title="Elastic cantilever")
    for ax in axes:
        if ax.lines:
            ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    paths["convergence"] = folder/f"{PREFIX}_convergence.png"
    fig.savefig(paths["convergence"], dpi=180)
    plt.close(fig)

    cross = [r for series, r in records if "cross" in series]
    if cross:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        indices = np.arange(len(cross))
        for ax, field, title, multiplier in zip(axes,
            ("tip_mm", "interior_sigma_xx_relative_l2", "interior_face_moment_relative_l2"),
            ("Mean tip displacement [mm]", "Interior axial stress difference [%]", "Interior moment error [%]"),
            (1., 100., 100.)):
            ax.plot(indices, [multiplier*r.metrics[field] for r in cross], "o-")
            ax.set(xticks=indices, xticklabels=[f"{r.case.shape[1]} × {r.case.shape[2]}" for r in cross],
                   xlabel="Width × thickness cells", ylabel=title)
            ax.grid(alpha=.25)
        axes[0].axhline(_paper_tip(cross[-1]), color="black", ls="--", label="Paper")
        axes[0].axhline(cross[-1].metrics["reference_tip_mm"], color="gray", ls=":", label="Beam estimate")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Prototype cantilever: cross-section sensitivity at {cross[-1].case.shape[0]} axial cells")
        paths["cross_section"] = folder/f"{PREFIX}_cross_section.png"
        fig.savefig(paths["cross_section"], dpi=180); plt.close(fig)

    groups = [(kind, [r for series, r in records if r.case.kind == kind and "cross" not in series])
              for kind in ("pure_bending", "cantilever")]
    if cross:
        groups.append(("cantilever_cross_section", cross))
    for kind, selected in groups:
        if not selected:
            continue
        # Keep separate fields for axial and cross-section refinement series.
        finest = max(selected, key=lambda r: r.metrics["cells"])
        stem = f"{PREFIX}_{kind}"
        paths[f"{kind}_fields"] = folder/f"{stem}_fields.npz"
        np.savez_compressed(paths[f"{kind}_fields"], cell_centers=finest.case.operators.grid.cell_centers,
            cell_volumes=finest.case.operators.grid.cell_volumes, cell_unknowns=finest.x,
            face_displacement=finest.face_displacement, strain=finest.strain, stress=finest.stress,
            reference_stress=finest.reference_stress, shape=finest.case.shape,
            x_nodes=finest.case.axes[0], y_nodes=finest.case.axes[1], z_nodes=finest.case.axes[2])
        for label, data, header in (
            ("face_sections", finest.face_sections, "x_mm,M_numerical_Nmm,M_continuum_Nmm,shear_force_N"),
            ("cell_sections", finest.cell_sections, "x_mm,M_material_Nmm,M_continuum_Nmm,M_sampled_reference_Nmm"),
        ):
            paths[f"{kind}_{label}"] = folder/f"{stem}_{label}.csv"
            np.savetxt(paths[f"{kind}_{label}"], data, delimiter=",", header=header, comments="")
        _stress_plot(finest, folder/f"{stem}_stress.png")
        paths[f"{kind}_stress"] = folder/f"{stem}_stress.png"
        paths[f"{kind}_profiles"] = folder/f"{stem}_profiles.png"
        _stress_profiles(finest.stress, finest.reference_stress, finest.case.axes,
                         finest.case.shape, paths[f"{kind}_profiles"],
                         reference_label="Exact" if finest.case.kind == "pure_bending" else "Beam approximation")
        fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
        for result in selected:
            # Plot a manageable set of refinement levels.
            if result is not selected[0] and result is not finest and result is not selected[len(selected)//2]:
                continue
            ax.plot(result.face_sections[:, 0], result.face_sections[:, 1], label=f"Numerical, {result.case.shape}")
        ax.plot(finest.cell_sections[:, 0], finest.cell_sections[:, 1], label="Finest material stress", color="#228c70")
        ax.plot(finest.face_sections[:, 0], finest.face_sections[:, 2], "k--", label="Continuum section moment")
        ax.set(xlabel="x [mm]", ylabel="Bending moment [N mm]", title=f"{kind.replace('_', ' ').title()}: section moments")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        paths[f"{kind}_moments"] = folder/f"{stem}_moments.png"
        fig.savefig(paths[f"{kind}_moments"], dpi=180); plt.close(fig)
    return paths


def _stress_plot(result: ElasticBendingResult, path: Path) -> None:
    case = result.case
    j = case.shape[1]//2
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), layout="constrained")
    for row, (i, k, label) in enumerate(((0, 0, r"$\sigma_{xx}$"), (0, 2, r"$\sigma_{xz}$"))):
        numerical = result.stress[i, k].reshape(case.shape, order="F")[:, j, :].T
        reference = result.reference_stress[i, k].reshape(case.shape, order="F")[:, j, :].T
        limit = max(float(abs(numerical).max()), float(abs(reference).max()), 1e-12)
        for column, (title, field) in enumerate((("Reconstructed", numerical), ("Reference", reference), ("Difference", numerical-reference))):
            bound = max(float(abs(field).max()), 1e-12) if column == 2 else limit
            im = axes[row, column].pcolormesh(case.axes[0], case.axes[2], field, cmap="RdBu_r", vmin=-bound, vmax=bound, shading="flat")
            axes[row, column].set(title=f"{title} {label}", xlabel="x [mm]", ylabel="z [mm]")
            fig.colorbar(im, ax=axes[row, column], label="MPa", shrink=.8)
    reference_label = "exact 3D reference" if case.kind == "pure_bending" else "beam approximation; end effects differ"
    y = .5*(case.axes[1][j] + case.axes[1][j+1])
    fig.suptitle(f"Prototype {case.kind.replace('_', ' ')} {case.shape}, y={y:g} mm; {reference_label}")
    fig.savefig(path, dpi=180); plt.close(fig)


def _stress_profiles(
    stress: FloatArray, reference: FloatArray, nodes: tuple[FloatArray, FloatArray, FloatArray],
    shape: Shape, path: Path, *, reference_label: str = "Reference",
) -> None:
    """Width-averaged midspan profiles retain the small shear-stress scale."""
    x = .5*(nodes[0][1:]+nodes[0][:-1])
    z = .5*(nodes[2][1:]+nodes[2][:-1])
    ix = int(np.argmin(abs(x-.5*(nodes[0][0]+nodes[0][-1]))))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
    for ax, (i, j, label) in zip(axes, ((0, 0, r"$\sigma_{xx}$"), (0, 2, r"$\sigma_{xz}$"))):
        for values, style, legend in ((stress, "o-", "Reconstructed"), (reference, "k--", reference_label)):
            section = values[i, j].reshape(shape, order="F")[ix]
            profile = np.average(section, axis=0, weights=np.diff(nodes[1]))
            ax.plot(profile, z, style, label=legend)
        ax.set(xlabel=f"{label} [MPa]", ylabel="z [mm]")
        ax.grid(alpha=.25); ax.legend(fontsize=9)
    fig.suptitle(f"Width-averaged stress at x={x[ix]:.4g} mm; mesh {tuple(int(n) for n in shape)}")
    fig.savefig(path, dpi=180); plt.close(fig)
