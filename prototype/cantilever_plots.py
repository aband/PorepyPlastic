"""PNG comparison figures for the paper's cantilever; no shared exporter changes."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.collections import PolyCollection
from scipy.interpolate import RegularGridInterpolator
import scipy.sparse as sps

from prototype.cantilever import CantileverResult, RESPONSE_COLUMNS


DATA = Path(__file__).with_name("data")


def _figure(width: float = 7.5, height: float = 5.) -> Figure:
    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def export_results(main: CantileverResult, study: list[CantileverResult], folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"cantilever_3d_retained_{main.jacobian}"
    paths: dict[str, Path] = {}
    reference = np.loadtxt(DATA / "cantilever_paper_figure3.csv", delimiter=",", skiprows=1)
    reference_meta = json.loads((DATA / "cantilever_paper_figure3.json").read_text())
    paper_A, paper_B = float(reference[10, 2]), float(reference[30, 2])
    case, response = main.case, main.response
    q = (len(response) - 1) // 4
    grading_label = "Both ends" if case.grading == "both-ends" else "Clamp only"

    def mesh_label(grading: str, ratio: float) -> str:
        if ratio == 1.:
            return "Uniform"
        ends = "Both ends" if grading == "both-ends" else "Clamp only"
        return f"{ends}, width ratio {ratio:g}"

    study_groups = sorted({(r.case.grading, r.case.grading_ratio) for r in study})

    def save(fig: Figure, name: str) -> None:
        path = folder / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=180)
        fig.clear()
        paths[name] = path

    # Orthographic projection on ordinary axes avoids clipping an extremely
    # slender 3D beam to Matplotlib's square Axes3D viewport.
    angle = np.deg2rad(20.)
    horizontal = np.array([np.cos(angle), np.sin(angle), 0.])
    vertical = np.array([-np.sin(angle)**2, np.cos(angle)*np.sin(angle), np.cos(angle)])
    projection = np.stack((horizontal, vertical), axis=1)
    depth = np.cross(horizontal, vertical)
    fig = _figure(10., 3.5)
    ax = fig.subplots()
    corners = np.array([[x, y, z] for x in (0., 100.) for y in (-5., 5.) for z in (-.5, .5)])
    corners[:, 2] *= 5.
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    order = np.argsort([np.mean(corners[f] @ depth) for f in faces])
    ax.add_collection(PolyCollection([corners[faces[i]] @ projection for i in order],
        facecolors=["#4d5961" if i == 0 else "#b9dbe5" for i in order], edgecolors="#344750"))
    for y in np.linspace(-4., 4., 5):
        start = np.array([100., y, 0.]) @ projection
        end = np.array([100., y, 8.]) @ projection
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle":"->", "color":"#007fa3"})
    ax.text(0., 7., "Clamp: u = 0")
    ax.autoscale_view(); ax.margins(x=.08, y=.4); ax.set_aspect("equal"); ax.set_axis_off()
    fig.text(.62, .80, r"Tip traction $t_z$: 0 → +0.6 → −0.6 → 0 MPa", ha="center", fontsize=10)
    fig.suptitle("3D cantilever — Nagler et al., section 4.1")
    fig.text(.5, .015, "100 × 10 × 1 mm; E = 210000 MPa, ν = 0.3, Y = 240 MPa, H₀ = 7000 MPa, θ = 0.5\nAll unlabelled surfaces: zero traction. Thickness enlarged in this schematic.", ha="center", fontsize=9)
    save(fig, "setup")

    fig = _figure(10., 5.)
    axes = fig.subplots(2, 1)
    for ax, xs, title in zip(axes, (np.linspace(0., 100., case.shape[0] + 1), case.axes[0]),
                              ("Uniform x spacing", f"{grading_label}: largest/smallest cell width = {case.grading_ratio:g}"), strict=True):
        for x in xs: ax.plot([x, x], [-.5, .5], color="#3d5c6b", lw=.7)
        for z in case.axes[2]: ax.plot([0., 100.], [z, z], color="#3d5c6b", lw=.7)
        ax.set(xlim=(0, 100), ylim=(-.5, .5), xlabel="x [mm]", ylabel="z [mm]", title=title)
    fig.suptitle(f"Mesh strategies — x–z view, {case.shape[1]} cells across width (z axis stretched)")
    save(fig, "meshes")

    fig = _figure()
    ax = fig.subplots()
    ax.plot(reference[:, 2], reference[:, 1], "k-", lw=1.7, label="Paper Abaqus curve (digitized)")
    ax.plot(response[:, 3], response[:, 2], "o-", color="#008fa1", ms=3, lw=1.2,
            label=f"TPSA {case.shape[0]} × {case.shape[1]} × {case.shape[2]}, {grading_label.lower()}, {len(response)-1} steps")
    for index, marker, label in ((q, "s", "A"), (3*q, "^", "B")):
        ax.scatter(response[index, 3], response[index, 2], marker=marker, s=70, color="#c04a00", zorder=5)
        ax.annotate(label, (response[index, 3], response[index, 2]), xytext=(8, -16 if label == "A" else 10), textcoords="offset points")
    first = np.flatnonzero(response[:, 4] > 1e-10)
    if first.size:
        row = response[first[0]]
        ax.scatter(row[3], row[2], facecolors="none", edgecolors="#b42052", s=110, zorder=6, label="First detected plasticity")
    ax.set(xlabel=r"Mean tip displacement $\overline{u_z}$ [mm]", ylabel=r"Tip traction $t_z$ [MPa]", title="Load–deflection cycle (paper Figure 3)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    save(fig, "load_deflection")

    ref_u = np.interp(np.linspace(0., 1., len(response)), np.linspace(0., 1., len(reference)), reference[:, 2])
    error = np.full(len(response), np.nan)
    valid = np.abs(ref_u) > 1e-4
    error[valid] = (response[valid, 3] - ref_u[valid]) / ref_u[valid]
    fig = _figure(); ax = fig.subplots()
    for start, stop, label in ((0, q + 1, "0 → A"), (q, 3*q + 1, "A → B"), (3*q, len(response), "B → 0")):
        ax.plot(response[start:stop, 2], error[start:stop], "o-", ms=3, label=label)
    ax.axhline(0., color="k", lw=.6)
    ax.set(xlabel=r"Tip traction $t_z$ [MPa]", ylabel=r"$(u_z^{TPSA}-u_z^{ref})/u_z^{ref}$", title="Tip-displacement error vs digitized paper curve")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    ax.set_title("Tip-displacement error vs digitized paper curve\nZero reference displacement is omitted", fontsize=12)
    save(fig, "relative_error")

    if study:
        for endpoint, ref_value, offset in (("A", paper_A, 1), ("B", paper_B, 3)):
            fig = _figure(); ax = fig.subplots()
            for grading, ratio in study_groups:
                group = sorted((r for r in study if (r.case.grading, r.case.grading_ratio) == (grading, ratio)), key=lambda r: r.case.operators.num_dofs)
                dofs = [r.case.operators.num_dofs for r in group]
                errors = [abs((r.response[offset * ((len(r.response)-1)//4), 3] - ref_value) / ref_value) for r in group]
                ax.loglog(dofs, errors, "o-", label=mesh_label(grading, ratio))
            ax.set(xlabel="Total TPSA DOFs (cells + retained faces)", ylabel="Absolute relative tip-displacement error", title=f"Mesh comparison at {endpoint} (paper Figures 6–7)")
            ax.grid(True, which="both", alpha=.3); ax.legend()
            save(fig, f"convergence_{endpoint}")
        fig = _figure(); ax = fig.subplots()
        for grading, ratio in study_groups:
            group = sorted((r for r in study if (r.case.grading, r.case.grading_ratio) == (grading, ratio)), key=lambda r: r.case.operators.num_dofs)
            ax.plot([r.case.operators.num_dofs for r in group], [r.response[(len(r.response)-1)//4, 3] for r in group], "o-",
                    label=mesh_label(grading, ratio))
        ax.axhline(paper_A, color="k", ls="--", label=f"Paper reference ≈ {paper_A:.3f} mm (digitized)")
        ax.set(xlabel="Total TPSA DOFs (cells + retained faces)", ylabel=r"Mean tip displacement at A [mm]", title="Peak deflection vs mesh size (paper Figure 5)")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        save(fig, "peak_deflection")

    state, grid, ops = main.state_A, case.grid, case.operators
    stress = np.stack([s.stress.to_numpy() for s in state.tpsa.material_states], axis=2)
    dev = stress - np.eye(3)[:, :, None] * np.trace(stress, axis1=0, axis2=1)[None, None, :] / 3.
    vm = np.sqrt(1.5 * np.einsum("ijc,ijc->c", dev, dev))
    centres = tuple((a[1:] + a[:-1]) / 2 for a in case.axes)
    u = state.tpsa.x[ops.u_slice].reshape((3, *case.shape), order="F").transpose(1, 2, 3, 0)
    nodal = RegularGridInterpolator(centres, u, bounds_error=False, fill_value=None)(grid.nodes.T)
    nodal[np.isclose(grid.nodes[0], 0.)] = 0.
    # Use retained tip-face motion for the displayed end surface.
    assert state.tpsa.face_displacement is not None
    tip_u = np.empty((case.shape[1], case.shape[2], 3))
    for face in case.tip_faces:
        iy = int(np.argmin(abs(centres[1] - grid.face_centers[1, face])))
        iz = int(np.argmin(abs(centres[2] - grid.face_centers[2, face])))
        tip_u[iy, iz] = state.tpsa.face_displacement[:, face]
    tip_nodes = np.isclose(grid.nodes[0], 100.)
    nodal[tip_nodes] = RegularGridInterpolator(centres[1:], tip_u, bounds_error=False, fill_value=None)(grid.nodes[1:, tip_nodes].T)
    deformed = grid.nodes.T + nodal
    polygons, colours = [], []
    fn, incidence = grid.face_nodes.tocsc(), sps.csr_array(grid.cell_faces)
    norm = Normalize(0., max(240., float(vm.max())))
    for face in ops.boundary_faces:
        nodes = fn.indices[fn.indptr[face]:fn.indptr[face + 1]]
        xyz = grid.nodes[:, nodes].T
        active = np.argsort(np.ptp(xyz, axis=0))[-2:]
        local = xyz[:, active] - xyz[:, active].mean(axis=0)
        order = np.argsort(np.arctan2(local[:, 1], local[:, 0]))
        polygons.append(deformed[nodes[order]])
        cell = incidence.indices[incidence.indptr[face]]
        colours.append(colormaps["viridis"](norm(vm[cell])))
    projected = [np.asarray(p) * np.array([1., 1., 5.]) for p in polygons]
    order = np.argsort([np.mean(p @ depth) for p in projected])
    fig = _figure(10., 3.5); ax = fig.subplots()
    ax.add_collection(PolyCollection([projected[i] @ projection for i in order],
        facecolors=[colours[i] for i in order], edgecolors="#333333", linewidths=.15))
    ax.autoscale_view(); ax.margins(.05); ax.set_aspect("equal"); ax.set_axis_off()
    from matplotlib.cm import ScalarMappable
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=.65, label="von Mises stress [MPa]", pad=.02)
    fig.suptitle(f"Deformed cantilever at A — mean tip u_z = {response[q, 3]:.3f} mm")
    fig.text(.5, .015, "Displacement scale 1; z axis enlarged 5× for visibility. Surface geometry interpolated from cell/face displacement.", ha="center", fontsize=8)
    save(fig, "deformed_A")

    np.savetxt(folder / f"{prefix}_response.csv", response, delimiter=",", header=",".join(RESPONSE_COLUMNS), comments="")
    study_rows = []
    for result in study:
        n = (len(result.response)-1)//4
        nx, ny, nz = result.case.shape
        study_rows.append([nx, ny, nz, result.case.grading_ratio, result.case.operators.num_dofs,
                           result.response[n, 3], result.response[3*n, 3], result.response[:, 4].max(), result.case.grading])
    if study_rows:
        with (folder / f"{prefix}_mesh_comparison.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["nx", "ny", "nz", "grading_ratio", "dofs", "tip_A_mm", "tip_B_mm", "max_alpha", "grading"])
            writer.writerows(study_rows)
    metadata = {
        "paper": reference_meta, "units": {"length":"mm", "stress":"MPa", "force":"N"},
        "shape":case.shape, "grading_ratio":case.grading_ratio, "dofs":ops.num_dofs,
        "grading":case.grading, "grading_ratio_definition":"largest/smallest axial cell width",
        "x_coordinates_mm":case.axes[0].tolist(),
        "minimum_x_cell_width_mm":float(np.min(np.diff(case.axes[0]))),
        "maximum_x_cell_width_mm":float(np.max(np.diff(case.axes[0]))),
        "jacobian":main.jacobian, "load_steps":len(response)-1,
        "newton":{"max_iterations":35,"atol_blocks":[2e-6,2e-7,2e-7,2e-6],"rtol_blocks":[1e-9]*4,"backtracking":True},
        "validation_status":"Benchmark comparison only; assess discretization error and yielding using the reported values.",
        "material":{"E":210000.,"nu":.3,"yield_stress":240.,"H_bar":7000.,"theta":.5},
        "boundary_conditions":"x=0: full displacement clamp; x=100: uniform transverse z traction; other faces: zero traction",
        "hardening_convention":"paper gamma = sqrt(3/2) * code alpha; H_bar=H0 gives the same linear combined law",
        "differences_from_paper":["TPSA instead of quadratic TDNNS finite elements", "At least two cells across width and thickness resolve the face-centre clamp", "Geometric grading ratio selected explicitly because it is unspecified in the paper", "Abaqus reference values digitized from Figure 3, not raw simulation data"]
            + (["Symmetric grading at both ends is an additional mesh experiment"] if case.grading == "both-ends" else []),
        "tip_A_mm":float(response[q,3]), "tip_B_mm":float(response[3*q,3]),
        "relative_error_A":float((response[q,3]-paper_A)/paper_A),
        "relative_error_B":float((response[3*q,3]-paper_B)/paper_B),
        "max_alpha":float(response[:,4].max()), "maximum_force_imbalance_N":float(response[:,7].max()),
        "png_files":{k:str(p) for k,p in paths.items()},
    }
    path = folder / f"{prefix}_summary.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    paths["summary"] = path
    return paths
