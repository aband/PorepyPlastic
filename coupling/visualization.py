"""VTK and PNG export of the cell fields in a two-dimensional TPSA solution."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator, SymmetricalLogLocator
import numpy as np
import porepy as pp
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.grid import Grid
from porepy.viz.exporter import DataInput

if TYPE_CHECKING:
    from coupling.loading import LoadStepResult


def _validate_solution(grid: Grid, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate the shared [u, r, p] input contract before creating output."""
    if grid.dim != 2:
        raise ValueError("This exporter expects a two-dimensional TPSA grid.")
    values = np.asarray(x, dtype=np.float64)
    if values.shape != (4 * grid.num_cells,):
        raise ValueError(
            f"Expected a flat TPSA solution with {4 * grid.num_cells} entries."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("The TPSA solution must contain only finite values.")
    return values


def _validate_tensor(grid: Grid, tensor: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Validate a finite symmetric full 3D cell tensor before creating output."""
    if grid.dim != 2 or grid.num_cells < 1:
        raise ValueError(f"{name} export expects a nonempty two-dimensional grid.")
    values = np.asarray(tensor, dtype=np.float64)
    if values.shape != (3, 3, grid.num_cells) or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected finite {name} with shape (3, 3, {grid.num_cells}).")
    if not np.allclose(values, values.swapaxes(0, 1), rtol=1e-10, atol=1e-14):
        raise ValueError(f"{name} tensors must be symmetric.")
    return values


def _validate_strain(grid: Grid, epsilon: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate dimensionless symmetric cell tensors before creating output."""
    return _validate_tensor(grid, epsilon, "strain")


def _solution_data(grid: Grid, values: NDArray[np.float64]) -> list[DataInput]:
    """VTK cell fields for a validated TPSA vector."""
    nc = grid.num_cells
    displacement = np.zeros((3, nc))
    displacement[:2] = values[:2 * nc].reshape((2, nc), order="F")
    return [
        (grid, "displacement", displacement),
        (grid, "displacement_magnitude", np.linalg.norm(displacement, axis=0)),
        (grid, "rotation_stress", values[2 * nc:3 * nc]),
        (grid, "total_pressure", values[3 * nc:]),
    ]


def _tensor_data(grid: Grid, name: str, values: NDArray[np.float64]) -> list[DataInput]:
    """Nine row-major entries per tensor plus four named component fields."""
    data: list[DataInput] = [(grid, name, values.reshape((9, grid.num_cells), order="C"))]
    for component, i, j in [("xx", 0, 0), ("yy", 1, 1), ("xy", 0, 1), ("zz", 2, 2)]:
        data.append((grid, f"{name}_{component}", values[i, j]))
    return data


def _publish_pvd_collections(folder: Path, file_name: str) -> None:
    """Group this export's PVDs in pvd/, referencing VTUs through ../vtu/.

    PorePy writes both formats to one directory. Move only this export's static,
    series, and numbered step collections; preserve their times and metadata.
    Match numeric suffixes without depending on PorePy's zero-padding width.
    """
    prefix = Path(file_name).stem
    destination = folder / "pvd"
    destination.mkdir(parents=True, exist_ok=True)
    for source in (folder / "vtu").glob("*.pvd"):
        is_step = source.stem.startswith(prefix + "_") and source.stem[len(prefix) + 1:].isdigit()
        if source.stem != prefix and not is_step:
            continue
        collection = ET.parse(source)
        for dataset in collection.iter("DataSet"):
            reference = Path(dataset.attrib["file"])
            if not reference.is_absolute():
                dataset.set("file", (Path("..") / "vtu" / reference).as_posix())
        collection.write(destination / source.name, encoding="utf-8", xml_declaration=True)
        source.unlink()


def export_vtk(
    grid: Grid,
    x: NDArray[np.float64],
    *,
    epsilon: NDArray[np.float64] | None = None,
    folder_name: str | Path = "results",
    file_name: str = "displacement",
) -> Path:
    """Write a static TPSA solution with pp.Exporter and return its PVD path.

    VTUs go in folder_name/vtu; PVD collections go in folder_name/pvd and use
    relative references through ../vtu/.

    The grid geometry must be computed. The vector x follows test_tpsa.py:
    interleaved cell displacement components, then rotation stress, then total
    pressure. All exported values are cell data on the undeformed grid:
    displacement [m], displacement_magnitude [m], rotation_stress [Pa], and
    total_pressure [Pa]. Displacement is padded with a zero z component for VTK.
    Optional epsilon (3, 3, nc) adds dimensionless total strain: a nine-component
    strain field in row-major tensor order (xx, xy, xz, yx, yy, yz, zx, zy, zz),
    plus strain_xx, strain_yy, strain_xy, and strain_zz scalar fields. The xy
    component is tensor shear strain, equal to half the engineering shear.

    For deformation in ParaView, apply Cell Data to Point Data, then Warp By
    Vector using displacement. Material histories and analytical reference
    stresses are not part of the numerical solution exported here.
    """
    values = _validate_solution(grid, x)

    data = _solution_data(grid, values)
    if epsilon is not None:
        data.extend(_tensor_data(grid, "strain", _validate_strain(grid, epsilon)))
    folder = Path(folder_name)
    exporter = pp.Exporter(grid, Path(file_name), folder_name=folder / "vtu")
    # write_vtu also writes the PVD collection for this static grid.
    exporter.write_vtu(data)
    _publish_pvd_collections(folder, file_name)
    return folder / "pvd" / f"{file_name}.pvd"


def export_png(
    grid: Grid,
    x: NDArray[np.float64],
    *,
    folder_name: str | Path = "results",
    file_name: str = "displacement",
) -> Path:
    """Save cell displacement vectors on the undeformed mesh as a PNG.

    Uses PorePy's native save_img vector_value argument. A common, labeled
    scale factor makes the longest arrow 40% of the smallest cell's square-root
    area, preserving relative lengths and directions. Coordinates are in SI
    units with equal spatial scales. Only the figure created here is closed;
    the solution and grid are not modified.
    """
    values = _validate_solution(grid, x)
    displacement = np.zeros((3, grid.num_cells))
    displacement[:2] = values[:2 * grid.num_cells].reshape(
        (2, grid.num_cells), order="F"
    )
    max_displacement = float(np.max(np.linalg.norm(displacement, axis=0)))
    cell_size = float(np.sqrt(np.min(grid.cell_volumes)))
    vector_scale = (
        0.4 * cell_size / max_displacement if max_displacement > 0.0 else 1.0
    )
    folder = Path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{file_name}.png"

    with plt.rc_context({"font.size": 10, "savefig.dpi": 180}):
        fig = plt.figure(figsize=(6.4, 5.6))
        try:
            pp.save_img(
                path,
                grid,
                vector_value=displacement,
                vector_scale=vector_scale,
                plot_2d=True,
                fig_num=fig.number,
                title="Displacement",
                rgb=[0.96, 0.96, 0.96],
                linewidth=0.8,
            )
            # PorePy's arrows use fixed endpoint padding by default. Remove it
            # so even small vectors retain their scaled lengths.
            axes = fig.axes[0]
            for arrow in axes.patches:
                if isinstance(arrow, FancyArrowPatch):
                    arrow.shrinkA = 0.0
                    arrow.shrinkB = 0.0
                    arrow.set_mutation_scale(8)
            axes.set_aspect("equal")
            axes.set_xlabel("x [m]")
            axes.set_ylabel("y [m]")
            axes.text(
                0.5, -0.17,
                #f"Arrow lengths = {vector_scale:.4g} × displacement; rectangular mesh",
                f"rectangular mesh",
                transform=axes.transAxes,
                ha="center",
                fontsize=9,
            )
            fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
        finally:
            plt.close(fig)
    return path


def export_strain_png(
    grid: Grid,
    epsilon: NDArray[np.float64],
    *,
    folder_name: str | Path = "results",
    file_name: str = "plane_strain",
) -> dict[str, Path]:
    """Save total-strain component maps and in-plane principal-strain glyphs.

    Returns paths under keys xx, yy, xy, and principal. Component maps use one
    symmetric color range; all-zero fields use [-1, 1]. Glyph lengths share one
    scale, with red for extension and blue for contraction. Equal principal
    strains use circles to avoid assigning an arbitrary direction. Only roundoff
    relative to the largest principal strain is suppressed in the glyph plot.
    Values, grid geometry, and existing figures are preserved.
    """
    values = _validate_strain(grid, epsilon)
    components = {"xx": values[0, 0], "yy": values[1, 1], "xy": values[0, 1]}
    limit = max(float(np.max(np.abs(value))) for value in components.values()) or 1.0
    folder = Path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    with plt.rc_context({"font.size": 10, "savefig.dpi": 180}):
        for name, value in components.items():
            path = folder / f"{file_name}_epsilon_{name}.png"
            fig = plt.figure(figsize=(6.4, 5.6))
            try:
                pp.save_img(
                    path, grid, cell_value=value, plot_2d=True, fig_num=fig.number,
                    title=rf"Total strain $\varepsilon_{{{name}}}$",
                    color_map="RdBu_r", color_map_limits=[-limit, limit], linewidth=0.5,
                )
                axes = fig.axes[0]
                axes.set_aspect("equal")
                axes.set_xlabel("x [m]")
                axes.set_ylabel("y [m]")
                fig.axes[1].set_ylabel("Strain [dimensionless]")
                fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
            finally:
                plt.close(fig)
            paths[name] = path
        paths["principal"] = folder / f"{file_name}_principal_strain.png"
        _export_principal_strain_png(grid, values, paths["principal"])
    return paths


def _export_principal_strain_png(
    grid: Grid, epsilon: NDArray[np.float64], path: Path,
) -> None:
    """Plot principal axes as centered segments; repeated eigenvalues as circles."""
    # Average the transpose to remove any asymmetry accepted by the input tolerance.
    in_plane = np.moveaxis(
        0.5 * (epsilon[:2, :2] + epsilon[:2, :2].swapaxes(0, 1)), 2, 0,
    )
    principal, directions = np.linalg.eigh(in_plane)
    peak = float(np.max(np.abs(principal)))
    tolerance = 64 * np.finfo(float).eps * peak
    # For the Cartesian example, the smallest face length is the smallest cell width.
    half_length = 0.3 * float(np.min(grid.face_areas))
    extension, contraction = "#c62828", "#1565c0"
    fig = plt.figure(figsize=(6.4, 6.4))
    try:
        pp.plot_grid(
            grid, plot_2d=True, fig_num=fig.number, if_plot=False,
            title="Principal strain (in-plane)", rgb=[0.97, 0.97, 0.97], linewidth=0.5,
        )
        axes = fig.axes[0]
        for cell, eigenvalues in enumerate(principal):
            center = grid.cell_centers[:2, cell]
            if np.max(np.abs(eigenvalues)) <= tolerance:
                continue
            if abs(eigenvalues[1] - eigenvalues[0]) <= tolerance:
                value = float(np.mean(eigenvalues))
                axes.add_patch(Circle(
                    (float(center[0]), float(center[1])),
                    radius=half_length * abs(value) / peak, fill=False,
                    edgecolor=extension if value > 0 else contraction, linewidth=1.8,
                ))
            else:
                for axis, value in enumerate(eigenvalues):
                    if abs(value) <= tolerance:
                        continue
                    offset = half_length * (abs(value) / peak) * directions[cell, :, axis]
                    endpoints = np.column_stack((center - offset, center + offset))
                    axes.plot(
                        endpoints[0], endpoints[1], linewidth=2.0,
                        color=extension if value > 0 else contraction,
                        solid_capstyle="round",
                    )
        axes.set_aspect("equal")
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        axes.legend(handles=[
            Line2D([], [], color=extension, label="Extension (+)"),
            Line2D([], [], color=contraction, label="Contraction (−)"),
            Line2D([], [], color="0.3", marker="o", markerfacecolor="none",
                   linestyle="none", label="Equal principal strains"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
        caption = (
            f"Longest segment or circle diameter represents |ε| = {peak:.3g}"
            if peak > 0 else "Zero strain: no glyphs"
        )
        fig.text(0.5, 0.02, caption, ha="center", fontsize=9)
        fig.subplots_adjust(bottom=0.23)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    finally:
        plt.close(fig)


def export_jacobian_png(
    matrix: sps.spmatrix | sps.sparray | NDArray[np.float64],
    *,
    folder_name: str | Path = "results",
    file_name: str = "jacobian",
    title: str = "Coupled TPSA Jacobian",
    linthresh: float = 1e-12,
) -> Path:
    """Plot signed entries of a coupled [u,r,p] Jacobian and return its PNG path.

    Accept a finite (4*nc,4*nc) dense or sparse matrix. Rows are [R_u,R_r,R_p];
    columns are [u,r,p], with displacement components interleaved by cell.
    Sparse input is densified for this diagnostic heatmap. Values retain their
    block-dependent SI units; no row/column normalization is applied.
    Symmetric-log colors preserve sign and are linear within +/-linthresh.
    Zero is white, negative entries blue, positive entries red. The matrix and
    existing figures are preserved. This function does not assemble a Jacobian.
    """
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0 or matrix.shape[0] % 4:
        raise ValueError("Expected a square coupled Jacobian with shape (4*nc, 4*nc).")
    if not np.isfinite(linthresh) or linthresh <= 0:
        raise ValueError("linthresh must be finite and positive.")
    if not file_name or Path(file_name).name != file_name or file_name in (".", ".."):
        raise ValueError("file_name must be a nonempty file prefix without directories.")
    values = np.asarray(matrix if isinstance(matrix, np.ndarray) else matrix.toarray(), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("The Jacobian must contain only finite values.")
    size = values.shape[0]
    nc = size // 4
    limit = max(float(np.max(np.abs(values))), linthresh)
    norm = SymLogNorm(linthresh=linthresh, vmin=-limit, vmax=limit, base=10)
    folder = Path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{file_name}.png"
    with plt.rc_context({"font.size": 10, "savefig.dpi": 180}):
        fig, axes = plt.subplots(figsize=(8.0, 7.0))
        try:
            heatmap = axes.imshow(values, cmap="RdBu_r", norm=norm,
                                  interpolation="nearest", origin="upper", aspect="equal")
            centers = np.array([nc, 2.5 * nc, 3.5 * nc]) - 0.5
            axes.set_xticks(centers, [r"$u$", r"$r$", r"$p$"])
            axes.set_yticks(centers, [r"$R_u$", r"$R_r$", r"$R_p$"])
            for boundary in (2 * nc - 0.5, 3 * nc - 0.5):
                axes.axhline(boundary, color="0.25", linestyle="--", linewidth=0.8)
                axes.axvline(boundary, color="0.25", linestyle="--", linewidth=0.8)
            axes.set_xlabel("Unknown columns (displacement interleaved by cell)")
            axes.set_ylabel("Residual rows")
            axes.set_title(title, fontsize=12, pad=12)
            ticks = SymmetricalLogLocator(linthresh=linthresh, base=10)
            ticks.set_params(numticks=9)
            colorbar = fig.colorbar(heatmap, ax=axes, fraction=0.047, pad=0.045, ticks=ticks)
            colorbar.set_label("Jacobian entry (symmetric log scale)")
            fig.text(0.5, 0.065, f"{size} × {size} matrix · Raw entries in block-dependent SI units",
                     ha="center", fontsize=9)
            fig.text(0.5, 0.035, f"Blue: negative · White: zero · Red: positive · Linear for |entry| ≤ {linthresh:g}",
                     ha="center", fontsize=9)
            fig.subplots_adjust(left=0.12, right=0.85, bottom=0.17, top=0.86)
            fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
        finally:
            plt.close(fig)
    return path


def export_plastic_history(
    grid: Grid,
    records: Sequence[LoadStepResult],
    *,
    folder_name: str | Path = "results",
    file_name: str = "plastic_plane_strain",
) -> dict[str, Path]:
    """Export accepted states to a pp.Exporter time series and two history PNGs.

    records must be accepted steps from the same grid, in chronological order.
    PVD times and the load_step cell field are 1..N (sequence indices, not physical
    time). load_factor is a separate cell field, preserving unloading/reversals.
    No initial state is invented or appended. All inputs are validated before
    files are written; records and grid remain unchanged. VTUs go in the vtu
    subfolder and PVD collections in pvd; PNGs stay directly in folder_name.

    Cell fields: displacement (zero z), its magnitude, rotation_stress,
    total_pressure, strain, stress, plastic_strain, backstress, and alpha.
    Tensors retain all nine row-major components, including zz/xz/yz; xx, yy, xy,
    zz are also exported separately. Strain/plastic strain use tensor shear.
    Stress/backstress use Pa, displacement m, and strains/alpha are dimensionless.
    The mesh remains undeformed. Numerical face tractions are not cell stresses.

    PNGs show volume-mean sigma_xx (MPa) against volume-mean epsilon_xx, and
    volume-mean alpha against step number. Decreasing-factor segments are marked
    on the stress-strain plot; no sorting by strain or load factor is performed.
    Both plots mark the first recorded state with alpha > 0 in any cell, labeled
    "Yield detected" with its accepted step and load factor. This is a sampled
    detection, not an interpolated exact yield point. If the history starts with
    plastic strain already accumulated, the first recorded state is marked.
    Entirely elastic histories have no yield marker.
    Return paths under pvd, stress_strain, alpha. Re-exporting replaces the PVD
    collection and matching files; unrelated outputs are retained.
    """
    if not records:
        raise ValueError("Plastic history export requires at least one accepted step.")
    if not file_name or Path(file_name).name != file_name or file_name in (".", ".."):
        raise ValueError("file_name must be a nonempty file prefix without directories.")
    nc = grid.num_cells
    volumes = np.asarray(grid.cell_volumes, dtype=np.float64)
    if volumes.shape != (nc,) or not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError("Cell volumes must be finite and positive for history averages.")
    total_volume = float(np.sum(volumes))
    if not np.isfinite(total_volume) or total_volume <= 0:
        raise ValueError("Total cell volume must be finite and positive.")
    weights = volumes / total_volume
    frames: list[list[DataInput]] = []
    factors = np.empty(len(records))
    means = np.empty((len(records), 3))  # epsilon_xx, sigma_xx [Pa], alpha
    first_plastic: int | None = None
    for step, record in enumerate(records, 1):
        state = record.state
        if not np.isfinite(record.load_factor):
            raise ValueError("History load factors must be finite.")
        factors[step - 1] = record.load_factor
        data = _solution_data(grid, _validate_solution(grid, state.x))
        tensors = {"strain": _validate_strain(grid, state.epsilon)}
        if len(state.material_states) != nc:
            raise ValueError("History must contain one material state per cell.")
        for name in ("stress", "plastic_strain", "backstress"):
            values = np.stack([getattr(history, name).to_numpy() for history in state.material_states], axis=2)
            tensors[name] = _validate_tensor(grid, values, name)
        alpha = np.array([history.alpha for history in state.material_states])
        if not np.all(np.isfinite(alpha)) or np.any(alpha < 0):
            raise ValueError("Accumulated plastic strain alpha must be finite and nonnegative.")
        if first_plastic is None and np.any(alpha > 0.0):
            first_plastic = step - 1
        for name, tensor in tensors.items():
            data.extend(_tensor_data(grid, name, tensor))
        data.extend([
            (grid, "alpha", alpha),
            (grid, "load_step", np.full(nc, step, dtype=np.int64)),
            (grid, "load_factor", np.full(nc, record.load_factor)),
        ])
        frames.append(data)
        means[step - 1] = [tensors["strain"][0, 0] @ weights, tensors["stress"][0, 0] @ weights, alpha @ weights]
    if not np.all(np.isfinite(means)):
        raise ValueError("Plastic history averages must be finite.")

    folder = Path(folder_name)
    exporter = pp.Exporter(grid, Path(file_name), folder_name=folder / "vtu")
    for step, data in enumerate(frames, 1):
        exporter.write_vtu(data, time_step=step)
    # A single collection spans all accepted steps; unloading never reverses time.
    exporter.write_pvd(times=np.arange(1, len(records) + 1, dtype=np.float64))
    _publish_pvd_collections(folder, file_name)
    paths = {
        "pvd": folder / "pvd" / f"{file_name}.pvd",
        "stress_strain": folder / f"{file_name}_stress_strain.png",
        "alpha": folder / f"{file_name}_alpha.png",
    }
    _export_plastic_history_png(factors, means, paths, first_plastic)
    return paths


def _export_plastic_history_png(
    factors: NDArray[np.float64], means: NDArray[np.float64], paths: dict[str, Path],
    first_plastic: int | None,
) -> None:
    """Plot accepted cell-volume averages and the first detected plastic state."""
    steps = np.arange(1, len(factors) + 1)
    with plt.rc_context({"font.size": 10, "savefig.dpi": 180}):
        fig, axes = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
        try:
            axes.plot(means[:, 0], means[:, 1] / 1e6, "o-", markersize=3,
                      color="#1565c0", label="Loading steps")
            decreasing = np.flatnonzero(np.diff(factors) < 0)
            groups = np.split(decreasing, np.flatnonzero(np.diff(decreasing) > 1) + 1)
            for index, group in enumerate(groups):
                if group.size:
                    segment = slice(int(group[0]), int(group[-1]) + 2)
                    axes.plot(means[segment, 0], means[segment, 1] / 1e6, "o--",
                              markersize=4, color="#c62828",
                              label="Unloading steps" if index == 0 else "_nolegend_")
            if first_plastic is not None:
                index = first_plastic
                _mark_yield(axes, float(means[index, 0]), float(means[index, 1] / 1e6),
                            index + 1, float(factors[index]))
            axes.set_title("Stress–strain history")
            axes.set_xlabel(r"Mean total strain $\varepsilon_{xx}$ [dimensionless]")
            axes.set_ylabel(r"Mean Cauchy stress $\sigma_{xx}$ [MPa]")
            axes.grid(alpha=0.25)
            axes.legend()
            fig.savefig(paths["stress_strain"])
        finally:
            plt.close(fig)
        fig, axes = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
        try:
            axes.plot(steps, means[:, 2], "o-", color="#1565c0", markersize=3)
            if first_plastic is not None:
                index = first_plastic
                _mark_yield(axes, float(steps[index]), float(means[index, 2]),
                            index + 1, float(factors[index]))
            axes.set_title("Accumulated plastic strain")
            axes.set_xlabel("Accepted step")
            axes.set_ylabel(r"Mean accumulated plastic strain $\alpha$ [dimensionless]")
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            axes.grid(alpha=0.25)
            fig.savefig(paths["alpha"])
        finally:
            plt.close(fig)


def _mark_yield(axes: Axes, x: float, y: float, step: int, factor: float) -> None:
    """Highlight the first recorded plastic state, keeping its label inside the axes."""
    color = "#a86400"
    axes.scatter(x, y, marker="D", s=55, color=color, edgecolors="white", linewidths=0.8, zorder=5)
    # Point toward the plot interior, including when the first/last sample yields.
    left, right = axes.get_xlim()
    # Prefer the earlier part of the path to keep the rising plastic branch visible.
    to_left = x > left + 0.3 * (right - left)
    below = y > sum(axes.get_ylim()) / 2
    axes.annotate(
        f"Yield detected\nStep {step}, load factor {factor:g}",
        xy=(x, y), xytext=(-16 if to_left else 16, -32 if below else 32),
        textcoords="offset points", ha="right" if to_left else "left",
        va="top" if below else "bottom", fontsize=9, color=color,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": color, "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": color}, zorder=6,
    )
