"""Read back plastic history files and check tensor, time, and averaging conventions."""

from copy import deepcopy
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import meshio  # type: ignore[import-untyped]
import numpy as np
import pytest
from porepy.grids.grid import Grid
from porepy.grids.structured import TensorGrid

from coupling.loading import LoadStepResult
from coupling.newton import NewtonResult
from coupling.residual import TpsaResidual, TpsaState
from coupling.visualization import export_plastic_history
from tensor import strain, stress


@pytest.fixture
def history() -> tuple[Grid, list[LoadStepResult]]:
    # Manufactured snapshots are an export oracle, not an equilibrium test.
    # Unequal cell volumes and distinct tensor entries expose ordering/averaging bugs.
    grid = TensorGrid(np.array([0.0, 0.25, 1.0]), np.array([0.0, 1.0]))
    grid.compute_geometry()
    records = []
    for index, factor in enumerate([0.25, 1.0, 0.8], 1):
        state = TpsaState.zeros(grid)
        state.x[:] = np.arange(1, 9) * factor
        for cell, material in enumerate(state.material_states):
            value = float(index + 10 * cell)
            full = np.array([[value, 2, 3], [2, -value, 4], [3, 4, 5 * value]])
            state.epsilon[:2, :2, cell] = full[:2, :2] * factor * 1e-4
            material.stress = stress(full * factor * 1e6)
            material.plastic_strain = strain(full * 1e-5)
            material.backstress = stress(full * 1e3)
            material.alpha = value * 1e-3
        evaluation = TpsaResidual(np.zeros(8), state, np.zeros((3, 3, 2)))
        records.append(LoadStepResult(factor, np.zeros(8), NewtonResult(
            evaluation, 0, np.zeros((1, 3)), np.ones(3),
        )))
    return grid, records


def test_history_vtk_round_trip_and_input_ownership(
    history: tuple[Grid, list[LoadStepResult]], tmp_path: Path,
) -> None:
    grid, records = history
    originals, nodes = deepcopy(records), grid.nodes.copy()
    paths = export_plastic_history(grid, records, folder_name=tmp_path)
    datasets = ET.parse(paths["pvd"]).findall(".//DataSet")
    assert paths["pvd"].parent == tmp_path / "pvd"
    assert not list(tmp_path.glob("*.vtu"))
    assert not list(tmp_path.glob("*.pvd"))
    assert not list((tmp_path / "vtu").glob("*.pvd"))
    # Every PorePy collection, including the individual-step PVDs, stays usable.
    collections = list((tmp_path / "pvd").glob("*.pvd"))
    assert len(collections) == len(records) + 1
    for collection in collections:
        for dataset in ET.parse(collection).findall(".//DataSet"):
            reference = Path(dataset.attrib["file"])
            assert reference.parent == Path("../vtu")
            assert (collection.parent / reference).is_file()
    assert [float(data.attrib["timestep"]) for data in datasets] == [1, 2, 3]
    for step, (dataset, record, original) in enumerate(zip(datasets, records, originals, strict=True), 1):
        mesh = meshio.read(paths["pvd"].parent / dataset.attrib["file"])
        np.testing.assert_array_equal(mesh.points, nodes.T)
        centers = np.concatenate([mesh.points[block.data].mean(axis=1) for block in mesh.cells])
        cells = np.argmin(np.linalg.norm(centers[:, None, :] - grid.cell_centers.T[None, :, :], axis=2), axis=1)
        assert sorted(cells) == list(range(grid.num_cells))
        data = {name: np.concatenate(blocks) for name, blocks in mesh.cell_data.items()}
        expected_u = np.column_stack((record.state.x[:4].reshape((2, 2)), np.zeros(2)))
        np.testing.assert_allclose(data["displacement"], expected_u[cells])
        np.testing.assert_allclose(data["displacement_magnitude"], np.linalg.norm(expected_u[cells], axis=1))
        np.testing.assert_array_equal(data["rotation_stress"], record.state.x[4:6][cells])
        np.testing.assert_array_equal(data["total_pressure"], record.state.x[6:][cells])
        np.testing.assert_array_equal(data["load_step"], step)
        np.testing.assert_array_equal(data["load_factor"], record.load_factor)
        np.testing.assert_array_equal(data["alpha"], [record.state.material_states[cell].alpha for cell in cells])
        for cell_row, cell in enumerate(cells):
            for name in ("strain", "stress", "plastic_strain", "backstress"):
                tensor = record.state.epsilon[:, :, cell] if name == "strain" else getattr(record.state.material_states[cell], name).to_numpy()
                # Explicit row/column order rather than reproducing the export reshape.
                expected = [tensor[i, j] for i in range(3) for j in range(3)]
                np.testing.assert_array_equal(data[name][cell_row], expected)
                for component, i, j in [("xx", 0, 0), ("yy", 1, 1), ("xy", 0, 1), ("zz", 2, 2)]:
                    assert data[f"{name}_{component}"][cell_row] == tensor[i, j]
        for name in ("x", "epsilon", "bc_values", "traction"):
            np.testing.assert_array_equal(getattr(record.state, name), getattr(original.state, name))
        for actual, saved in zip(record.state.material_states, original.state.material_states, strict=True):
            assert actual.alpha == saved.alpha
            for name in ("stress", "plastic_strain", "backstress"):
                np.testing.assert_array_equal(getattr(actual, name).to_numpy(), getattr(saved, name).to_numpy())
    np.testing.assert_array_equal(grid.nodes, nodes)


def test_history_png_uses_volume_means_and_preserves_unloading_order(
    history: tuple[Grid, list[LoadStepResult]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid, records = history
    captured: dict[str, Figure] = {}
    original_savefig = Figure.savefig

    def capture(self: Figure, fname: str | Path, **kwargs: Any) -> None:
        captured[Path(fname).name] = self
        original_savefig(self, fname, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    existing = plt.figure()
    figures_before = plt.get_fignums()
    try:
        paths = export_plastic_history(grid, records, folder_name=tmp_path)
        # Cell areas are 1/4 and 3/4. Distinct fields make arithmetic averaging fail.
        mean_strain, mean_stress, mean_alpha = [], [], []
        for record in records:
            a, b = record.state.material_states
            mean_strain.append(0.25 * record.state.epsilon[0, 0, 0] + 0.75 * record.state.epsilon[0, 0, 1])
            mean_stress.append((0.25 * a.stress.to_numpy()[0, 0] + 0.75 * b.stress.to_numpy()[0, 0]) / 1e6)
            mean_alpha.append(0.25 * a.alpha + 0.75 * b.alpha)
        stress_axes = captured[paths["stress_strain"].name].axes[0]
        np.testing.assert_allclose(np.asarray(stress_axes.lines[0].get_xdata(), dtype=np.float64), mean_strain)
        np.testing.assert_allclose(np.asarray(stress_axes.lines[0].get_ydata(), dtype=np.float64), mean_stress)
        assert "MPa" in stress_axes.get_ylabel()
        assert len(stress_axes.lines) == 2
        np.testing.assert_allclose(np.asarray(stress_axes.lines[1].get_xdata(), dtype=np.float64), mean_strain[1:])
        np.testing.assert_allclose(np.asarray(stress_axes.lines[1].get_ydata(), dtype=np.float64), mean_stress[1:])
        assert mean_strain[2] < mean_strain[1]  # The path must double back, not be sorted.
        alpha_axes = captured[paths["alpha"].name].axes[0]
        np.testing.assert_array_equal(np.asarray(alpha_axes.lines[0].get_xdata(), dtype=np.float64), [1, 2, 3])
        np.testing.assert_allclose(np.asarray(alpha_axes.lines[0].get_ydata(), dtype=np.float64), mean_alpha)
        for name in ("stress_strain", "alpha"):
            assert paths[name].read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a"))
            pixels = plt.imread(paths[name])
            assert min(pixels.shape[:2]) > 500
            assert np.ptp(pixels[:, :, :3]) > 0.5
        assert plt.get_fignums() == figures_before
    finally:
        plt.close(existing)


@pytest.mark.parametrize("invalid", ["empty", "factor", "x", "strain", "stress", "alpha", "count", "volumes"])
def test_bad_late_frame_is_rejected_before_creating_files(
    history: tuple[Grid, list[LoadStepResult]], tmp_path: Path, invalid: str,
) -> None:
    grid, records = history
    last = records[-1]
    if invalid == "empty":
        records = []
    elif invalid == "factor":
        last.load_factor = np.nan
    elif invalid == "x":
        last.state.x[0] = np.inf
    elif invalid == "strain":
        last.state.epsilon[0, 1, 0] += 1  # Asymmetry.
    elif invalid == "stress":
        last.state.material_states[0].stress = stress(np.full((3, 3), np.nan))
    elif invalid == "alpha":
        last.state.material_states[0].alpha = -1.0
    elif invalid == "count":
        last.state.material_states.pop()
    elif invalid == "volumes":
        grid.cell_volumes[0] = 0.0
    folder = tmp_path / "not_created"
    with pytest.raises(ValueError):
        export_plastic_history(grid, records, folder_name=folder)
    assert not folder.exists()


def test_reexport_replaces_collection_and_preserves_other_results(
    history: tuple[Grid, list[LoadStepResult]], tmp_path: Path,
) -> None:
    grid, records = history
    unrelated = tmp_path / "pvd" / "displacement.pvd"
    unrelated.parent.mkdir()
    unrelated.write_text("previous elastic output")
    export_plastic_history(grid, records, folder_name=tmp_path)
    paths = export_plastic_history(grid, records[:1], folder_name=tmp_path)
    datasets = ET.parse(paths["pvd"]).findall(".//DataSet")
    assert len(datasets) == 1 and float(datasets[0].attrib["timestep"]) == 1
    assert unrelated.read_text() == "previous elastic output"
