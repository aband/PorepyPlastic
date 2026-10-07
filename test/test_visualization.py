"""Read back the actual VTK files to check geometry and TPSA field ordering."""

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import meshio  # type: ignore[import-untyped]
import numpy as np
import pytest
from numpy.typing import NDArray

from coupling.plane_strain import PlaneStrainTpsa
from coupling.visualization import export_png, export_strain_png, export_vtk


def test_export_vtk_round_trip(tmp_path: Path) -> None:
    gradient = np.array([[1.0e-4, 2.0e-4], [-1.0e-4, 0.5e-4]])
    translation = np.array([4.0e-5, -3.0e-5])
    case = PlaneStrainTpsa(
        cells_per_axis=3,
        displacement_gradient=gradient,
        translation=translation,
    )
    x = case.solve()
    saved_x = x.copy()
    saved_nodes = case.grid.nodes.copy()
    folder = tmp_path / "nested" / "vtk"
    pvd = export_vtk(case.grid, x, folder_name=folder, file_name="patch")

    assert pvd == folder / "pvd" / "patch.pvd"
    datasets = ET.parse(pvd).findall(".//DataSet")
    assert len(datasets) == 1
    vtu_name = datasets[0].get("file")
    assert vtu_name is not None
    vtu = (pvd.parent / vtu_name).resolve()
    assert vtu == folder / "vtu" / "patch_2.vtu"
    assert not list(folder.glob("*.vtu"))
    assert not list(folder.glob("*.pvd"))
    assert not list((folder / "vtu").glob("*.pvd"))
    mesh = meshio.read(vtu)
    assert "strain" not in mesh.cell_data  # Strain remains optional.

    assert sum(len(block.data) for block in mesh.cells) == case.grid.num_cells
    np.testing.assert_allclose(mesh.points, saved_nodes.T)
    # Compare the fields to physics at the exported cell centers, independently
    # of PorePy's cell ordering or vector flattening convention.
    centers = np.concatenate(
        [mesh.points[block.data].mean(axis=1) for block in mesh.cells], axis=0
    )
    u_ex = np.zeros((case.grid.num_cells, 3))
    u_ex[:, :2] = centers[:, :2] @ gradient.T + translation
    displacement = np.concatenate(mesh.cell_data["displacement"])
    np.testing.assert_allclose(displacement, u_ex, rtol=1.0e-10, atol=1.0e-14)
    np.testing.assert_allclose(
        np.concatenate(mesh.cell_data["displacement_magnitude"]),
        np.linalg.norm(u_ex, axis=1),
        rtol=1.0e-10, atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.concatenate(mesh.cell_data["rotation_stress"]),
        case.material.isotropic_shear_modulus * (gradient[0, 1] - gradient[1, 0]),
        rtol=1.0e-10, atol=1.0e-5,
    )
    np.testing.assert_allclose(
        np.concatenate(mesh.cell_data["total_pressure"]),
        case.material.lame_parameter * np.trace(gradient),
        rtol=1.0e-10, atol=1.0e-5,
    )
    np.testing.assert_array_equal(x, saved_x)
    np.testing.assert_array_equal(case.grid.nodes, saved_nodes)
    assert all(point.trial is None for point in case.material_points)


@pytest.mark.parametrize("output_format", ["vtk", "png"])
@pytest.mark.parametrize(
    "x",
    [
        np.zeros(15),
        np.zeros((4, 4)),
        np.full(16, np.nan),
    ],
    ids=["wrong_length", "not_flat", "nonfinite"],
)
def test_export_rejects_invalid_solution(
    tmp_path: Path, x: NDArray[np.float64], output_format: str
) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.set_geometry()
    folder = tmp_path / "vtk"
    export = export_vtk if output_format == "vtk" else export_png
    with pytest.raises(ValueError):
        export(case.grid, x, folder_name=folder)
    assert not folder.exists()


def test_export_png_preserves_solution_and_existing_figures(tmp_path: Path) -> None:
    case = PlaneStrainTpsa(cells_per_axis=3)
    x = case.solve()
    saved_x = x.copy()
    saved_nodes = case.grid.nodes.copy()
    existing_figure = plt.figure()
    figures_before = plt.get_fignums()
    try:
        path = export_png(
            case.grid, x, folder_name=tmp_path / "png", file_name="patch"
        )
        assert path == tmp_path / "png" / "patch.png"
        assert path.read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a"))
        pixels = plt.imread(path)
        assert pixels.shape[0] > 500 and pixels.shape[1] > 500
        assert np.ptp(pixels[:, :, :3]) > 0.5
        assert plt.get_fignums() == figures_before
        np.testing.assert_array_equal(x, saved_x)
        np.testing.assert_array_equal(case.grid.nodes, saved_nodes)
    finally:
        plt.close(existing_figure)



def test_export_vtk_strain_tensor_round_trip(tmp_path: Path) -> None:
    case = PlaneStrainTpsa(cells_per_axis=3)
    solution = case.solve()

    def field(points: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y = points[:2]
        # Distinct entries and cell-varying values expose component/cell-order errors.
        return 1e-4 * np.array([
            [1 + x, x - y, 2 * x],
            [x - y, -2 - y, -3 * y],
            [2 * x, -3 * y, x + y],
        ])

    epsilon = field(case.grid.cell_centers)
    saved_epsilon, saved_solution = epsilon.copy(), solution.copy()
    saved_nodes = case.grid.nodes.copy()
    pvd = export_vtk(case.grid, solution, epsilon=epsilon, folder_name=tmp_path)
    dataset = ET.parse(pvd).find(".//DataSet")
    assert dataset is not None
    vtu_name = dataset.get("file")
    assert vtu_name is not None
    mesh = meshio.read(pvd.parent / vtu_name)
    centers = np.concatenate([
        mesh.points[block.data].mean(axis=1) for block in mesh.cells
    ])
    expected = field(centers.T)
    # VTK lists the nine entries of each tensor row by row.
    expected_vtk = np.column_stack([
        expected[0, 0], expected[0, 1], expected[0, 2],
        expected[1, 0], expected[1, 1], expected[1, 2],
        expected[2, 0], expected[2, 1], expected[2, 2],
    ])
    np.testing.assert_allclose(
        np.concatenate(mesh.cell_data["strain"]), expected_vtk, rtol=1e-12, atol=1e-18,
    )
    for name, i, j in [("xx", 0, 0), ("yy", 1, 1), ("xy", 0, 1), ("zz", 2, 2)]:
        np.testing.assert_allclose(
            np.concatenate(mesh.cell_data[f"strain_{name}"]), expected[i, j],
            rtol=1e-12, atol=1e-18,
        )
    np.testing.assert_array_equal(epsilon, saved_epsilon)
    np.testing.assert_array_equal(solution, saved_solution)
    np.testing.assert_array_equal(case.grid.nodes, saved_nodes)


@pytest.mark.parametrize("output_format", ["vtk", "png"])
@pytest.mark.parametrize("defect", ["shape", "nan", "infinite", "asymmetric"])
def test_strain_export_rejects_invalid_tensors(
    tmp_path: Path, output_format: str, defect: str,
) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.set_geometry()
    epsilon = np.zeros((3, 3, case.grid.num_cells))
    if defect == "shape":
        epsilon = epsilon[:, :, :-1]
    elif defect == "asymmetric":
        epsilon[0, 1, 0] = 1e-4
    else:
        epsilon[0, 0, 0] = np.nan if defect == "nan" else np.inf
    folder = tmp_path / "invalid"
    with pytest.raises(ValueError, match="strain|Strain"):
        if output_format == "vtk":
            export_vtk(case.grid, np.zeros(16), epsilon=epsilon, folder_name=folder)
        else:
            export_strain_png(case.grid, epsilon, folder_name=folder)
    assert not folder.exists()


@pytest.mark.parametrize("kind", ["varying", "uniform", "shear", "zero"])
def test_strain_png_components_and_principal_glyphs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.set_geometry()
    grid = case.grid
    epsilon = np.zeros((3, 3, grid.num_cells))
    if kind == "varying":
        # Cell 0: extension at +45 degrees, contraction at -45 degrees.
        epsilon[:2, :2, 0] = 1e-4 * np.array([[0.5, 1.5], [1.5, 0.5]])
        epsilon[:2, :2, 1] = np.eye(2) * 3e-4  # No preferred direction.
        epsilon[:2, :2, 2] = np.eye(2) * -2e-4
        # Cell 3 is exactly zero.
    elif kind == "uniform":
        epsilon[0, 0] = 1e-4
    elif kind == "shear":
        epsilon[0, 1] = epsilon[1, 0] = 1e-4
    saved_epsilon, saved_nodes = epsilon.copy(), grid.nodes.copy()
    saved_figures: dict[str, Figure] = {}
    original_savefig = Figure.savefig

    def capture(self: Figure, fname: str | Path, **kwargs: Any) -> None:
        saved_figures[Path(fname).name] = self
        original_savefig(self, fname, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    existing_figure = plt.figure()
    figures_before = plt.get_fignums()
    try:
        paths = export_strain_png(
            grid, epsilon, folder_name=tmp_path / kind, file_name="patch",
        )
        assert set(paths) == {"xx", "yy", "xy", "principal"}
        limit = float(np.max(np.abs(epsilon))) or 1.0
        for name, path in paths.items():
            assert path.parent == tmp_path / kind
            assert path.read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a"))
            pixels = plt.imread(path)
            assert min(pixels.shape[:2]) > 500
            assert np.ptp(pixels[:, :, :3]) > 0.5
            if name != "principal":
                assert path.name == f"patch_epsilon_{name}.png"
                colorbar = saved_figures[path.name].axes[1]
                np.testing.assert_allclose(colorbar.get_ylim(), [-limit, limit])
                assert colorbar.get_ylabel() == "Strain [dimensionless]"
        assert paths["principal"].name == "patch_principal_strain.png"
        axes = saved_figures[paths["principal"].name].axes[0]
        lines = list(axes.lines)
        circles = [patch for patch in axes.patches if isinstance(patch, Circle)]
        if kind == "varying":
            assert len(lines) == 2 and len(circles) == 2
            vectors = []
            for line in lines:
                endpoints = np.asarray(line.get_xydata(), dtype=np.float64)
                np.testing.assert_allclose(
                    endpoints.mean(axis=0), grid.cell_centers[:2, 0],
                )
                vector = endpoints[1] - endpoints[0]
                vectors.append(vector)
                length = np.linalg.norm(vector)
                direction = vector / length
                red, _, blue, _ = to_rgba(line.get_color())
                if red > blue:
                    np.testing.assert_allclose(
                        np.outer(direction, direction), [[0.5, 0.5], [0.5, 0.5]],
                    )
                    assert length == pytest.approx(0.2)
                else:
                    np.testing.assert_allclose(
                        np.outer(direction, direction), [[0.5, -0.5], [-0.5, 0.5]],
                    )
                    assert length == pytest.approx(0.1)
            assert abs(np.dot(vectors[0], vectors[1])) < 1e-14
            for circle, cell, radius in zip(circles, [1, 2], [0.15, 0.1]):
                np.testing.assert_allclose(circle.center, grid.cell_centers[:2, cell])
                assert circle.radius == pytest.approx(radius)
                red, _, blue, _ = to_rgba(circle.get_edgecolor())
                assert (red > blue) == (cell == 1)
        elif kind == "uniform":
            assert len(lines) == 4 and not circles
            for line in lines:
                endpoints = np.asarray(line.get_xydata(), dtype=np.float64)
                assert endpoints[1, 1] == pytest.approx(endpoints[0, 1])
                assert abs(endpoints[1, 0] - endpoints[0, 0]) == pytest.approx(0.3)
        elif kind == "shear":
            assert len(lines) == 8 and not circles
            for line in lines:
                endpoints = np.asarray(line.get_xydata(), dtype=np.float64)
                vector = endpoints[1] - endpoints[0]
                assert np.linalg.norm(vector) == pytest.approx(0.3)
                assert abs(vector[0]) == pytest.approx(abs(vector[1]))
                red, _, blue, _ = to_rgba(line.get_color())
                assert (vector[0] * vector[1] > 0) == (red > blue)
        else:
            assert not lines and not circles
        assert plt.get_fignums() == figures_before
        np.testing.assert_array_equal(epsilon, saved_epsilon)
        np.testing.assert_array_equal(grid.nodes, saved_nodes)
    finally:
        plt.close(existing_figure)
