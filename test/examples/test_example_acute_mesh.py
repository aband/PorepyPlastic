"""examples/example_acute_mesh.py: acute Cook and strip meshes."""

import numpy as np
import pytest

pytest.importorskip("gmsh")

from examples import example_acute_mesh as example  # noqa: E402

MM = example.MM


def _has_node(mesh: object, point: tuple[float, float]) -> bool:
    nodes = mesh.grid.nodes[:2]  # type: ignore[attr-defined]
    return bool(np.min(np.linalg.norm(nodes - np.array(point)[:, None], axis=0)) < 1e-15)


@pytest.mark.parametrize("case", example.CASES)
def test_meshes_are_acute_and_refine(case: str) -> None:
    coarse = example.run_mesh(case, 0, export=False)
    fine = example.build_mesh(case, 1)
    assert fine.grid.num_cells > 3 * coarse.grid.num_cells
    for mesh in (coarse, fine):
        smallest, largest = example.angle_range(mesh)
        assert 0.0 < smallest and largest < 90.0
        assert mesh.capped_cells == 0 and mesh.tpsa_orthogonal
        np.testing.assert_array_equal(np.sort(np.concatenate(list(mesh.groups.values()))), np.sort(mesh.boundary))


def test_cook_geometry_and_groups() -> None:
    mesh = example.build_mesh("cook", 0)
    assert set(mesh.groups) == {"bottom", "load", "top", "clamped"}
    assert np.isclose(mesh.grid.cell_volumes.sum(), 1440.0 * MM**2)
    assert _has_node(mesh, (48.0 * MM, 60.0 * MM))                      # point A
    np.testing.assert_allclose(mesh.grid.face_centers[0, mesh.groups["clamped"]], 0.0, atol=1e-15)
    np.testing.assert_allclose(mesh.grid.face_centers[0, mesh.groups["load"]], 48.0 * MM)


def test_strip_geometry_and_groups() -> None:
    mesh = example.build_mesh("strip", 0)
    assert set(mesh.groups) == {"bottom", "right", "top", "left", "hole"}
    exact = (100.0 * 180.0 - np.pi * 50.0**2 / 4) * MM**2
    assert np.isclose(mesh.grid.cell_volumes.sum(), exact, rtol=5e-3)   # polygonal hole
    assert _has_node(mesh, (0.0, 50.0 * MM)) and _has_node(mesh, (50.0 * MM, 0.0))   # points A, B
    hole = mesh.grid.face_areas[mesh.groups["hole"]]
    top = mesh.grid.face_areas[mesh.groups["top"]]
    assert hole.max() < top.min()                                      # refined toward the hole


def test_invalid_input_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="case"):
        example.build_mesh("square", 0)
    with pytest.raises(ValueError, match="level"):
        example.build_mesh("cook", -1)
    example.main(["--case", "cook", "--levels", "0", "--no_png"])
    assert "cook  level 0" in capsys.readouterr().out
