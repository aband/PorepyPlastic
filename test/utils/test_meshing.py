"""Mesh builders: geometry, boundary groups, quality measures and validation."""

from collections.abc import Sequence

import numpy as np
import porepy as pp
import pytest

from utils import meshing
from utils.geometry import preprocess_simplex_grid


def _outward(mesh: meshing.Mesh, faces: np.ndarray) -> np.ndarray:
    """Outward normal dotted with (face center - cell center): positive if outward."""
    _, cells = mesh.grid.signs_and_cells_of_boundary_faces(faces=faces)
    offset = mesh.grid.face_centers[: mesh.dim, faces] - mesh.grid.cell_centers[: mesh.dim, cells]
    return np.asarray(np.sum(mesh.outward_normals(faces) * offset, axis=0))


@pytest.mark.parametrize("kind", ["cartesian", "graded", "perturbed"])
def test_structured_2d_groups_and_geometry(kind: str) -> None:
    mesh = meshing.structured(kind, (6, 4), (3.0, 2.0), origin=(1.0, -1.0))  # type: ignore[arg-type]
    grid = mesh.grid
    assert mesh.dim == 2 and grid.num_cells == 24
    np.testing.assert_allclose(grid.nodes[:2].min(axis=1), [1.0, -1.0])
    np.testing.assert_allclose(grid.nodes[:2].max(axis=1), [4.0, 1.0])
    sizes = {name: faces.size for name, faces in mesh.groups.items()}
    assert sizes == {"xmin": 4, "xmax": 4, "ymin": 6, "ymax": 6}
    for name, axis, value in (("xmin", 0, 1.0), ("xmax", 0, 4.0), ("ymin", 1, -1.0), ("ymax", 1, 1.0)):
        np.testing.assert_allclose(grid.face_centers[axis, mesh.groups[name]], value)
    np.testing.assert_array_equal(mesh.faces("xmin", "xmax", "ymin", "ymax"), np.sort(mesh.boundary))
    assert np.isclose(grid.cell_volumes.sum(), 6.0)
    assert 0.0 < mesh.min_distance <= 1.0
    assert mesh.tpsa_orthogonal is (kind != "perturbed")


@pytest.mark.parametrize("kind", ["cartesian", "graded"])
def test_structured_3d(kind: str) -> None:
    mesh = meshing.structured(kind, (3, 4, 2), (1.0, 2.0, 3.0))  # type: ignore[arg-type]
    assert mesh.dim == 3 and mesh.grid.num_cells == 24
    sizes = {name: faces.size for name, faces in mesh.groups.items()}
    assert sizes == {"xmin": 8, "xmax": 8, "ymin": 6, "ymax": 6, "zmin": 12, "zmax": 12}
    assert np.isclose(mesh.grid.cell_volumes.sum(), 6.0)
    assert mesh.tpsa_orthogonal
    assert np.all(_outward(mesh, mesh.boundary) > 0.0)


def test_graded_is_finer_at_the_ends() -> None:
    x = np.unique(meshing.structured("graded", (10, 1), (1.0, 1.0), grading=0.6).grid.nodes[0])
    widths = np.diff(x)
    assert widths[0] < widths[5] and widths[-1] < widths[5]
    uniform = np.unique(meshing.structured("graded", (10, 1), (1.0, 1.0), grading=0.0).grid.nodes[0])
    np.testing.assert_allclose(np.diff(uniform), 0.1)


def test_perturbed_is_reproducible_and_keeps_boundary() -> None:
    first = meshing.structured("perturbed", (5, 5), (1.0, 1.0), seed=3)
    second = meshing.structured("perturbed", (5, 5), (1.0, 1.0), seed=3)
    other = meshing.structured("perturbed", (5, 5), (1.0, 1.0), seed=4)
    np.testing.assert_array_equal(first.grid.nodes, second.grid.nodes)
    assert not np.array_equal(first.grid.nodes, other.grid.nodes)
    boundary = first.grid.tags["domain_boundary_nodes"]
    regular = pp.CartGrid(np.array([5, 5]), np.array([1.0, 1.0]))
    np.testing.assert_array_equal(first.grid.nodes[:, boundary], regular.nodes[:, boundary])


def test_simplex_rectangle() -> None:
    mesh = meshing.simplex_rectangle((3, 2), (1.5, 1.0))
    assert isinstance(mesh.grid, pp.TriangleGrid)
    assert mesh.grid.num_cells == 14 * 6
    assert mesh.tpsa_orthogonal and mesh.min_distance > 0.0
    assert {name: faces.size for name, faces in mesh.groups.items()} == {
        "xmin": 4, "xmax": 4, "ymin": 6, "ymax": 6,
    }
    assert np.isclose(mesh.grid.cell_volumes.sum(), 1.5)
    with pytest.raises(ValueError, match="TPSA-orthogonal"):
        meshing.simplex_rectangle((2, 2), (3.0, 1.0))           # elongated blocks
    with pytest.raises(ValueError, match="2D"):
        meshing.simplex_rectangle((2, 2, 2), (1.0, 1.0, 1.0))


def test_plate_with_hole() -> None:
    pytest.importorskip("gmsh")
    mesh = meshing.plate_with_hole(radius=1.0, half_width=4.0, level=0)
    grid = mesh.grid
    assert isinstance(grid, pp.TriangleGrid)
    hole = mesh.groups["hole"]
    radius = np.linalg.norm(grid.face_centers[:2, hole], axis=0)
    assert hole.size > 0 and np.all((radius > 0.9) & (radius <= 1.0))
    np.testing.assert_array_equal(mesh.groups["outer"], mesh.faces("xmin", "xmax", "ymin", "ymax"))
    np.testing.assert_array_equal(mesh.faces("outer", "hole"), np.sort(mesh.boundary))
    assert np.isclose(grid.cell_volumes.sum(), 64.0 - np.pi, rtol=0.01)   # polygonal hole
    assert np.all(_outward(mesh, mesh.boundary) > 0.0)
    finer = meshing.plate_with_hole(level=1)
    assert finer.grid.num_cells > 2 * grid.num_cells
    with pytest.raises(ValueError):
        meshing.plate_with_hole(radius=2.0, half_width=4.0)      # fine zone exceeds the plate


def test_from_grid_adds_box_sides_and_custom_groups() -> None:
    grid = pp.CartGrid(np.array([3, 3]), np.array([1.0, 1.0]))
    grid.compute_geometry()
    corner = np.intersect1d(meshing.box_sides(grid)["xmin"], np.where(grid.face_centers[1] < 0.34)[0])
    mesh = meshing.from_grid(grid, groups={"corner": corner}, description="custom")
    assert set(mesh.groups) == {"xmin", "xmax", "ymin", "ymax", "corner"}
    np.testing.assert_array_equal(mesh.faces("corner"), corner)
    assert mesh.description == "custom"
    with pytest.raises(ValueError, match="not on the boundary"):
        meshing.from_grid(grid, groups={"inside": np.array([1])})   # face 1 is interior


def test_from_grid_with_preprocessed_simplices() -> None:
    grid = meshing.simplex_rectangle((2, 2), (1.0, 1.0)).grid
    preprocess_simplex_grid(grid)
    assert meshing.from_grid(grid).tpsa_orthogonal


def test_groups_are_read_only_and_faces_validates_names() -> None:
    mesh = meshing.structured("cartesian", (2, 2), (1.0, 1.0))
    with pytest.raises(TypeError):
        mesh.groups["new"] = np.array([0])  # type: ignore[index]
    with pytest.raises(ValueError):
        mesh.groups["xmin"][0] = 0
    with pytest.raises(KeyError, match="available"):
        mesh.faces("left")
    assert mesh.faces().size == 0
    assert mesh.outward_normals(np.zeros(0, dtype=np.int64)).shape == (2, 0)


def test_tpsa_geometry_detects_nonorthogonality_and_bad_centers() -> None:
    grid = pp.CartGrid(np.array([2, 2]), np.array([1.0, 1.0]))
    grid.compute_geometry()
    distance, skew = meshing.tpsa_geometry(grid)
    assert np.isclose(distance, 1.0) and skew < 1e-12
    grid.cell_centers[0, 0] += 0.1            # off-center: non-orthogonal but valid
    distance, skew = meshing.tpsa_geometry(grid)
    assert 0.0 < distance < 1.0 and skew > 0.1
    grid.cell_centers[0, 0] += 1.0            # outside the cell
    with pytest.raises(ValueError, match="outside"):
        meshing.from_grid(grid)


@pytest.mark.parametrize(("args", "kwargs"), [
    (("hexagonal", (2, 2), (1.0, 1.0)), {}),
    (("cartesian", (2,), (1.0,)), {}),
    (("cartesian", (2, 0), (1.0, 1.0)), {}),
    (("cartesian", (2, True), (1.0, 1.0)), {}),
    (("cartesian", (2, 2), (1.0, -1.0)), {}),
    (("cartesian", (2, 2), (1.0, 1.0, 1.0)), {}),
    (("cartesian", (2, 2), (1.0, 1.0)), {"origin": (0.0,)}),
    (("graded", (2, 2), (1.0, 1.0)), {"grading": 1.0}),
    (("perturbed", (2, 2), (1.0, 1.0)), {"perturbation": 0.25}),
    (("perturbed", (2, 2, 2), (1.0, 1.0, 1.0)), {}),
])
def test_structured_rejects_invalid_input(args: tuple[object, ...], kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        meshing.structured(*args, **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------ planar_domain
def _cook(scale: float = 1.0) -> list[meshing.Line]:
    """Cook's membrane (mm): clamped at x = 0, tapered to the right."""
    p = [(scale * x, scale * y) for x, y in ((0, 0), (48, 44), (48, 60), (0, 44))]
    names = ("bottom", "right", "top", "left")
    return [meshing.Line(p[k], p[(k + 1) % 4], names[k]) for k in range(4)]


def _strip(scale: float = 1.0) -> list[meshing.Line | meshing.Arc]:
    """Quarter of the perforated strip (mm): 100 x 180 minus a radius-50 quarter disk."""
    r, w, h = 50 * scale, 100 * scale, 180 * scale
    return [
        meshing.Line((r, 0), (w, 0), "bottom"), meshing.Line((w, 0), (w, h), "right"),
        meshing.Line((w, h), (0, h), "top"), meshing.Line((0, h), (0, r), "left"),
        meshing.Arc((0, r), (r, 0), (0, 0), "hole"),
    ]


def _check_curves(mesh: meshing.Mesh, curves: Sequence[meshing.Line | meshing.Arc], tol: float) -> None:
    """Every boundary face is in exactly one curve group and both its nodes lie on that curve."""
    grid = mesh.grid
    groups = [mesh.groups[curve.name] for curve in curves if curve.name]
    np.testing.assert_array_equal(np.sort(np.concatenate(list(mesh.groups.values()))), np.sort(mesh.boundary))
    face_nodes = grid.face_nodes.tocsc()
    for curve in curves:
        faces = mesh.groups[curve.name]   # type: ignore[index]
        assert faces.size > 0
        nodes = face_nodes.indices.reshape((2, -1), order="F")[:, faces].ravel()
        assert curve.distance(grid.nodes[:2, nodes]).max() <= tol
    assert len(groups) == len(set(mesh.groups))


@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_planar_domain_cook_membrane_is_acute_in_any_unit(scale: float) -> None:
    pytest.importorskip("gmsh")
    curves = _cook(scale)
    mesh = meshing.planar_domain(curves, size=4.0 * scale)
    assert isinstance(mesh.grid, pp.TriangleGrid)
    assert mesh.tpsa_orthogonal and mesh.capped_cells == 0
    assert np.isclose(mesh.min_distance, 1.0)
    assert set(mesh.groups) == {"bottom", "right", "top", "left"}
    _check_curves(mesh, curves, tol=1e-12 * 60 * scale)
    assert np.isclose(mesh.grid.cell_volumes.sum(), 1440.0 * scale**2)
    for corner in ((48, 60), (48, 44)):                     # loop vertices are nodes (point A of the benchmark)
        assert np.min(np.linalg.norm(mesh.grid.nodes[:2] - scale * np.array(corner)[:, None], axis=0)) == 0.0
    assert np.all(_outward(mesh, mesh.boundary) > 0.0)


def test_planar_domain_strip_with_arc_and_graded_size() -> None:
    pytest.importorskip("gmsh")
    curves = _strip()
    size = lambda x, y: 4.0 + 8.0 * min(1.0, max(0.0, (np.hypot(x, y) - 50.0) / 100.0))  # noqa: E731
    mesh = meshing.planar_domain(curves, size=size)
    assert mesh.tpsa_orthogonal and mesh.capped_cells == 0
    _check_curves(mesh, curves, tol=1e-10)
    exact = 100.0 * 180.0 - np.pi * 50.0**2 / 4
    assert np.isclose(mesh.grid.cell_volumes.sum(), exact, rtol=2e-3)   # polygonal arc
    hole = mesh.groups["hole"]
    near = np.linalg.norm(mesh.grid.face_centers[:2, hole], axis=0) < 60.0
    assert near.all()
    # graded: faces on the hole are much shorter than on the far top side
    assert np.median(mesh.grid.face_areas[hole]) < 0.5 * np.median(mesh.grid.face_areas[mesh.groups["top"]])


def test_planar_domain_holes_shared_names_and_default_names() -> None:
    pytest.importorskip("gmsh")
    outer = [meshing.Line((-4.0, -4.0), (4.0, -4.0)), meshing.Line((4.0, -4.0), (4.0, 4.0), "side"),
             meshing.Line((4.0, 4.0), (-4.0, 4.0)), meshing.Line((-4.0, 4.0), (-4.0, -4.0), "side")]
    corners = [(np.cos(t), np.sin(t)) for t in np.radians([90.0, 210.0, 330.0])]
    hole = [meshing.Arc(corners[k], corners[(k + 1) % 3], (0.0, 0.0), "hole") for k in range(3)]
    mesh = meshing.planar_domain(outer, holes=[hole], size=0.4)
    assert set(mesh.groups) == {"curve0", "side", "curve2", "hole"}
    assert mesh.tpsa_orthogonal and mesh.capped_cells == 0
    radius = np.linalg.norm(mesh.grid.face_centers[:2, mesh.groups["hole"]], axis=0)
    assert np.all((radius > 0.97) & (radius < 1.0))
    assert np.isclose(mesh.grid.cell_volumes.sum(), 64.0 - np.pi, rtol=5e-3)
    assert np.all(_outward(mesh, mesh.boundary) > 0.0)


def test_planar_domain_without_acute_reports_gmsh_quality() -> None:
    pytest.importorskip("gmsh")
    raw = meshing.planar_domain(_strip(), size=5.0, acute=False)
    acute = meshing.planar_domain(_strip(), size=5.0)
    assert raw.grid.num_cells == acute.grid.num_cells
    assert raw.capped_cells > 0 and not raw.tpsa_orthogonal
    assert acute.capped_cells == 0 and acute.tpsa_orthogonal


def test_planar_domain_raises_when_acute_is_unreachable() -> None:
    pytest.importorskip("gmsh")
    square = [meshing.Line((0.0, 0.0), (1.0, 0.0)), meshing.Line((1.0, 0.0), (1.0, 1.0)),
              meshing.Line((1.0, 1.0), (0.0, 1.0)), meshing.Line((0.0, 1.0), (0.0, 0.0))]
    with pytest.raises(ValueError, match="did not reach TPSA orthogonality"):
        meshing.planar_domain(square, size=10.0)            # two triangles: 90-degree corners
    assert meshing.planar_domain(square, size=10.0, acute=False).capped_cells >= 0


@pytest.mark.parametrize(("kwargs", "error", "match"), [
    ({"boundary": [meshing.Line((0, 0), (1, 0)), meshing.Line((1, 0), (1, 1)), meshing.Line((1, 1), (0, 0.5))],
      "size": 0.2}, ValueError, "not closed"),
    ({"boundary": [meshing.Line((0, 0), (1, 0)), meshing.Line((1, 0), (0, 0))], "size": 0.2}, ValueError, "enclose"),
    ({"boundary": [(0, 0), (1, 0), (0, 1)], "size": 0.2}, TypeError, "Line"),
    ({"size": 0.0}, ValueError, "size"),
    ({"size": lambda x, y: -1.0}, ValueError, "size"),
    ({"size": 4.0, "max_angle": 95.0}, ValueError, "max_angle"),
    ({"size": 4.0, "smoothing": -1}, ValueError, "smoothing"),
])
def test_planar_domain_rejects_invalid_input(kwargs: dict[str, object], error: type[Exception], match: str) -> None:
    pytest.importorskip("gmsh")
    kwargs = {"boundary": _cook(), **kwargs}
    with pytest.raises(error, match=match):
        meshing.planar_domain(**kwargs)  # type: ignore[arg-type]
