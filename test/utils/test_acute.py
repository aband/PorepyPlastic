"""Acute triangulation: curves, degree-raising flips and angle optimization (no gmsh)."""

import numpy as np
import pytest

from utils.acute import (
    FIXED,
    INTERIOR,
    Arc,
    Line,
    counterclockwise,
    deficient_nodes,
    make_acute,
    raise_low_degree,
    required_triangles,
    triangle_angles,
)


def _union_jack(m: int = 2, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Line]]:
    """[0, 2m]^2 split into right isosceles triangles with alternating diagonals.

    Nodes with i + j odd have 4 triangles (interior) or 2 (on a side): every
    one of them forces a 90-degree angle.
    """
    n = 2 * m + 1
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    points = scale * np.vstack((i.ravel(), j.ravel())).astype(float)
    def index(a: int, b: int) -> int:
        return a * n + b

    triangles = []
    for a in range(n - 1):
        for b in range(n - 1):
            p, q, r, s = index(a, b), index(a + 1, b), index(a + 1, b + 1), index(a, b + 1)
            triangles += [[p, q, r], [p, r, s]] if (a + b) % 2 == 0 else [[p, q, s], [q, r, s]]
    triangles_array = np.array(triangles).T
    side = 2 * m * scale
    corners = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)]
    curves = [Line(corners[k], corners[(k + 1) % 4], name) for k, name in enumerate(("ymin", "xmax", "ymax", "xmin"))]
    node_curve = np.full(n * n, INTERIOR)
    x, y = points
    on = np.isclose
    node_curve[on(y, 0)] = 0
    node_curve[on(x, side)] = 1
    node_curve[on(y, side)] = 2
    node_curve[on(x, 0)] = 3
    node_curve[(on(x, 0) | on(x, side)) & (on(y, 0) | on(y, side))] = FIXED
    return points, counterclockwise(points, triangles_array), node_curve, curves


def test_required_triangles_and_deficient_nodes() -> None:
    points, triangles, node_curve, _ = _union_jack()
    required = required_triangles(points, triangles)
    assert np.all(required[node_curve == INTERIOR] == 5)
    assert np.all(required[node_curve >= 0] == 3)
    assert np.all(required[node_curve == FIXED] == 2)
    i, j = points
    odd = (i + j) % 2 == 1
    np.testing.assert_array_equal(deficient_nodes(points, triangles), np.flatnonzero(odd))
    assert np.allclose(np.abs(triangle_angles(points, triangles)).max(), 90.0)


def _hexagon(radius: int = 3, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Line]]:
    """Regular hexagonal patch of equilateral triangles, then two flips that leave
    the center node with 4 triangles (two 120-degree angles there)."""
    from scipy.spatial import Delaunay

    a, b = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="ij")
    keep = np.abs(a + b) <= radius
    a, b = a[keep], b[keep]
    points = scale * np.vstack((a + 0.5 * b, np.sqrt(3) / 2 * b))
    triangles = counterclockwise(points, Delaunay(points.T).simplices.T.astype(np.int64))
    corners = [(scale * radius * np.cos(t), scale * radius * np.sin(t)) for t in np.radians(np.arange(0, 360, 60))]
    curves = [Line(corners[k], corners[(k + 1) % 6]) for k in range(6)]
    node_curve = np.full(points.shape[1], INTERIOR)
    for k, curve in enumerate(curves):
        node_curve[curve.distance(points) < 1e-9 * scale] = k
    for corner in corners:
        node_curve[np.linalg.norm(points - np.array(corner)[:, None], axis=0) < 1e-9 * scale] = FIXED

    center = int(np.argmin(np.linalg.norm(points, axis=0)))
    for direction in (0.0, np.pi):                      # flip the edges to (1, 0) and (-1, 0)
        target = scale * np.array([np.cos(direction), np.sin(direction)])
        neighbour = int(np.argmin(np.linalg.norm(points - target[:, None], axis=0)))
        pair = np.flatnonzero(np.isin(triangles, [center, neighbour]).sum(axis=0) == 2)
        t, s = pair
        others = [int(x) for x in np.concatenate([triangles[:, t], triangles[:, s]]) if x not in (center, neighbour)]
        triangles[:, t], triangles[:, s] = [center, others[0], others[1]], [neighbour, others[1], others[0]]
    return points, counterclockwise(points, triangles), node_curve, curves


def _area(points: np.ndarray, triangles: np.ndarray) -> float:
    u = points[:, triangles[1]] - points[:, triangles[0]]
    v = points[:, triangles[2]] - points[:, triangles[0]]
    return float(0.5 * np.sum(u[0] * v[1] - u[1] * v[0]))


def test_hexagon_fixture_has_one_deficient_node() -> None:
    points, triangles, node_curve, _ = _hexagon()
    center = int(np.argmin(np.linalg.norm(points, axis=0)))
    np.testing.assert_array_equal(deficient_nodes(points, triangles), [center])
    assert np.isclose(np.abs(triangle_angles(points, triangles)).max(), 120.0)
    required = required_triangles(points, triangles)
    assert np.all(required[node_curve == INTERIOR] == 5)
    assert np.all(required[node_curve >= 0] == 3)
    assert np.all(required[node_curve == FIXED] == 2)   # 120-degree corners


def test_flips_remove_forced_obtuse_angles() -> None:
    points, triangles, _, _ = _hexagon()
    flipped, flips = raise_low_degree(points, triangles)
    assert flips == 1
    assert deficient_nodes(points, flipped).size == 0
    angles = triangle_angles(points, flipped)
    assert angles.min() > 0.0                              # counterclockwise, none inverted
    assert np.isclose(_area(points, flipped), _area(points, triangles))


def test_union_jack_side_deficits_cannot_all_be_flipped_away() -> None:
    """Each side has 2m nodes with 2 triangles but only 2m - 1 neighbours that
    can spare one: flips alone cannot help, and make_acute reports it."""
    points, triangles, node_curve, curves = _union_jack()
    i, j = points
    np.testing.assert_array_equal(deficient_nodes(points, triangles), np.flatnonzero((i + j) % 2 == 1))
    result = make_acute(points, triangles, node_curve, curves)
    assert result.deficient_after > 0
    assert result.max_angle >= 90.0 - 1e-6


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e3])
def test_make_acute_reaches_target_and_keeps_boundary(scale: float) -> None:
    points, triangles, node_curve, curves = _hexagon(scale=scale)
    result = make_acute(points, triangles, node_curve, curves, max_angle=80.0, min_angle=20.0)
    assert (result.deficient_before, result.deficient_after, result.flips) == (1, 0, 1)
    assert result.max_angle <= 80.0 + 1e-6 and result.min_angle > 20.0 - 1e-6
    angles = triangle_angles(result.points, result.triangles)
    assert np.isclose(angles.max(), result.max_angle)
    moved = result.points
    np.testing.assert_array_equal(moved[:, node_curve == FIXED], points[:, node_curve == FIXED])
    for k, curve in enumerate(curves):
        assert curve.distance(moved[:, node_curve == k]).max() <= 1e-12 * scale


def test_make_acute_is_independent_of_the_length_unit() -> None:
    small = make_acute(*_hexagon(scale=1e-3), max_angle=65.0)
    large = make_acute(*_hexagon(scale=1e3), max_angle=65.0)
    np.testing.assert_array_equal(small.triangles, large.triangles)
    np.testing.assert_allclose(small.points * 1e6, large.points, rtol=0, atol=1e-8 * 3e3)
    assert small.iterations == large.iterations


def test_arc_nodes_slide_on_the_circle() -> None:
    """Fan of triangles under an arc: the free arc node must stay on the circle."""
    radius = 2.0
    angles = np.radians([30.0, 60.0, 150.0])
    points = np.column_stack([(0.0, 0.0), *[(radius * np.cos(t), radius * np.sin(t)) for t in angles]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]]).T
    curves = [Arc(tuple(points[:, 1]), tuple(points[:, 3]), (0.0, 0.0))]
    node_curve = np.array([FIXED, FIXED, 0, FIXED])
    result = make_acute(points, triangles, node_curve, curves, max_angle=80.0)
    moved = result.points[:, 2]
    assert np.isclose(np.linalg.norm(moved), radius)
    assert 30.0 < np.degrees(np.arctan2(moved[1], moved[0])) < 150.0
    assert result.max_angle <= 80.0 + 1e-6                  # the centre angle was 90 degrees


def test_line_and_arc_parametrization() -> None:
    line = Line((1.0, 2.0), (4.0, 6.0))
    s = np.array([0.0, 0.25, 1.0])
    p, tangent = line.evaluate(s)
    np.testing.assert_allclose(line.parameter(p), s)
    np.testing.assert_allclose(tangent, [[3.0] * 3, [4.0] * 3])
    assert line.distance(np.array([[1.0], [7.0]]))[0] == pytest.approx(3.0)   # nearest point (3.4, 5.2)

    # Arc across the +-pi branch cut, clockwise: from 170 to -170 degrees the short way.
    start, end = (2 * np.cos(np.radians(170)), 2 * np.sin(np.radians(170))), (2 * np.cos(np.radians(-170)), 2 * np.sin(np.radians(-170)))
    arc = Arc(start, end, (0.0, 0.0), "hole")
    assert np.isclose(arc.radius, 2.0) and np.isclose(abs(arc.sweep), np.radians(20))
    lo, hi = arc.bounds
    s = np.linspace(lo, hi, 5)
    p, tangent = arc.evaluate(s)
    np.testing.assert_allclose(np.linalg.norm(p, axis=0), 2.0)
    np.testing.assert_allclose(arc.parameter(p), s, atol=1e-14)
    np.testing.assert_allclose(p[:, 0], start)
    np.testing.assert_allclose(p[:, -1], end)
    assert np.all(p[0] < 0)                                         # through 180 degrees, not 0
    h = 1e-7
    numeric = (arc.evaluate(s + h)[0] - arc.evaluate(s - h)[0]) / (2 * h)
    np.testing.assert_allclose(tangent, numeric, atol=1e-6)
    assert arc.distance(np.array([[3.0 * np.cos(np.pi)], [0.0]]))[0] == pytest.approx(1.0)


@pytest.mark.parametrize(("build", "match"), [
    (lambda: Line((0.0, 0.0), (0.0, 0.0)), "coincide"),
    (lambda: Line((0.0, np.nan), (1.0, 0.0)), "finite"),
    (lambda: Arc((1.0, 0.0), (0.0, 2.0), (0.0, 0.0)), "same positive distance"),
    (lambda: Arc((1.0, 0.0), (-1.0, 0.0), (0.0, 0.0)), "180"),
    (lambda: Arc((1.0, 0.0), (1.0, 0.0), (0.0, 0.0)), "180"),
])
def test_curve_validation(build: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build()  # type: ignore[operator]


def test_make_acute_rejects_invalid_input() -> None:
    points, triangles, node_curve, curves = _union_jack()
    with pytest.raises(ValueError, match="max_angle"):
        make_acute(points, triangles, node_curve, curves, max_angle=90.0)
    with pytest.raises(ValueError, match="node_curve"):
        make_acute(points, triangles, node_curve[:-1], curves)
    with pytest.raises(ValueError, match="node_curve"):
        make_acute(points, triangles, np.where(node_curve == 3, 4, node_curve), curves)
    with pytest.raises(ValueError, match="points"):
        make_acute(points[:1], triangles, node_curve, curves)
    with pytest.raises(ValueError, match="triangles"):
        make_acute(points, triangles + points.shape[1], node_curve, curves)
