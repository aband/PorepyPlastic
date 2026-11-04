"""Make 2D triangulations strictly acute, so circumcenters give TPSA-orthogonal cells.

A triangle's circumcenter lies strictly inside it, and the segment from the
circumcenter to each edge midpoint is normal to that edge, exactly when all
three angles are below 90 degrees. make_acute moves the nodes of a given
triangulation toward that state in two steps:

1. Edge flips. A node whose angle sum is alpha needs more than alpha / 90
   triangles, or one of its angles is at least 90 degrees whatever the node
   positions: 5 for interior nodes, 3 on a straight side, 2 in a 90-degree
   corner. Each deficient node gets one triangle more per flip of an edge
   opposite to it.
2. Node optimization (L-BFGS-B, analytic gradients) of the squared hinge
   penalty on signed angles outside [min_angle, max_angle]. Interior nodes move
   freely, boundary nodes slide along their Line or Arc, and corners stay fixed.
   Inverted triangles have negative angles and are penalized.

The result is not guaranteed; callers check the angles (or the TPSA geometry).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from core.layout import FloatArray, IntArray

INTERIOR = -1
FIXED = -2


def _point(value: Sequence[float], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 2D point.")
    return array


@dataclass(frozen=True)
class Line:
    """Straight boundary segment from start to end; parameter s in [0, 1]."""

    start: tuple[float, float]
    end: tuple[float, float]
    name: str | None = None

    def __post_init__(self) -> None:
        a, b = _point(self.start, "Line start"), _point(self.end, "Line end")
        if np.allclose(a, b, rtol=0.0, atol=1e-12 * max(1.0, float(np.abs(a).max()))):
            raise ValueError("Line start and end coincide.")

    @property
    def bounds(self) -> tuple[float, float]:
        return 0.0, 1.0

    def parameter(self, points: FloatArray) -> FloatArray:
        a, d = np.asarray(self.start, float), np.subtract(self.end, self.start)
        s: FloatArray = (d @ (points - a[:, None])) / float(d @ d)
        return s

    def evaluate(self, s: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Points and d(point)/ds, both (2, n)."""
        a, d = np.asarray(self.start, float), np.subtract(self.end, self.start)
        return a[:, None] + d[:, None] * s[None, :], np.repeat(d[:, None], s.size, axis=1)

    def distance(self, points: FloatArray) -> FloatArray:
        s = np.clip(self.parameter(points), 0.0, 1.0)
        nearest, _ = self.evaluate(s)
        result: FloatArray = np.linalg.norm(points - nearest, axis=0)
        return result


@dataclass(frozen=True)
class Arc:
    """Circular boundary arc from start to end about center, the short way round.

    The sweep must be below 180 degrees (gmsh's limit); split longer arcs and
    full circles into several arcs. Parameter s in [0, |sweep|] radians.
    """

    start: tuple[float, float]
    end: tuple[float, float]
    center: tuple[float, float]
    name: str | None = None
    radius: float = field(init=False, repr=False)
    phase: float = field(init=False, repr=False)
    sweep: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        a = _point(self.start, "Arc start")
        b = _point(self.end, "Arc end")
        c = _point(self.center, "Arc center")
        radius = float(np.linalg.norm(a - c))
        if radius <= 0.0 or not np.isclose(np.linalg.norm(b - c), radius, rtol=1e-9, atol=0.0):
            raise ValueError("Arc start and end must be at the same positive distance from center.")
        phase = float(np.arctan2(a[1] - c[1], a[0] - c[0]))
        sweep = float(np.angle(np.exp(1j * (np.arctan2(b[1] - c[1], b[0] - c[0]) - phase))))
        if not 1e-9 < abs(sweep) < np.pi - 1e-9:
            raise ValueError("Arc sweep must lie strictly between 0 and 180 degrees; split longer arcs.")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "sweep", sweep)

    @property
    def bounds(self) -> tuple[float, float]:
        return 0.0, abs(self.sweep)

    def parameter(self, points: FloatArray) -> FloatArray:
        c = np.asarray(self.center, float)[:, None]
        angle = np.arctan2(points[1] - c[1], points[0] - c[0])
        s: FloatArray = np.mod(np.sign(self.sweep) * (angle - self.phase), 2.0 * np.pi)
        # Points just before the start wrap to ~2 pi; map them back near zero.
        s[s > np.pi] -= 2.0 * np.pi
        return s

    def evaluate(self, s: FloatArray) -> tuple[FloatArray, FloatArray]:
        sign = np.sign(self.sweep)
        angle = self.phase + sign * s
        c = np.asarray(self.center, float)[:, None]
        unit = np.vstack((np.cos(angle), np.sin(angle)))
        tangent = sign * self.radius * np.vstack((-np.sin(angle), np.cos(angle)))
        return c + self.radius * unit, tangent

    def distance(self, points: FloatArray) -> FloatArray:
        s = np.clip(self.parameter(points), *self.bounds)
        nearest, _ = self.evaluate(s)
        result: FloatArray = np.linalg.norm(points - nearest, axis=0)
        return result


Curve = Line | Arc


# ----------------------------------------------------------------------------- angles
def counterclockwise(points: FloatArray, triangles: IntArray) -> IntArray:
    """Copy of triangles with every triangle ordered counterclockwise."""
    triangles = np.array(triangles, dtype=np.int64, copy=True)
    a = points[:, triangles[1]] - points[:, triangles[0]]
    b = points[:, triangles[2]] - points[:, triangles[0]]
    clockwise = a[0] * b[1] - a[1] * b[0] < 0
    triangles[:, clockwise] = triangles[[0, 2, 1]][:, clockwise]
    return triangles


def triangle_angles(points: FloatArray, triangles: IntArray) -> FloatArray:
    """(3, nt) signed angles in degrees; angle k is at node triangles[k].

    Positive for counterclockwise triangles, negative for inverted ones.
    """
    angles = np.empty(triangles.shape, dtype=np.float64)
    for k in range(3):
        p = points[:, triangles[k]]
        u = points[:, triangles[(k + 1) % 3]] - p
        v = points[:, triangles[(k + 2) % 3]] - p
        angles[k] = np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], np.sum(u * v, axis=0)))
    return angles


def required_triangles(points: FloatArray, triangles: IntArray) -> IntArray:
    """Fewest triangles at each node that allow all its angles below 90 degrees."""
    angle_sum = np.bincount(
        triangles.ravel(), np.abs(triangle_angles(points, triangles)).ravel(), minlength=points.shape[1],
    )
    required: IntArray = np.floor(angle_sum / 90.0 + 1e-6).astype(np.int64) + 1
    return required


def deficient_nodes(points: FloatArray, triangles: IntArray) -> IntArray:
    """Nodes with fewer triangles than required_triangles: an angle >= 90 is forced."""
    degree = np.bincount(triangles.ravel(), minlength=points.shape[1])
    nodes: IntArray = np.flatnonzero(degree < required_triangles(points, triangles))
    return nodes


# ------------------------------------------------------------------------------ flips
def _neighbours(triangles: IntArray, num_nodes: int) -> IntArray:
    """(3, nt): triangle across the edge opposite local node k, or -1 on the boundary."""
    nt = triangles.shape[1]
    first = np.concatenate([triangles[(k + 1) % 3] for k in range(3)])
    second = np.concatenate([triangles[(k + 2) % 3] for k in range(3)])
    key = np.minimum(first, second) * num_nodes + np.maximum(first, second)
    order = np.argsort(key, kind="stable")
    same = np.flatnonzero(key[order][1:] == key[order][:-1])
    neighbour = np.full(3 * nt, -1, dtype=np.int64)
    neighbour[order[same]] = order[same + 1] % nt
    neighbour[order[same + 1]] = order[same] % nt
    return neighbour.reshape(3, nt)


def raise_low_degree(points: FloatArray, triangles: IntArray, *, max_passes: int = 10) -> tuple[IntArray, int]:
    """Flip edges opposite deficient nodes; return (triangles, number of flips).

    A flip of edge (a, b) in triangles (v, a, b) and (b, a, w) gives (v, a, w)
    and (v, w, b): v and w gain a triangle, a and b lose one. A flip is used
    only if the quadrilateral is strictly convex and a and b keep their
    required count; among candidates the one with the smallest largest angle
    wins. Triangles must be counterclockwise.
    """
    triangles = np.array(triangles, dtype=np.int64, copy=True)
    n = points.shape[1]
    need = required_triangles(points, triangles)   # angle sums do not change under flips
    flips = 0
    for _ in range(max_passes):
        degree = np.bincount(triangles.ravel(), minlength=n)
        bad = np.flatnonzero(degree < need)
        if bad.size == 0:
            break
        neighbour = _neighbours(triangles, n)
        flipped = False
        for v in map(int, bad):
            if degree[v] >= need[v]:
                continue
            best: tuple[float, int, int, IntArray, IntArray] | None = None
            for t in map(int, np.flatnonzero((triangles == v).any(axis=0))):
                k = int(np.flatnonzero(triangles[:, t] == v)[0])
                a, b = triangles[(k + 1) % 3, t], triangles[(k + 2) % 3, t]
                s = int(neighbour[k, t])
                if s < 0 or degree[a] - 1 < need[a] or degree[b] - 1 < need[b]:
                    continue
                w = next(int(x) for x in triangles[:, s] if x != a and x != b)
                candidate = np.array([[v, v], [a, w], [w, b]], dtype=np.int64)
                angles = triangle_angles(points, candidate)
                if angles.min() <= 0.0:   # nonconvex quadrilateral
                    continue
                score = float(angles.max())
                if best is None or score < best[0]:
                    best = (score, t, s, candidate[:, 0], candidate[:, 1])
            if best is not None:
                _, t, s, first, second = best
                triangles[:, t], triangles[:, s] = first, second
                flips += 1
                flipped = True
                degree = np.bincount(triangles.ravel(), minlength=n)
                neighbour = _neighbours(triangles, n)
        if not flipped:
            break
    return triangles, flips


# ----------------------------------------------------------------------- optimization
@dataclass(frozen=True)
class AcuteResult:
    points: FloatArray
    triangles: IntArray
    max_angle: float           # largest angle reached, degrees
    min_angle: float           # smallest angle reached, degrees
    flips: int
    deficient_before: int      # nodes with a forced angle >= 90 before / after the flips
    deficient_after: int
    iterations: int


def make_acute(
    points: FloatArray,
    triangles: IntArray,
    node_curve: IntArray,
    curves: Sequence[Curve],
    *,
    max_angle: float = 80.0,
    min_angle: float = 20.0,
    max_iterations: int = 5000,
) -> AcuteResult:
    """Flip and move nodes so that all angles approach [min_angle, max_angle].

    points (2, n); triangles (3, nt); node_curve (n,): INTERIOR (-1) for free
    nodes, FIXED (-2) for corners, or the index into curves of the curve a
    boundary node slides on. Connectivity changes only by flips; the node
    count and numbering are kept.
    """
    points = np.asarray(points, dtype=np.float64)
    node_curve = np.asarray(node_curve, dtype=np.int64)
    n = points.shape[1]
    if points.ndim != 2 or points.shape[0] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite (2, n) array.")
    if node_curve.shape != (n,) or np.any(node_curve < FIXED) or np.any(node_curve >= len(curves)):
        raise ValueError("node_curve must hold INTERIOR, FIXED or a curve index for every node.")
    if not 60.0 < max_angle < 90.0 or not 0.0 <= min_angle < 60.0:
        raise ValueError("Require 60 < max_angle < 90 and 0 <= min_angle < 60 (degrees).")
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[0] != 3 or triangles.min() < 0 or triangles.max() >= n:
        raise ValueError("triangles must be a (3, nt) array of node indices.")
    triangles = counterclockwise(points, triangles)

    before = deficient_nodes(points, triangles).size
    triangles, flips = raise_low_degree(points, triangles)
    after = deficient_nodes(points, triangles).size

    interior = np.flatnonzero(node_curve == INTERIOR)
    sliding = np.flatnonzero(node_curve >= 0)
    groups = [(curve, sliding[node_curve[sliding] == k]) for k, curve in enumerate(curves)]
    groups = [(curve, nodes) for curve, nodes in groups if nodes.size]

    # Optimize in units of the typical edge length, so convergence does not depend
    # on the length unit or on the curve parametrizations (radians, [0, 1]).
    edges = np.hstack([triangles[[0, 1]], triangles[[1, 2]], triangles[[2, 0]]])
    unit = float(np.median(np.linalg.norm(points[:, edges[0]] - points[:, edges[1]], axis=0)))
    origin = points.mean(axis=1, keepdims=True)
    factors = [np.full(2 * interior.size, unit)]
    starts = [((points[:, interior] - origin).ravel())]
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * (2 * interior.size)
    for curve, nodes in groups:
        lo, hi = curve.bounds
        length = float(np.linalg.norm(curve.evaluate(np.array([lo]))[1]))   # |d point / ds|, constant
        factor = unit / length
        margin = 1e-9 * (hi - lo)
        factors.append(np.full(nodes.size, factor))
        starts.append(np.clip(curve.parameter(points[:, nodes]), lo + margin, hi - margin))
        bounds += [((lo + margin) / factor, (hi - margin) / factor)] * nodes.size
    factor_all = np.concatenate(factors)
    upper, lower = np.radians(max_angle), np.radians(min_angle)

    def unpack(z: FloatArray) -> tuple[FloatArray, list[FloatArray]]:
        physical = z * factor_all
        p = points.copy()
        p[:, interior] = origin + physical[: 2 * interior.size].reshape(2, -1)
        offset, tangents = 2 * interior.size, []
        for curve, nodes in groups:
            p[:, nodes], tangent = curve.evaluate(physical[offset: offset + nodes.size])
            tangents.append(tangent)
            offset += nodes.size
        return p, tangents

    def objective(z: FloatArray) -> tuple[float, FloatArray]:
        p, tangents = unpack(z)
        gradient = np.zeros_like(p)
        value = 0.0
        for k in range(3):
            iv, iq, ir = triangles[k], triangles[(k + 1) % 3], triangles[(k + 2) % 3]
            u, v = p[:, iq] - p[:, iv], p[:, ir] - p[:, iv]
            cross = u[0] * v[1] - u[1] * v[0]
            dot = np.sum(u * v, axis=0)
            theta = np.arctan2(cross, dot)
            excess = np.maximum(theta - upper, 0.0) - np.maximum(lower - theta, 0.0)
            value += float(np.sum(excess ** 2))
            weight = 2.0 * excess / (cross ** 2 + dot ** 2)
            du = weight * (dot * np.vstack((v[1], -v[0])) - cross * v)
            dv = weight * (dot * np.vstack((-u[1], u[0])) - cross * u)
            for index, g in ((iq, du), (ir, dv), (iv, -(du + dv))):
                gradient[0] += np.bincount(index, g[0], minlength=n)
                gradient[1] += np.bincount(index, g[1], minlength=n)
        parts = [gradient[:, interior].ravel()]
        parts += [np.sum(gradient[:, nodes] * t, axis=0) for (_, nodes), t in zip(groups, tangents)]
        return value, np.concatenate(parts) * factor_all

    z0 = np.concatenate(starts) / factor_all
    result = minimize(
        objective, z0, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iterations, "ftol": 1e-16, "gtol": 1e-14},
    )
    moved, _ = unpack(result.x)
    angles = triangle_angles(moved, triangles)
    return AcuteResult(
        points=moved, triangles=triangles, max_angle=float(angles.max()), min_angle=float(angles.min()),
        flips=flips, deficient_before=int(before), deficient_after=int(after), iterations=int(result.nit),
    )


__all__ = [
    "AcuteResult",
    "Arc",
    "Curve",
    "FIXED",
    "INTERIOR",
    "Line",
    "counterclockwise",
    "deficient_nodes",
    "make_acute",
    "raise_low_degree",
    "required_triangles",
    "triangle_angles",
]
