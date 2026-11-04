"""Mesh builders for core runs: a grid plus named boundary groups and TPSA quality.

Every builder returns a Mesh. Boundary faces are grouped by name, so run
scripts select boundaries without coordinate tests:

    mesh = structured("perturbed", cells=(32, 32), size=(2.0, 1.0))
    mesh.faces("xmin", "xmax")          # Dirichlet faces, say
    mesh.groups                         # {"xmin": ..., "xmax": ..., "ymin": ..., "ymax": ...}

Builders:
    structured        2D/3D "cartesian" or "graded"; 2D "perturbed"
    simplex_rectangle 2D triangles from a 12-node block pattern (TPSA-orthogonal)
    plate_with_hole   2D square plate minus a circular hole (gmsh), groups + "hole"
    planar_domain     2D domain bounded by Line/Arc curves (gmsh), groups = curve
                      names; strictly acute (TPSA-orthogonal) by default
    from_grid         wrap any computed PorePy grid; box sides found automatically

Side groups are "xmin", "xmax", "ymin", "ymax" (and "zmin", "zmax" in 3D):
boundary faces lying on the corresponding side of the bounding box.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import porepy as pp

from core.layout import FloatArray, IntArray
from utils.acute import FIXED, INTERIOR, Arc, Curve, Line, counterclockwise, make_acute
from utils.geometry import preprocess_simplex_grid

StructuredKind = Literal["cartesian", "graded", "perturbed"]
_STRUCTURED = ("cartesian", "graded", "perturbed")
_SIDES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


@dataclass(eq=False)
class Mesh:
    """A computed grid, its named boundary-face groups and TPSA geometry measures.

    min_distance:        smallest cell-center-to-face distance along the outward
                         normal, relative to the cell-to-face-center offset (> 0
                         is required; small values mean nearly degenerate cells).
    max_nonorthogonality: largest tangential part of a cell-to-face-center offset,
                         relative to the offset length (0 on orthogonal meshes).
    capped_cells:        simplex cells whose circumcenter move was capped.
    """

    grid: pp.Grid
    groups: Mapping[str, IntArray]
    min_distance: float
    max_nonorthogonality: float
    capped_cells: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        boundary = set(self.grid.get_all_boundary_faces().tolist())
        frozen: dict[str, IntArray] = {}
        for name, faces in self.groups.items():
            values = np.unique(np.asarray(faces, dtype=np.int64))
            if not set(values.tolist()) <= boundary:
                raise ValueError(f"Group {name!r} contains faces that are not on the boundary.")
            values.flags.writeable = False
            frozen[name] = values
        self.groups = MappingProxyType(frozen)

    @property
    def dim(self) -> int:
        return int(self.grid.dim)

    @property
    def boundary(self) -> IntArray:
        faces: IntArray = np.asarray(self.grid.get_all_boundary_faces(), dtype=np.int64)
        return faces

    def faces(self, *names: str) -> IntArray:
        """Union of the named groups (sorted, unique)."""
        unknown = [name for name in names if name not in self.groups]
        if unknown:
            raise KeyError(f"Unknown boundary group(s) {unknown}; available: {sorted(self.groups)}.")
        if not names:
            return np.zeros(0, dtype=np.int64)
        union: IntArray = np.unique(np.concatenate([self.groups[name] for name in names]))
        return union

    def outward_normals(self, faces: IntArray) -> FloatArray:
        """Outward unit normals (dim, n) of boundary faces."""
        faces = np.asarray(faces, dtype=np.int64)
        if faces.size == 0:
            return np.zeros((self.dim, 0))
        signs, _ = self.grid.signs_and_cells_of_boundary_faces(faces=faces)
        normals: FloatArray = (
            self.grid.face_normals[: self.dim, faces] * signs[None, :] / self.grid.face_areas[faces]
        )
        return normals

    @property
    def tpsa_orthogonal(self) -> bool:
        return self.max_nonorthogonality <= 1e-10

    def summary(self) -> str:
        sizes = ", ".join(f"{name} {faces.size}" for name, faces in self.groups.items())
        text = (
            f"{self.description or type(self.grid).__name__}: {self.dim}D, {self.grid.num_cells} cells, "
            f"{self.grid.num_faces} faces; min distance {self.min_distance:.2e}, "
            f"max nonorthogonality {self.max_nonorthogonality:.2e}"
        )
        if self.capped_cells:
            text += f", capped cells {self.capped_cells}"
        return f"{text}\n  boundary groups: {sizes}"


# ----------------------------------------------------------------------------- quality
def tpsa_geometry(grid: pp.Grid) -> tuple[float, float]:
    """(min relative normal distance, max relative nonorthogonality) over all cell sides."""
    dim = grid.dim
    incidence = grid.cell_faces.tocoo()
    faces, cells = incidence.row, incidence.col
    normals = grid.face_normals[:dim, faces] * incidence.data[None, :] / grid.face_areas[faces][None, :]
    offset = grid.face_centers[:dim, faces] - grid.cell_centers[:dim, cells]
    length = np.linalg.norm(offset, axis=0)
    distance = np.sum(offset * normals, axis=0)
    tangential = np.linalg.norm(offset - normals * distance[None, :], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_distance = distance / length
        skew = tangential / length
    if not (np.all(np.isfinite(relative_distance)) and np.all(np.isfinite(skew))):
        raise ValueError("Invalid cell-to-face geometry (zero offset or non-finite values).")
    return float(relative_distance.min()), float(skew.max())


def _finish(
    grid: pp.Grid,
    groups: Mapping[str, IntArray],
    *,
    description: str,
    capped_cells: int = 0,
    require_orthogonal: bool = False,
) -> Mesh:
    min_distance, nonorthogonality = tpsa_geometry(grid)
    if min_distance <= 0.0:
        raise ValueError(
            f"{description}: a cell center lies on or outside one of its faces "
            f"(min relative distance {min_distance:.2e})."
        )
    if require_orthogonal and nonorthogonality > 1e-10:
        raise ValueError(
            f"{description}: mesh is not TPSA-orthogonal (max nonorthogonality "
            f"{nonorthogonality:.2e}); choose cells so the block aspect ratio is close to one."
        )
    return Mesh(grid, groups, min_distance, nonorthogonality, capped_cells, description)


def box_sides(grid: pp.Grid, *, rtol: float = 1e-10) -> dict[str, IntArray]:
    """Boundary faces on each side of the bounding box: xmin, xmax, ymin, ..."""
    boundary = np.asarray(grid.get_all_boundary_faces(), dtype=np.int64)
    centers = grid.face_centers[: grid.dim, boundary]
    lower, upper = grid.nodes[: grid.dim].min(axis=1), grid.nodes[: grid.dim].max(axis=1)
    tolerance = rtol * float(np.max(upper - lower))
    groups: dict[str, IntArray] = {}
    for axis in range(grid.dim):
        for name, value in ((_SIDES[2 * axis], lower[axis]), (_SIDES[2 * axis + 1], upper[axis])):
            # Face centers on the side plane; faces merely touching it have centers off it.
            groups[name] = boundary[np.abs(centers[axis] - value) <= tolerance]
    return groups


# ---------------------------------------------------------------------------- builders
def _positive_ints(cells: Sequence[int]) -> tuple[int, ...]:
    if len(cells) not in (2, 3) or any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in cells
    ):
        raise ValueError("cells must contain 2 or 3 positive integers.")
    return tuple(int(n) for n in cells)


def _box(cells: Sequence[int], size: Sequence[float], origin: Sequence[float] | None) -> tuple[
    tuple[int, ...], FloatArray, FloatArray,
]:
    counts = _positive_ints(cells)
    lengths = np.asarray(size, dtype=np.float64)
    start = np.zeros(len(counts)) if origin is None else np.asarray(origin, dtype=np.float64)
    if lengths.shape != (len(counts),) or start.shape != (len(counts),):
        raise ValueError("size and origin must have one entry per axis of cells.")
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0) or not np.all(np.isfinite(start)):
        raise ValueError("size must be finite and positive, origin finite.")
    return counts, lengths, start


def structured(
    kind: StructuredKind,
    cells: Sequence[int],
    size: Sequence[float],
    *,
    origin: Sequence[float] | None = None,
    grading: float = 0.6,
    perturbation: float = 0.15,
    seed: int = 0,
) -> Mesh:
    """Quadrilateral (2D) or hexahedral (3D) mesh of a box.

    kind:
        "cartesian"  uniform cells.
        "graded"     smooth mapping x = s - grading * sin(2 pi s) / (2 pi) per axis:
                     finer near both ends; grading in [0, 1), 0 = uniform.
        "perturbed"  2D only: interior nodes moved randomly by up to
                     perturbation * cell size per axis; perturbation in [0, 0.25)
                     keeps the quadrilaterals convex. (3D perturbation would make
                     faces non-planar.)
    """
    if kind not in _STRUCTURED:
        raise ValueError(f"kind must be one of {_STRUCTURED}; got {kind!r}.")
    counts, lengths, start = _box(cells, size, origin)
    dim = len(counts)

    if kind == "graded":
        if not np.isfinite(grading) or not 0.0 <= grading < 1.0:
            raise ValueError("grading must lie in [0, 1).")
        axes = []
        for n, length, offset in zip(counts, lengths, start):
            s = np.linspace(0.0, 1.0, n + 1)
            mapped = s - grading * np.sin(2.0 * np.pi * s) / (2.0 * np.pi)
            mapped[0], mapped[-1] = 0.0, 1.0
            axes.append(offset + length * mapped)
        grid: pp.Grid = pp.TensorGrid(*axes)
    else:
        grid = pp.CartGrid(np.array(counts), physdims=lengths)
        grid.nodes[:dim] += start[:, None]
        if kind == "perturbed":
            if dim != 2:
                raise ValueError("perturbed meshes are 2D only (3D faces would become non-planar).")
            if not np.isfinite(perturbation) or not 0.0 <= perturbation < 0.25:
                raise ValueError("perturbation must lie in [0, 0.25).")
            interior = ~grid.tags["domain_boundary_nodes"]
            spacing = (lengths / np.array(counts))[:, None]
            shifts = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(2, int(interior.sum())))
            grid.nodes[:2, interior] += perturbation * spacing * shifts

    grid.compute_geometry()
    label = f"{kind} {'x'.join(map(str, counts))}"
    return _finish(grid, box_sides(grid), description=label)


def simplex_rectangle(
    cells: Sequence[int],
    size: Sequence[float],
    *,
    origin: Sequence[float] | None = None,
) -> Mesh:
    """Triangles from a 14-triangle pattern per block, circumcenter-preprocessed.

    cells = (nx, ny) blocks; the mesh has 14 nx ny triangles. The pattern is
    TPSA-orthogonal only for nearly square blocks (size[0]/nx close to
    size[1]/ny); otherwise a ValueError is raised.
    """
    counts, lengths, start = _box(cells, size, origin)
    if len(counts) != 2:
        raise ValueError("simplex_rectangle is 2D: cells = (nx, ny).")
    nx, ny = counts

    # Local coordinates on a 20 x 20 block: corners, edge midpoints, interior nodes.
    vertices = np.array([
        [0, 0], [20, 0], [20, 20], [0, 20],
        [10, 0], [20, 10], [10, 20], [0, 10],
        [6, 6], [13, 6], [14, 14], [7, 14],
    ], dtype=np.int64).T
    pattern = []
    for corner in range(4):
        pattern.extend([
            [corner, 4 + corner, 8 + corner],
            [corner, 8 + corner, 4 + (corner - 1) % 4],
            [4 + corner, 8 + (corner + 1) % 4, 8 + corner],
        ])
    pattern.extend([[8, 9, 11], [9, 10, 11]])
    triangles = np.array(pattern, dtype=np.int64).T

    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    offsets = 20 * np.vstack((ix.ravel(), iy.ravel()))
    nodes = (vertices[:, :, None] + offsets[:, None, :]).reshape((2, -1), order="F")
    nodes, node_map = np.unique(nodes, axis=1, return_inverse=True)   # merge shared nodes
    connectivity = (triangles[:, :, None] + 12 * np.arange(nx * ny)[None, None, :]).reshape((3, -1), order="F")
    points = start[:, None] + nodes * (lengths / (20 * np.array([nx, ny])))[:, None]

    grid = pp.TriangleGrid(points, node_map.ravel()[connectivity])
    grid.compute_geometry()
    _, shift, _ = preprocess_simplex_grid(grid)
    capped = int(np.count_nonzero((shift > 0) & (shift < 1 - 1e-12)))
    return _finish(
        grid, box_sides(grid), description=f"simplex {nx}x{ny} blocks",
        capped_cells=capped, require_orthogonal=True,
    )


def plate_with_hole(
    *,
    radius: float = 1.0,
    half_width: float = 4.0,
    level: int = 0,
    near_size: float | None = None,
    far_factor: float = 4.0,
    fine_radius_factor: float = 2.5,
) -> Mesh:
    """Triangulated square [-half_width, half_width]^2 minus a centered disk (gmsh).

    Cell size is near_size (default radius / 2 / 2**level) within
    fine_radius_factor * radius of the center and grows linearly to
    far_factor * near_size at the outer boundary. Groups: the four sides,
    "outer" (all four) and "hole". Circumcenter preprocessing is applied; the
    result is reported in max_nonorthogonality and capped_cells.
    """
    import gmsh  # type: ignore[import-untyped]

    if isinstance(level, bool) or not isinstance(level, (int, np.integer)) or level < 0:
        raise ValueError("level must be a nonnegative integer.")
    if not np.all(np.isfinite([radius, half_width, far_factor, fine_radius_factor])):
        raise ValueError("Geometry parameters must be finite.")
    if not 0 < fine_radius_factor * radius < half_width or radius <= 0:
        raise ValueError("Require 0 < radius and fine_radius_factor * radius < half_width.")
    if far_factor < 1:
        raise ValueError("far_factor must be at least one.")
    if gmsh.isInitialized():
        raise RuntimeError("plate_with_hole needs an inactive gmsh session.")

    h_near = radius / (2 * 2 ** int(level)) if near_size is None else float(near_size)
    if not np.isfinite(h_near) or h_near <= 0:
        raise ValueError("near_size must be finite and positive.")
    h_far = far_factor * h_near
    fine_radius = fine_radius_factor * radius

    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("plate_with_hole")
        occ = gmsh.model.occ
        square = occ.addRectangle(-half_width, -half_width, 0, 2 * half_width, 2 * half_width)
        disk = occ.addDisk(0, 0, 0, radius, radius)
        surfaces, _ = occ.cut([(2, square)], [(2, disk)])
        occ.synchronize()

        size_field = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            size_field, "F",
            f"{h_near} + {h_far - h_near} * "
            f"Min(1, Max(0, (Sqrt(x*x+y*y)-{fine_radius})/{half_width - fine_radius}))",
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(size_field)
        for option in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary", "RecombineAll"):
            gmsh.option.setNumber(f"Mesh.{option}", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay
        gmsh.model.mesh.generate(2)

        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        _, connectivity = gmsh.model.mesh.getElementsByType(2, surfaces[0][1])   # 3-node triangles
        order = np.argsort(tags)   # gmsh tags need not be contiguous or sorted
        points = coordinates.reshape(-1, 3)[order, :2].T
        triangles = np.searchsorted(tags[order], connectivity).reshape(-1, 3).T
    finally:
        gmsh.finalize()

    # PorePy expects counterclockwise triangles.
    a = points[:, triangles[1]] - points[:, triangles[0]]
    b = points[:, triangles[2]] - points[:, triangles[0]]
    clockwise = a[0] * b[1] - a[1] * b[0] < 0
    triangles[:, clockwise] = triangles[[0, 2, 1]][:, clockwise]

    grid = pp.TriangleGrid(points, triangles)
    grid.compute_geometry()
    _, shift, _ = preprocess_simplex_grid(grid)

    groups = box_sides(grid)
    outer = np.unique(np.concatenate([groups[name] for name in ("xmin", "xmax", "ymin", "ymax")]))
    groups["outer"] = outer
    groups["hole"] = np.setdiff1d(grid.get_all_boundary_faces(), outer)
    capped = int(np.count_nonzero((shift > 0) & (shift < 1 - 1e-12)))
    return _finish(grid, groups, description=f"plate with hole, level {level}", capped_cells=capped)


def planar_domain(
    boundary: Sequence[Curve],
    *,
    size: float | Callable[[float, float], float],
    holes: Sequence[Sequence[Curve]] = (),
    acute: bool = True,
    max_angle: float = 80.0,
    min_angle: float = 20.0,
    smoothing: int = 10,
    description: str = "",
) -> Mesh:
    """Triangulate a 2D domain bounded by Line and Arc curves (gmsh), optionally acute.

    boundary is the outer loop and each entry of holes an inner loop: curves in
    order, each ending where the next one starts (closing back to the first).
    Arcs must sweep less than 180 degrees, so a circular hole takes at least
    three arcs. size is the target cell size, a number or a function h(x, y).

    Groups are the curve names: curves with the same name share a group, and
    unnamed curves get "curve<k>" with k counting over all loops in order.
    Every curve endpoint is a mesh node.

    With acute=True (default) the triangulation is made strictly acute
    (utils.acute.make_acute: edge flips, then nodes moved with boundary nodes
    sliding along their curves and loop vertices fixed), aiming at angles in
    [min_angle, max_angle]. A ValueError is raised if the circumcenter mesh is
    then not TPSA-orthogonal (some angle too close to 90 degrees); a finer
    size or acute=False are the alternatives. With acute=False the gmsh mesh
    is used as it is, and capped_cells / max_nonorthogonality report its
    quality.

    Circumcenters are computed in normalized coordinates, so the geometry can
    be given in any length unit.
    """
    import gmsh

    loops = [list(boundary), *(list(hole) for hole in holes)]
    curves: list[Curve] = [curve for loop in loops for curve in loop]
    if any(not isinstance(curve, (Line, Arc)) for curve in curves):
        raise TypeError("Loops must contain utils.acute.Line and utils.acute.Arc curves.")
    if any(len(loop) < 2 or (len(loop) == 2 and all(isinstance(c, Line) for c in loop)) for loop in loops):
        raise ValueError("Each loop needs at least two curves and must enclose an area.")
    coordinates = np.array([curve.start for curve in curves] + [curve.end for curve in curves], dtype=np.float64)
    extent = float(np.max(coordinates.max(axis=0) - coordinates.min(axis=0)))
    for loop in loops:
        for curve, following in zip(loop, loop[1:] + loop[:1]):
            if not np.allclose(curve.end, following.start, rtol=0.0, atol=1e-9 * extent):
                raise ValueError(f"Loop is not closed: {curve} does not end where {following} starts.")
    names = [curve.name or f"curve{k}" for k, curve in enumerate(curves)]
    if not callable(size) and (not np.isfinite(size) or size <= 0):
        raise ValueError("size must be positive and finite, or a function h(x, y).")
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, np.integer)) or smoothing < 0:
        raise ValueError("smoothing must be a nonnegative integer.")
    if acute and (not 60.0 < max_angle < 90.0 or not 0.0 <= min_angle < 60.0):
        raise ValueError("Require 60 < max_angle < 90 and 0 <= min_angle < 60 (degrees).")
    if gmsh.isInitialized():
        raise RuntimeError("planar_domain needs an inactive gmsh session.")

    size_errors: list[float] = []

    def size_callback(dim: int, tag: int, x: float, y: float, z: float, lc: float) -> float:
        h = float(size(x, y)) if callable(size) else float(size)
        if not np.isfinite(h) or h <= 0.0:
            size_errors.append(h)
            return extent   # coarsest sensible size, so gmsh finishes quickly
        return h

    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("planar_domain")
        geo = gmsh.model.geo
        curve_tags: list[int] = []
        point_tags: list[int] = []
        loop_tags: list[int] = []
        for loop in loops:
            starts = [geo.addPoint(*curve.start, 0.0) for curve in loop]
            point_tags += starts
            tags = []
            for k, curve in enumerate(loop):
                begin, finish = starts[k], starts[(k + 1) % len(loop)]
                if isinstance(curve, Line):
                    tags.append(geo.addLine(begin, finish))
                else:
                    center = geo.addPoint(*curve.center, 0.0)
                    tags.append(geo.addCircleArc(begin, center, finish))
            curve_tags += tags
            loop_tags.append(geo.addCurveLoop(tags))
        surface = geo.addPlaneSurface(loop_tags)
        geo.synchronize()

        for option in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary", "RecombineAll"):
            gmsh.option.setNumber(f"Mesh.{option}", 0)
        gmsh.model.mesh.setSizeCallback(size_callback)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Smoothing", int(smoothing))
        gmsh.model.mesh.generate(2)
        if size_errors:
            raise ValueError(f"size(x, y) must be positive and finite; got {size_errors[0]!r}.")

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)   # gmsh tags need not be contiguous or sorted
        sorted_tags = tags[order]
        points = xyz.reshape(-1, 3)[order, :2].T
        _, connectivity = gmsh.model.mesh.getElementsByType(2, surface)   # 3-node triangles
        triangles = np.searchsorted(sorted_tags, connectivity).reshape(-1, 3).T
        node_curve = np.full(points.shape[1], INTERIOR, dtype=np.int64)
        for k, tag in enumerate(curve_tags):
            on_curve, _, _ = gmsh.model.mesh.getNodes(1, tag, includeBoundary=False)
            node_curve[np.searchsorted(sorted_tags, on_curve)] = k
        for tag in point_tags:
            corner, _, _ = gmsh.model.mesh.getNodes(0, tag)
            node_curve[np.searchsorted(sorted_tags, corner)] = FIXED
    finally:
        gmsh.finalize()

    # Arc centers are geometry points, not mesh nodes: drop unused nodes.
    used = np.unique(triangles)
    renumber = np.full(points.shape[1], -1, dtype=np.int64)
    renumber[used] = np.arange(used.size)
    points, triangles, node_curve = points[:, used], renumber[triangles], node_curve[used]
    triangles = counterclockwise(points, triangles)

    label = description or f"planar domain, {len(curves)} curves"
    if acute:
        result = make_acute(points, triangles, node_curve, curves, max_angle=max_angle, min_angle=min_angle)
        points, triangles = result.points, result.triangles
        label = description or f"{label}, acute (max angle {result.max_angle:.1f} deg)"

    grid = pp.TriangleGrid(points, triangles)
    grid.compute_geometry()
    _, shift, _ = preprocess_simplex_grid(grid, normalize=True)
    capped = int(np.count_nonzero((shift > 0) & (shift < 1 - 1e-12)))
    groups = _curve_groups(grid, node_curve, curves, names)
    mesh = _finish(grid, groups, description=label, capped_cells=capped)
    if acute and (capped or not mesh.tpsa_orthogonal):
        raise ValueError(
            f"{label}: acute meshing did not reach TPSA orthogonality (largest angle "
            f"{result.max_angle:.2f} deg, {capped} capped cells); try a finer size or acute=False."
        )
    return mesh


def _curve_groups(
    grid: pp.Grid, node_curve: IntArray, curves: Sequence[Curve], names: Sequence[str],
) -> dict[str, IntArray]:
    """Boundary faces per curve name, from the curve of each face's nodes."""
    boundary = np.asarray(grid.get_all_boundary_faces(), dtype=np.int64)
    face_nodes = grid.face_nodes.tocsc()
    ends = face_nodes.indices.reshape((2, -1), order="F")[:, boundary]
    first, second = node_curve[ends[0]], node_curve[ends[1]]
    owner = np.where(first >= 0, first, second)
    # Faces joining two loop vertices: a curve meshed as a single edge. Match the
    # face to the curve with these endpoints (up to orientation).
    corners = np.flatnonzero(owner < 0)
    if corners.size:
        nodes = grid.nodes[:2]
        for face in corners:
            p, q = nodes[:, ends[0, face]], nodes[:, ends[1, face]]
            matches = [
                k for k, curve in enumerate(curves)
                if {tuple(np.round(curve.start, 12)), tuple(np.round(curve.end, 12))}
                == {tuple(np.round(p, 12)), tuple(np.round(q, 12))}
            ]
            if len(matches) != 1:
                raise ValueError("Could not assign a boundary face to a unique curve.")
            owner[face] = matches[0]
    groups: dict[str, list[IntArray]] = {}
    for k, name in enumerate(names):
        groups.setdefault(name, []).append(boundary[owner == k])
    return {name: np.unique(np.concatenate(parts)) for name, parts in groups.items()}


def from_grid(
    grid: pp.Grid,
    *,
    groups: Mapping[str, IntArray] | None = None,
    description: str = "",
) -> Mesh:
    """Wrap a computed PorePy grid (call compute_geometry, and for simplices
    preprocess_simplex_grid, first). Box-side groups are added automatically;
    `groups` adds or overrides named face sets."""
    named = box_sides(grid)
    named.update({} if groups is None else dict(groups))
    return _finish(grid, named, description=description or type(grid).__name__)


__all__ = [
    "Arc",
    "Line",
    "Mesh",
    "box_sides",
    "from_grid",
    "planar_domain",
    "plate_with_hole",
    "simplex_rectangle",
    "structured",
    "tpsa_geometry",
]
