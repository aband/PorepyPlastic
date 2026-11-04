"""Shared mesh preparation before constructing cached discretization operators."""

import numpy as np
import porepy as pp
from numpy.typing import NDArray

from utils.reconstruction import validate_reconstruction_geometry


def preprocess_simplex_grid(
    grid: pp.Grid,
    *,
    threshold: float = 0.95,
    tol: float | None = None,
    normalize: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Move simplex cell centers using PorePy's circumcenter preprocessing.

    Call after ``grid.compute_geometry()`` and before creating material histories,
    TPSA matrices, or retained-face operators. Only cell-center coordinates change;
    nodes, face geometry, cell volumes, and connectivity are preserved. Calling
    ``compute_geometry()`` afterwards would reset the centers.

    Return PorePy's ``(centers, shift, changed)``. ``shift`` is the fraction of
    the barycenter-to-circumcenter movement. It is one for a full move, smaller
    for a capped move, and zero if the two centers already coincide. ``changed``
    compares the proposed centers with the grid's previous centers.

    PorePy's ``tol`` is absolute and serves two checks with opposite scaling: a
    degeneracy test on the squared cell measure (~ h^(2 dim)) and an
    equidistance test limited by round-off in the coordinates (~ eps * |x|).
    With the default ``normalize=False`` (and ``tol=1e-14``) PorePy runs on the
    grid as given, so millimetre coordinates of order 100 can fail the
    equidistance check and fine meshes in metres the degeneracy check. With
    ``normalize=True`` the computation runs on a copy translated to the bounding
    box center and scaled by the typical cell size (median cell measure to the
    power 1 / dim); ``tol`` is then relative to that size and defaults to
    max(1e-12, 1e-14 * domain extent / cell size), just above round-off (PorePy
    also keeps barycenters that are within tol of the circumcenter). The
    centers agree with the unnormalized ones to round-off.

    Capped moves keep centers inside cells but generally leave nonorthogonality.
    This is not a TPSA admissibility certificate. In 3D, moving cell centers also
    does not align general tetrahedral face centroids with the circumcenters.

    Example::

        grid.compute_geometry()
        centers, shift, changed = preprocess_simplex_grid(grid)
        layout = DofLayout.from_grid(grid, bc)
        operators = RetainedFaceOperators(grid, layout, mu=mu)
        elastic = ElasticTpsa(grid, bc, mu=mu, lam=lam)
    """
    if not isinstance(grid, (pp.TriangleGrid, pp.TetrahedralGrid)):
        raise ValueError("Circumcenter preprocessing requires a simplex grid.")
    validate_reconstruction_geometry(grid)

    origin = np.zeros(3)
    scale = 1.0
    if normalize:
        origin = 0.5 * (grid.nodes.min(axis=1) + grid.nodes.max(axis=1))
        scale = float(np.median(grid.cell_volumes)) ** (1.0 / grid.dim)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Cannot normalize a grid with zero or invalid cell volumes.")
    if tol is None:
        # Round-off in normalized coordinates grows with domain size / cell size.
        reach = float(np.max(np.abs(grid.nodes - origin[:, None]))) / scale
        tol = max(1e-12, 1e-14 * reach) if normalize else 1e-14

    simplex: pp.TriangleGrid | pp.TetrahedralGrid = grid
    if normalize or type(grid) not in (pp.TriangleGrid, pp.TetrahedralGrid):
        # The local PorePy helper checks exact types, excluding structured
        # subclasses. Use a temporary base grid with the same cell ordering
        # (and, if normalizing, translated and scaled nodes).
        simplex = _simplex_copy(grid, (grid.nodes - origin[:, None]) / scale)
        # Preserve the meaning of PorePy's changed mask on repeated calls.
        simplex.cell_centers = (grid.cell_centers - origin[:, None]) / scale

    centers, shift, changed = pp.grid_utils.compute_circumcenters(
        simplex, threshold=threshold, tol=tol
    )
    if normalize:
        centers = origin[:, None] + scale * centers
    grid.cell_centers = centers
    return centers, shift, changed


def _simplex_copy(
    grid: pp.Grid, nodes: NDArray[np.float64]
) -> pp.TriangleGrid | pp.TetrahedralGrid:
    """Plain TriangleGrid/TetrahedralGrid with grid's cell ordering on the given nodes."""
    cells = grid.cell_nodes().tocsc().indices.reshape(
        (grid.dim + 1, grid.num_cells), order="F"
    ).copy()
    simplex: pp.TriangleGrid | pp.TetrahedralGrid
    if grid.dim == 2:
        first = nodes[:2, cells[1]] - nodes[:2, cells[0]]
        second = nodes[:2, cells[2]] - nodes[:2, cells[0]]
        clockwise = first[0] * second[1] - first[1] * second[0] < 0
        cells[:, clockwise] = cells[[0, 2, 1]][:, clockwise]
        simplex = pp.TriangleGrid(nodes, cells)
    else:
        simplex = pp.TetrahedralGrid(nodes, cells)
    simplex.compute_geometry()
    return simplex
