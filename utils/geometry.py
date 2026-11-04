"""Shared mesh preparation before constructing cached discretization operators."""

import numpy as np
import porepy as pp
from numpy.typing import NDArray

from utils.reconstruction import validate_reconstruction_geometry


def preprocess_simplex_grid(
    grid: pp.Grid,
    *,
    threshold: float = 0.95,
    tol: float = 1e-14,
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

    simplex = grid
    if type(grid) not in (pp.TriangleGrid, pp.TetrahedralGrid):
        # The local PorePy helper checks exact types, excluding structured
        # subclasses. Use a temporary base grid with the same cell ordering.
        cells = grid.cell_nodes().tocsc().indices.reshape(
            (grid.dim + 1, grid.num_cells), order="F"
        ).copy()
        if grid.dim == 2:
            first = grid.nodes[:2, cells[1]] - grid.nodes[:2, cells[0]]
            second = grid.nodes[:2, cells[2]] - grid.nodes[:2, cells[0]]
            clockwise = first[0] * second[1] - first[1] * second[0] < 0
            cells[:, clockwise] = cells[[0, 2, 1]][:, clockwise]
            simplex = pp.TriangleGrid(grid.nodes.copy(), cells)
        else:
            simplex = pp.TetrahedralGrid(grid.nodes.copy(), cells)
        simplex.compute_geometry()
        # Preserve the meaning of PorePy's changed mask on repeated calls.
        simplex.cell_centers = grid.cell_centers.copy()

    centers, shift, changed = pp.grid_utils.compute_circumcenters(
        simplex, threshold=threshold, tol=tol
    )
    grid.cell_centers = centers
    return centers, shift, changed
