"""Fixed, configurable transfer of cell fields to shared numerical face values."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.grid import Grid

WeightRule: TypeAlias = Callable[[Grid, int, NDArray[np.int64]], NDArray[np.float64]]


def equal_weights(grid: Grid, face: int, cells: NDArray[np.int64]) -> NDArray[np.float64]:
    """Assign equal weight to the supplied adjacent cells."""
    return np.full(cells.size, 1.0 / cells.size)


def _adjacency(grid: Grid) -> sps.csr_array:
    """Unsigned face-to-cell incidence for faces with one or two physical neighbors."""
    if grid.num_cells < 1 or grid.num_faces < 1:
        raise ValueError("Transfer requires a nonempty grid.")
    adjacency = sps.csr_array(grid.cell_faces, dtype=np.float64, copy=True)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    if adjacency.shape != (grid.num_faces, grid.num_cells) or not np.all(
        np.isfinite(adjacency.data)
    ):
        raise ValueError("Face-cell connectivity must be finite and match the grid.")
    if not np.all(np.isin(np.diff(adjacency.indptr), [1, 2])):
        raise ValueError("Each face must have one or two adjacent cells.")
    adjacency.data[:] = 1.0  # Orientation signs belong in divergence, not in weights.
    return adjacency


def _weights_from_rule(grid: Grid, rule: WeightRule) -> sps.csr_array:
    weights = _adjacency(grid)
    for face in range(grid.num_faces):
        start, end = weights.indptr[face:face + 2]
        if end - start == 1:
            continue  # Boundary faces always use their single physical neighbor.
        cells = np.array(weights.indices[start:end], dtype=np.int64, copy=True)
        values = np.asarray(rule(grid, face, cells), dtype=np.float64)
        if values.shape != cells.shape:
            raise ValueError(f"Weight rule for face {face} must return shape {cells.shape}.")
        weights.data[start:end] = values
    return weights


class CellToFaceTransfer:
    """Store W[f, cell] and apply the same weights to every field component.

    Default: 1/2 per adjacent cell on interior faces, 1 on boundary faces.
    Supply a sparse weights matrix, or use from_rule(grid, rule) to customize
    interior weights. Rules receive (grid, face_index, sorted_adjacent_cell_ids)
    and return normalized weights in that cell order. A rule can close over fixed
    material coefficients; it is evaluated only during construction.

    Weights must be finite, nonnegative, supported on adjacent cells, and sum
    to one on every face. Boundary rows are consequently one-sided. No face
    areas or orientation signs enter W, and no silent normalization is applied.
    This is a convex transfer choice, not a claim of spatial consistency for
    arbitrary grids or material interfaces.

    The object owns a matrix snapshot; weights returns a copy for inspection or
    editing. Construct a new object to change the rule or weights. Keep it fixed
    throughout a Newton solve so the same W can be used in the later Jacobian.
    """

    def __init__(
        self, grid: Grid, weights: sps.spmatrix | sps.sparray | None = None,
    ) -> None:
        adjacency = _adjacency(grid)
        self.num_cells, self.num_faces = grid.num_cells, grid.num_faces
        matrix = (
            _weights_from_rule(grid, equal_weights) if weights is None
            else sps.csr_array(weights, dtype=np.float64, copy=True)
        )
        if matrix.shape != adjacency.shape:
            raise ValueError(f"Transfer weights must have shape {adjacency.shape}.")
        if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0.0):
            raise ValueError("Transfer weights must be finite and nonnegative.")
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Transfer weights must be finite and nonnegative.")
        if (matrix - matrix.multiply(adjacency)).nnz:
            raise ValueError("Transfer weights may only use adjacent cells.")
        if not np.allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("Transfer weights must sum to one on every face.")
        self._weights = matrix

    @classmethod
    def from_rule(cls, grid: Grid, rule: WeightRule) -> CellToFaceTransfer:
        """Build once from a custom interior rule; boundary weights remain one."""
        return cls(grid, _weights_from_rule(grid, rule))

    @property
    def weights(self) -> sps.csr_array:
        """Return a copy of the (num_faces, num_cells) transfer matrix."""
        return self._weights.copy()

    def apply(self, cell_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transfer (..., nc) to (..., nf), preserving all components and inputs.

        For a stress correction, (3, 3, nc) becomes (3, 3, nf). Scalar and vector
        fields are also supported. Each face gets one shared value, independent
        of which neighboring cell later uses it in the divergence.
        """
        values = np.asarray(cell_values, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != self.num_cells or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected finite cell values with last axis of size {self.num_cells}.")
        flat = values.reshape((-1, self.num_cells))
        face_values = np.asarray(self._weights @ flat.T, dtype=np.float64).T
        return face_values.reshape((*values.shape[:-1], self.num_faces))
