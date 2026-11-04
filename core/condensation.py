from collections.abc import Callable

import numpy as np
import scipy.sparse as sps

from core.layout import DofLayout, FloatArray, IntArray

class CondensedElasticSystem:
    """Exact static condensation of cell unknowns [u, r, p] onto free faces.

    Uses only the elastic blocks of L. Plastic terms enter the face-face block
    alone (see AnalyticalJacobian.plastic_face_block), so these operators stay
    fixed for the whole simulation.
    """

    def __init__(self, L: sps.csr_array, layout: DofLayout) -> None:
        cs, fs = layout.cell_slice, layout.face_slice
        A_cc, A_cf = L[cs, cs], L[cs, fs]
        A_fc, A_ff = L[fs, cs], L[fs, fs]
        self._check_cell_structure(A_cc, layout)   # raises ValueError otherwise
        d = A_cc.diagonal()
        D_inv = sps.diags_array(1.0 / d)
        N = sps.csr_array(A_cc - sps.diags_array(d))
        self.A_cc_inv = sps.csr_array(D_inv - D_inv @ N @ D_inv)
        self.A_cf, self.A_fc = sps.csr_array(A_cf), sps.csr_array(A_fc)
        self.S_elastic = sps.csr_array(A_ff - A_fc @ (self.A_cc_inv @ A_cf))
        self.S_elastic.sort_indices()
        self.layout = layout

        # Cached for the whole simulation and handed out by face_matrix(None):
        # make accidental in-place changes by a face solver raise.
        # Canonical form first, so scipy never needs to sort them in place later.
        for matrix in (self.A_cc_inv, self.A_cf, self.A_fc, self.S_elastic):
            matrix.sum_duplicates()
            for array in (matrix.data, matrix.indices, matrix.indptr):
                array.flags.writeable = False

    @property
    def num_face_dofs(self) -> int:
        return int(self.S_elastic.shape[0])

    @staticmethod
    def _check_cell_structure(A_cc: sps.csr_array, layout: DofLayout) -> None:
        """A_cc = D + N with D diagonal (nonzero) and N only in the (u, r) and (u, p) blocks."""
        blocks = (layout.u_slice, layout.r_slice, layout.p_slice)
        for i, rows in enumerate(blocks):
            for j, cols in enumerate(blocks):
                block = sps.csr_array(A_cc[rows, cols])
                block.eliminate_zeros()
                if i == j:
                    off = sps.csr_array(block - sps.diags_array(block.diagonal()))
                    off.eliminate_zeros()
                    if off.nnz or np.any(block.diagonal() == 0.0):
                        raise ValueError("Cell diagonal blocks must be diagonal and nonsingular.")
                elif not (i == 0 and j > 0) and block.nnz:
                    raise ValueError("Only momentum rows may couple to rotation/pressure cells.")

    def face_matrix(self, plastic: sps.csr_array | None) -> sps.csr_array:
        return self.S_elastic if plastic is None else sps.csr_array(self.S_elastic + plastic)

    def reduce(self, rhs: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Face right-hand side b_f - A_fc A_cc^-1 b_c, and the cell part for recovery."""
        b_c, b_f = rhs[self.layout.cell_slice], rhs[self.layout.face_slice]
        b_hat = np.asarray(b_f - self.A_fc @ (self.A_cc_inv @ b_c), dtype=np.float64)
        return b_hat, b_c

    def recover(self, x_f: FloatArray, b_c: FloatArray) -> FloatArray:
        """Full correction from the face solution: x_c = A_cc^-1 (b_c - A_cf x_f)."""
        x_c = np.asarray(self.A_cc_inv @ (b_c - self.A_cf @ x_f), dtype=np.float64)
        return np.concatenate((x_c, np.asarray(x_f, dtype=np.float64)))

    def solve(
        self,
        rhs: FloatArray,
        plastic: sps.csr_array | None,
        face_solver: Callable[[sps.csr_array, FloatArray], FloatArray],
    ) -> FloatArray:
        """Full correction for J x = rhs; face_solver(S, b_f) must return x_f with S x_f = b_f."""
        b_f, b_c = self.reduce(rhs)
        if self.num_face_dofs == 0:
            return self.recover(np.empty(0), b_c)
        x_f = face_solver(self.face_matrix(plastic), b_f)
        return self.recover(x_f, b_c)

def _sorted_unique(keys: IntArray) -> IntArray:
    """Sorted unique integers; much faster than np.unique (hash-based in NumPy 2) here."""
    keys = np.sort(keys, kind="stable")
    keep = np.ones(keys.size, dtype=bool)
    keep[1:] = keys[1:] != keys[:-1]
    return keys[keep]


def _cell_local(
    cells: IntArray, dofs: IntArray, components: IntArray, values: FloatArray,
    num_cells: int, num_dofs: int, m: int,
) -> tuple[IntArray, FloatArray]:
    """Group sparse (cell, dof, Mandel component, value) entries into dense per-cell blocks.

    Returns the face dofs of each cell, shape (nc, K), padded with -1, and the
    dense values, shape (nc, K, m), with zero rows for the padding.
    """
    keys = _sorted_unique(cells * num_dofs + dofs)
    key_cells, key_dofs = keys // num_dofs, keys % num_dofs
    counts = np.bincount(key_cells, minlength=num_cells)
    width = int(counts.max(initial=0))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    slots = np.arange(keys.size) - starts[key_cells]

    local_dofs = np.full((num_cells, width), -1, dtype=np.int64)
    local_dofs[key_cells, slots] = key_dofs
    dense = np.zeros((num_cells, width, m))
    entry_slots = slots[np.searchsorted(keys, cells * num_dofs + dofs)]
    np.add.at(dense, (cells, entry_slots, components), values)
    return local_dofs, dense


class FixedFacePattern:
    """Condensed face matrix S = S_el + P_f D_Q B_f on one fixed CSR pattern.

    The pattern is the union of S_el and the plastic term with every cell
    plastic, so it never changes during a simulation: a solver can keep one
    matrix and update its values in place (and GAMG can reuse its
    interpolation). The union is larger than S_el, since the n (x) n term
    couples displacement components that cancel in S_el on regular grids.

    Each cell adds one dense block P_c D_c B_c, where P_c holds the rows of
    P_f on the cell's face dofs, B_c the matching columns of B_f, and D_c is
    the cell's m x m block of C_e - C_alg (zero on elastic cells).
    """

    def __init__(
        self,
        S_elastic: sps.csr_array,
        P_face: sps.csr_array,
        B_face: sps.csr_array,
        mandel_size: int,
    ) -> None:
        m = int(mandel_size)
        num_dofs = int(S_elastic.shape[0])
        if S_elastic.shape != (num_dofs, num_dofs):
            raise ValueError("S_elastic must be square.")
        if P_face.shape[0] != num_dofs or B_face.shape[1] != num_dofs:
            raise ValueError("P_face rows and B_face columns must match S_elastic.")
        if P_face.shape[1] != B_face.shape[0] or P_face.shape[1] % m:
            raise ValueError("P_face columns and B_face rows must be m per cell.")
        num_cells = P_face.shape[1] // m

        P = sps.coo_array(P_face)
        B = sps.coo_array(B_face)
        P_rows, P_cols = np.asarray(P.row, dtype=np.int64), np.asarray(P.col, dtype=np.int64)
        B_rows, B_cols = np.asarray(B.row, dtype=np.int64), np.asarray(B.col, dtype=np.int64)
        row_dofs, P_local = _cell_local(
            P_cols // m, P_rows, P_cols % m, P.data, num_cells, num_dofs, m,
        )
        col_dofs, B_local = _cell_local(
            B_rows // m, B_cols, B_rows % m, B.data, num_cells, num_dofs, m,
        )
        self.P_local = P_local                                   # (nc, R, m)
        self.B_local = np.ascontiguousarray(B_local.transpose(0, 2, 1))   # (nc, m, C)

        # All block entries (padding excluded) together with S_el form the pattern.
        valid = (row_dofs[:, :, None] >= 0) & (col_dofs[:, None, :] >= 0)
        block_keys = row_dofs[:, :, None] * num_dofs + col_dofs[:, None, :]
        S = sps.csr_array(S_elastic, copy=True)
        S.sum_duplicates()
        S_rows = np.repeat(np.arange(num_dofs, dtype=np.int64), np.diff(S.indptr))
        S_keys = S_rows * num_dofs + S.indices.astype(np.int64)
        keys = _sorted_unique(np.concatenate((S_keys, block_keys[valid])))

        self.nnz = int(keys.size)
        rows = keys // num_dofs
        self.indptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=num_dofs))))
        self.indices = keys % num_dofs
        self.shape = (num_dofs, num_dofs)

        # Padding entries go to a dump slot at index nnz, dropped after accumulation.
        self.position = np.where(valid, np.searchsorted(keys, block_keys), self.nnz)
        self.elastic_data = np.zeros(self.nnz)
        self.elastic_data[np.searchsorted(keys, S_keys)] = S.data

        for array in (
            self.P_local, self.B_local, self.position,
            self.indptr, self.indices, self.elastic_data,
        ):
            array.flags.writeable = False

    @property
    def num_face_dofs(self) -> int:
        return self.shape[0]

    def values(self, cells: IntArray, blocks: FloatArray) -> FloatArray:
        """Data of S_el + sum_c P_c D_c B_c on the fixed pattern; blocks has shape (k, m, m)."""
        cells = np.asarray(cells, dtype=np.int64)
        blocks = np.asarray(blocks, dtype=np.float64)
        m = self.P_local.shape[2]
        if blocks.shape != (cells.size, m, m):
            raise ValueError(f"Expected blocks of shape ({cells.size}, {m}, {m}).")
        if cells.size == 0:
            return self.elastic_data.copy()
        local = np.einsum(
            "kra,kab,kbs->krs",
            self.P_local[cells], blocks, self.B_local[cells], optimize=True,
        )
        plastic = np.bincount(
            self.position[cells].ravel(), weights=local.ravel(), minlength=self.nnz + 1,
        )
        return self.elastic_data + plastic[: self.nnz]

    def matrix(self, data: FloatArray) -> sps.csr_array:
        """CSR matrix with the fixed pattern and the given data (copied)."""
        data = np.array(data, dtype=np.float64, copy=True)
        if data.shape != (self.nnz,):
            raise ValueError(f"Expected {self.nnz} values.")
        return sps.csr_array((data, self.indices.copy(), self.indptr.copy()), shape=self.shape)
