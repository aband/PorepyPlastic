from collections.abc import Callable

import numpy as np
import scipy.sparse as sps

from core.layout import DofLayout, FloatArray

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