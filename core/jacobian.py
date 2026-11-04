import numpy as np
import scipy.sparse as sps

from core.residual import ResidualAssembler, ResidualEvaluation

from utils.reconstruction import mandel_basis

from core.state import MaterialHistory, MaterialTrial
from material.tensor import deviatoric_projector

from core.condensation import CondensedElasticSystem

class AnalyticalJacobian:
    """Analytical derivatives of the retained cell/face residual."""

    def __init__(self, assembler: ResidualAssembler) -> None:
        self.assembler = assembler
        self.operators = assembler.operators
        self.layout = assembler.layout

        nd = self.layout.dim
        self._mandel_basis = mandel_basis(nd)

        m = self._mandel_basis.shape[0]
        self._mandel_projection = self._mandel_basis.reshape(
            m, nd * nd, order="C"
        )

        # Cache local block indices
        indices = np.arange(m, dtype=np.int64)
        self._block_rows = np.repeat(indices, m)
        self._block_cols = np.tile(indices, m)

        # In-plane Mandel form of the deviatoric projector, used by every
        # plastic block of C_e - C_alg.
        projector = deviatoric_projector(3)[:nd, :nd, :nd, :nd]
        self._deviatoric_mandel = (
            self._mandel_projection
            @ projector.reshape(nd * nd, nd * nd, order="C")
            @ self._mandel_projection.T
        )

        # Strain variations depend only on free-face displacement variations.
        B_free = self.operators.B_free
        self.B_z = sps.csr_array(
            sps.hstack(
                (
                    sps.csr_array(
                        (B_free.shape[0], self.layout.cell_size)
                    ),
                    B_free,
                ),
                format="csr",
            )
        )

        self.T = self._build_elastic_traction_matrix()
        self.S = self._build_balance_matrix()
        self.P_side = self._build_stress_correction_matrix()
        self.P = sps.csr_array(self.S @ self.P_side)
        self.P.eliminate_zeros()

        face = self.layout.face_slice
        cell_rows = self.P[: self.layout.cell_size]

        # Cell rows of P vanish analytically (eq. 21: sum_s A_s n_s = 0 per cell).
        # Allow cancellation round-off relative to the individual side terms in
        # P_side; larger entries would mean a broken P_side or S. Comparing with
        # max|P| fails when there are no free faces, since P is then round-off only.
        # The condensed path uses P_face alone, so round-off there never matters.
        scale = np.abs(self.P_side.data).max(initial=0.0)
        if cell_rows.nnz and np.abs(cell_rows.data).max() > 1e-12 * scale:
            raise ValueError("Stress-correction map must not reach cell rows.")
        self.P_face = sps.csr_array(self.P[face])
        self.B_face = sps.csr_array(self.operators.B_free)
        self._condensed: CondensedElasticSystem | None = None

        self.H = self._build_auxiliary_matrix()

        self.L = sps.csr_array(self.S @ self.T + self.H)
        self.L.eliminate_zeros()

        if not np.all(np.isfinite(self.L.data)):
            raise ValueError("Elastic Jacobian must remain finite.")

    def plastic_face_block(
            self, 
            committed:MaterialHistory, 
            evaluation:ResidualEvaluation,
        ) -> sps.csr_array | None:
        """P_f @ D_Q @ B_f for the evaluated candidate; None if every cell is elastic."""
        trial = evaluation.material_trial   # same validation as evaluate()
        if trial is None:
            raise ValueError(
                "Jacobian evaluation requires a material trial "
                "from ResidualAssembler.evaluate()."
            )

        D_Q = self._build_material_correction_tangent(committed, trial)
        if D_Q.nnz == 0:
            return None
        return sps.csr_array(self.P_face @ (D_Q @ self.B_face))

    def condensed_elastic(self) -> CondensedElasticSystem:
        """Built on first use and cached: several solvers on one Jacobian share it."""
        if self._condensed is None:
            self._condensed = CondensedElasticSystem(self.L, self.layout)
        return self._condensed

    def _build_elastic_traction_matrix(self) -> sps.csr_array:
        """Differentiate integrated elastic side tractions with respect to z."""
        operators, layout = self.operators, self.layout
        nd, nr, nc = layout.dim, layout.rotation_dim, layout.num_cells
        ns = operators.num_sides

        stiffness = operators.side_stiffness
        transfer = operators.rotation_transfer
        if stiffness is None or transfer is None:
            raise ValueError("Traction coefficients must be initialized.")

        faces, cells = operators.side_faces, operators.side_cells
        areas, normals = operators.side_areas, operators.side_normals

        rows = np.arange(nd * ns, dtype=np.int64)
        components = np.arange(nd, dtype=np.int64)[:, None]
        gain = np.repeat(areas * stiffness, nd)

        # Cell displacement derivative T = -A_s * a_s * I
        cell_cols = (nd * cells[None, :] + components).ravel(order="F")
        T_u = sps.coo_array(
            (-gain, (rows, cell_cols)),
            shape=(nd*ns, nd*nc),
        )

        # Free-face displacement derivative
        face_cols = layout.face_dofs[:,faces].ravel(order="F")
        free = face_cols >= 0
        T_d = sps.coo_array(
            (
                gain[free],
                (rows[free], face_cols[free] - layout.cell_size),
            ),
            shape=(nd*ns, layout.size - layout.cell_size),
        )

        T_p = sps.coo_array(
            (
                (areas[None, :] * normals).ravel(order="F"),
                (rows, np.repeat(cells, nd)),
            ),
            shape=(nd*ns, nc),
        )

        # rotation
        if nd == 2:
            C = np.vstack((-normals[1], normals[0]))[:, None, :]
        else:
            C = np.zeros((nd, nr, ns))
            C[0, 1], C[0, 2] = normals[2], -normals[1]
            C[1, 0], C[1, 2] = -normals[2], normals[0]
            C[2, 0], C[2, 1] = normals[1], -normals[0]

        i, j, side = np.nonzero(C)
        side_rotation = sps.coo_array(
            (
                areas[side] * C[i, j, side],
                (nd * side + i, nr * side + j),
            ),
            shape=(nd * ns, nr * ns),
        ).tocsr()

        rotation_transfer = sps.kron(
            transfer.weights[faces, :],
            sps.eye_array(nr, format="csr"),
            format="csr",
        )
        T_r = side_rotation @ rotation_transfer

        matrix = sps.csr_array(
            sps.hstack((T_u, T_r, T_p, T_d), format="csr")
        )
        matrix.eliminate_zeros()

        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Elastic traction matrix must remain finite.")

        return matrix


    def _build_balance_matrix(self) -> sps.csr_array:
        """Sum side tractions into cell momentum and free-face balance rows."""
        operators, layout = self.operators, self.layout
        nd = layout.dim
        ns = operators.num_sides

        # Columns follow traction_increment.ravel(order="F").
        side_cols = np.arange(nd * ns, dtype=np.int64)
        components = np.arange(nd, dtype=np.int64)[:, None]

        # Every side contributes to its adjacent cell's momentum balance.
        # The displacement/momentum block starts at index zero.
        momentum_rows = (
            nd * operators.side_cells[None, :] + components
        ).ravel(order="F")

        # Free-face residual rows use the same global indices as face DOFs.
        # Dirichlet face components have index -1.
        face_rows = layout.face_dofs[
            :, operators.side_faces
        ].ravel(order="F")
        free = face_rows >= 0

        rows = np.concatenate(
            (momentum_rows, face_rows[free])
        )
        cols = np.concatenate(
            (side_cols, side_cols[free])
        )

        # Both balances sum outward tractions with coefficient +1.
        values = np.ones(rows.size, dtype=np.float64)

        return sps.coo_array(
            (values, (rows, cols)),
            shape=(layout.size, nd * ns),
        ).tocsr()


    def _build_auxiliary_matrix(self) -> sps.csr_array:
        """Differentiate rotation/pressure fluxes and subtract accumulation."""
            
        operators, layout = self.operators, self.layout
        nd, nr, nc = layout.dim, layout.rotation_dim, layout.num_cells
        ns = operators.num_sides
        nf_dofs = layout.size - layout.cell_size

        faces, cells = operators.side_faces, operators.side_cells
        areas, normals = operators.side_areas, operators.side_normals

        # Local free-face column indices; Dirichlet entries remain negative.
        face_cols = layout.face_dofs[:, faces] - layout.cell_size

        # Map face displacement to d × n.
        # Shape: (rotation_dim, dim, num_sides).
        if nd == 2:
            rotation_map = np.vstack(
                (normals[1], -normals[0])
            )[None, :, :]
        else:
            rotation_map = np.zeros((nr, nd, ns))
            rotation_map[0, 1], rotation_map[0, 2] = (
                normals[2], -normals[1]
            )
            rotation_map[1, 0], rotation_map[1, 2] = (
                -normals[2], normals[0]
            )
            rotation_map[2, 0], rotation_map[2, 1] = (
                normals[1], -normals[0]
            )

        # Rotation-flux derivative: sum_s A_s * (delta_d_f × n_s).
        r, i, side = np.nonzero(rotation_map)
        cols = face_cols[i, side]
        free = cols >= 0
        values = areas[side] * rotation_map[r, i, side]

        G_r = sps.coo_array(
            (
                values[free],
                (
                    (nr * cells[side] + r)[free],
                    cols[free],
                ),
            ),
            shape=(nr * nc, nf_dofs),
        ).tocsr()

        # Pressure-flux derivative: sum_s A_s * (delta_d_f · n_s).
        i, side = np.nonzero(normals)
        cols = face_cols[i, side]
        free = cols >= 0
        values = areas[side] * normals[i, side]

        G_p = sps.coo_array(
            (
                values[free],
                (cells[side][free], cols[free]),
            ),
            shape=(nc, nf_dofs),
        ).tocsr()

        # Cell rows follow [momentum, rotation, pressure].
        G_free = sps.csr_array(
            sps.vstack(
                (
                    sps.csr_array((nd * nc, nf_dofs)),
                    G_r,
                    G_p,
                ),
                format="csr",
            )
        )

        matrix = sps.csr_array(
            sps.block_array(
                [
                    [-self.assembler.accumulation, G_free],
                    [
                        sps.csr_array((nf_dofs, layout.cell_size)),
                        sps.csr_array((nf_dofs, nf_dofs)),
                    ],
                ],
                format="csr",
            )
        )
        matrix.eliminate_zeros()

        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Auxiliary matrix must remain finite.")

        return matrix

    def _build_stress_correction_matrix(self) -> sps.csr_array:
        """Map cell Mandel stress corrections to integrated side tractions.

        P_side @ mandel(Q, dim=nd) gives (-A_s * Q_cell @ n_s),
        flattened in the same order as traction_increment.ravel(order="F").
        """
        operators, layout = self.operators, self.layout
        nd, nc = layout.dim, layout.num_cells
        ns = operators.num_sides

        basis = mandel_basis(nd)  # Shape: (m, nd, nd).
        m = basis.shape[0]

        # Each column is the traction generated by one Mandel basis tensor.
        # Shape: (nd, m, ns).
        blocks = -operators.side_areas[None, None, :] * np.einsum(
            "aij,js->ias",
            basis,
            operators.side_normals,
            optimize=True,
        )

        component, mandel_component, side = np.nonzero(blocks)

        rows = nd * side + component
        cols = m * operators.side_cells[side] + mandel_component

        matrix = sps.coo_array(
            (
                blocks[component, mandel_component, side],
                (rows, cols),
            ),
            shape=(nd * ns, m * nc),
        ).tocsr()
        matrix.eliminate_zeros()

        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Stress-correction matrix must remain finite.")

        return matrix

    def _build_material_correction_tangent(
        self,
        committed: MaterialHistory,
        trial: MaterialTrial,
    ) -> sps.csr_array:
        """
        Return blockdiag(C_e^M - C_alg^M) for the current material trial.

        committed is fixed throughout the load step.
        trial must come from the residual evaluation at the current candidate.

        On plastic cells, C_e - C_alg = c_dev P_dev + c_nn (n outer n), so
        each block is formed directly without the elastic tensor. Elastic
        cells contribute nothing. No return map is repeated.
        """
        nd, nc = self.layout.dim, self.layout.num_cells
        shape = (3, 3, nc)

        if committed.dim != nd:
            raise ValueError("Material history dimension does not match the layout.")

        if committed.epsilon.shape != shape or trial.epsilon.shape != shape:
            raise ValueError(f"Expected committed and trial strains with shape {shape}.")

        if (
            committed.alpha.shape != (nc,)
            or trial.history.alpha.shape != (nc,)
            or trial.return_map.dgamma.shape != (nc,)
        ):
            raise ValueError("Expected one committed and trial material state per cell.")

        M = self._mandel_projection
        m = M.shape[0]
        size = m * nc

        cells, c_dev, c_nn = self.assembler.material.tangent_correction(trial)

        # All cells followed the elastic branch.
        if cells.size == 0:
            return sps.csr_array((size, size), dtype=np.float64)

        # In-plane Mandel flow directions, shape (m, plastic cells).
        direction = trial.return_map.flow_direction[:nd, :nd, cells]
        n_mandel = M @ direction.reshape(nd * nd, cells.size, order="C")

        # Blocks ordered (cell, row, column) to match the index layout below.
        blocks = (
            c_dev[:, None, None] * self._deviatoric_mandel[None, :, :]
            + c_nn[:, None, None]
            * np.einsum("ak,bk->kab", n_mandel, n_mandel)
        )

        # Preserve each plastic cell's original global position
        offsets = m * cells

        rows = (
            offsets[:,None] + self._block_rows[None, :]
        ).ravel(order="C")

        cols = (
            offsets[:,None] + self._block_cols[None, :]
        ).ravel(order="C")

        entries = blocks.ravel(order="C")

        matrix = sps.coo_array(
            (entries,(rows,cols)),
            shape=(size,size),
        ).tocsr()

        matrix.eliminate_zeros()

        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Material correction tangent must remain finite.")

        return matrix

    def evaluate(
        self,
        committed: MaterialHistory,
        evaluation: ResidualEvaluation,
    )->sps.csr_array:
        """Assemble the analytical Jacobian for an evaluated candidate.

        evaluation must come from ResidualAssembler.evaluate() using the same
        committed history. Reuse its material trial without another return map.
        """
        trial = evaluation.material_trial
        if trial is None:
            raise ValueError(
                "Jacobian evaluation requires a material trial "
                "from ResidualAssembler.evaluate()."
            )

        D_Q = self._build_material_correction_tangent(committed, trial)

        if D_Q.nnz == 0:
            return self.L.copy()

        matrix = sps.csr_array(
            self.L + self.P @ (D_Q @ self.B_z)
        )
        matrix.eliminate_zeros()

        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Analytical Jacobian must remain finite.")

        return matrix