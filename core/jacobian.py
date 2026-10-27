import numpy as np
import scipy.sparse as sps

from core.residual import ResidualAssembler, ResidualEvaluation

from utils.reconstruction import mandel_basis

from core.layout import FloatArray
from core.state import MaterialHistory
from material.tangent import analytical_material_tangent
from utils.material_evaluation import MaterialTrial

class AnalyticalJacobian:
    """Analytical derivatives of the retained cell/face residual."""

    def __init__(self, assembler: ResidualAssembler) -> None:
        self.assembler = assembler
        self.operators = assembler.operators
        self.layout = assembler.layout

        nd = self.layout.dim
        self._mandel_basis = mandel_basis(nd)

        # Elastic material parameters remain fixed during Newton iterations.
        self._elastic_mandel = tuple(
            np.einsum(
                "aij,ijkl,bkl->ab",
                self._mandel_basis,
                point.material.elastic_tensor.to_numpy(copy=False)[
                    :nd, :nd, :nd, :nd
                ],
                self._mandel_basis,
                optimize=True,
            )
            for point in self.assembler.material_points
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

        self.H = self._build_auxiliary_matrix()

        self.L = sps.csr_array(self.S @ self.T + self.H)
        self.L.eliminate_zeros()

        if not np.all(np.isfinite(self.L.data)):
            raise ValueError("Elastic Jacobian must remain finite.")

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
        """Return blockdiag(C_e^M - C_alg^M) for the current material trial.

        committed is fixed throughout the load step.
        trial must come from the residual evaluation at the current candidate.
        """
        nd, nc = self.layout.dim, self.layout.num_cells
        shape = (3, 3, nc)

        if committed.dim != nd:
            raise ValueError("Material history dimension does not match the layout.")

        if committed.epsilon.shape != shape or trial.epsilon.shape != shape:
            raise ValueError(f"Expected committed and trial strains with shape {shape}.")

        if (
            len(committed.material_states) != nc
            or len(trial.material_states) != nc
        ):
            raise ValueError("Expected one committed and trial material state per cell.")

        increments = trial.epsilon - committed.epsilon
        basis = self._mandel_basis
        blocks: list[FloatArray] = []

        for cell, point in enumerate(self.assembler.material_points):
            algorithmic = analytical_material_tangent(
                point,
                committed.material_states[cell],
                increments[:, :, cell],
                dim=nd,
                returned=trial.material_states[cell],
            )

            # Input strain indices already have size nd; select active stress rows.
            algorithmic_mandel = np.einsum(
                "aij,ijkl,bkl->ab",
                basis,
                algorithmic[:nd, :nd, :, :],
                basis,
                optimize=True,
            )

            blocks.append(
                self._elastic_mandel[cell] - algorithmic_mandel
            )

        matrix = sps.csr_array(sps.block_diag(blocks, format="csr"))
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

        matrix = sps.csr_array(
            self.L + self.P @ (D_Q @ self.B_z)
        )
        matrix.eliminate_zeros()

        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Analytical Jacobian must remain finite.")

        return matrix