"""Sparse nested Jacobians for TPSA with independent face displacements."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray

from utils.reconstruction import MANDEL_BASIS
from material.tangent import _validate_step, analytical_material_tangent, finite_difference_material_tangent
from utils.state import TpsaState
from material.material_state import MaterialPoint, MaterialState
from prototype.operators import RetainedFaceOperators
from prototype.residual import RetainedFaceResidual


class RetainedAnalyticalJacobian:
    """Equation (28) of Plasticity_revised.pdf: K = L + P D_Q B.

    All matrices use z = [delta_u,delta_r,delta_p,d_free]. B maps z variations
    to cell-major Mandel strain [xx,yy,sqrt(2)*xy], with no prescribed-face or
    cell-field variations in that reconstruction. D_Q = blockdiag(C_e-C_alg)
    uses the existing 3D return-map tangent restricted to plane strain.

    L is the fixed elastic residual derivative. P maps cell Mandel Q to the
    global residual; cellwise constant Q cancels in cell momentum, and does not
    enter the auxiliary rows. Only interior face-balance rows receive P D_Q B.
    T and P_side map z and Mandel Q to outward integrated one-sided tractions.
    face_jacobian includes boundary reaction derivatives as well as both sides
    of interior faces, with component-interleaved rows in operators.side order.

    Construct once per load step using the same operators/materials/history as
    residual evaluation. Committed history is copied; keep geometry, operator
    maps, material parameters, and hardening laws fixed. Each call requires the
    evaluation for that same z, boundary data, and committed history. Analytical
    assembly reuses its returned histories without additional local solves.
    At yield switches the selected return-map branch supplies the tangent.
    """

    def __init__(
        self, operators: RetainedFaceOperators, material_points: Sequence[MaterialPoint],
        committed: TpsaState,
    ) -> None:
        mu, lam = operators.mu, operators.lam
        if mu is None or lam is None:
            raise ValueError("Retained-face Jacobians require mu and lam.")
        if len(material_points) != operators.num_cells or len(committed.material_states) != operators.num_cells:
            raise ValueError("Expected one material point and committed history per cell.")
        for cell, point in enumerate(material_points):
            if not np.allclose(
                [point.material.isotropic_shear_modulus, point.material.lame_parameter],
                [mu[cell], lam[cell]], rtol=1e-12, atol=0,
            ):
                raise ValueError("Material elastic coefficients must match the retained-face operators.")
        # Reuse the kinematic checks for the fixed old boundary and plane strain.
        operators.reconstruct_strain(np.zeros(operators.num_dofs), committed.bc_values,
                                     committed.bc_values, committed.epsilon)
        self._operators = operators
        self._points = tuple(material_points)
        self._committed = committed.copy()
        self._assemble_fixed_maps()

    def _assemble_fixed_maps(self) -> None:
        """Differentiate the one-sided laws and balance equations directly."""
        ops = self._operators
        nc, ns, nz = ops.num_cells, ops.num_sides, ops.num_dofs
        stiffness, transfer = ops.side_stiffness, ops.rotation_transfer
        ar, ap = ops.rotation_accumulation, ops.pressure_accumulation
        if stiffness is None or transfer is None or ar is None or ap is None:
            raise ValueError("Retained-face Jacobians require traction and accumulation coefficients.")
        B = sps.lil_array((3 * nc, nz))
        T = sps.lil_array((2 * ns, nz))
        P_side = sps.lil_array((2 * ns, 3 * nc))
        sums = sps.lil_array((nz, 2 * ns))
        auxiliary = sps.lil_array((nz, nz))
        weights = transfer.weights
        identity = np.eye(2)
        for side, (face, cell) in enumerate(zip(ops.side_faces, ops.side_cells, strict=True)):
            area, normal = ops.side_areas[side], ops.side_normals[:, side]
            side_rows = slice(2 * side, 2 * side + 2)
            cell_rows = slice(2 * cell, 2 * cell + 2)
            strain_rows = slice(3 * cell, 3 * cell + 3)
            # N maps Mandel Q to Q*n; N.T maps d to Mandel sym(d outer n).
            N = np.einsum("aij,j->ia", MANDEL_BASIS, normal)
            T[side_rows, cell_rows] = -area * stiffness[side] * identity
            T[side_rows, 3 * nc + cell] = (area * normal)[:, None]
            start, end = weights.indptr[face:face + 2]
            for neighbor, weight in zip(weights.indices[start:end], weights.data[start:end], strict=True):
                T[side_rows, 2 * nc + neighbor] = (area * weight * np.array([-normal[1], normal[0]]))[:, None]
            P_side[side_rows, strain_rows] = -area * N
            sums[cell_rows, side_rows] = identity
            first_dof = ops.face_dofs[0, face]
            if first_dof >= 0:
                face_cols = slice(first_dof, first_dof + 2)
                T[side_rows, face_cols] = area * stiffness[side] * identity
                B[strain_rows, face_cols] = (area / ops.grid.cell_volumes[cell]) * N.T
                auxiliary[2 * nc + cell, face_cols] = (area * np.array([normal[1], -normal[0]]))[None, :]
                auxiliary[3 * nc + cell, face_cols] = (area * normal)[None, :]
                sums[face_cols, side_rows] = identity
        auxiliary.setdiag(np.r_[np.zeros(2 * nc), -ar, -ap, np.zeros(2 * ops.num_free_faces)])
        sums = sums.tocsr()
        self.B, self.T, self.P_side = B.tocsr(), T.tocsr(), P_side.tocsr()
        self.L = sps.csr_array(sums @ self.T + auxiliary.tocsr())
        # Express exact closed-cell cancellation rather than leaving roundoff
        # entries in the three cell blocks of the plastic correction map.
        self.P = sps.csr_array(sps.vstack((
            sps.csr_array((4 * nc, 3 * nc)), (sums @ self.P_side)[4 * nc:],
        ), format="csr"))
        for matrix in (self.B, self.T, self.P_side, self.L, self.P):
            matrix.eliminate_zeros()
            if not np.all(np.isfinite(matrix.data)):
                raise ValueError("Fixed retained-face derivative maps must be finite.")

    def _material_tangent(
        self, point: MaterialPoint, committed: MaterialState,
        increment: NDArray[np.float64], returned: MaterialState,
    ) -> NDArray[np.float64]:
        return analytical_material_tangent(point, committed, increment, returned=returned)

    def plastic_tangent(self, z: NDArray[np.float64], evaluation: RetainedFaceResidual) -> sps.csr_array:
        """Return D_Q = dQ_M/d epsilon_M as independent 3-by-3 cell blocks."""
        ops, material = self._operators, evaluation.material_trial
        ops._validate_unknowns(z)
        if not np.array_equal(z, evaluation.z):
            raise ValueError("Jacobian requires a candidate z matching the residual evaluation.")
        expected = ops.reconstruct_strain(z, evaluation.bc_values, self._committed.bc_values,
                                          self._committed.epsilon)
        if (material.epsilon.shape != expected.shape or not np.all(np.isfinite(material.epsilon))
                or not np.allclose(material.epsilon, expected, rtol=1e-12, atol=1e-14)):
            raise ValueError("Trial strain must match the candidate, boundary data, and committed strain.")
        if len(material.material_states) != ops.num_cells:
            raise ValueError("Expected one returned material history per cell.")
        blocks = []
        for cell, point in enumerate(self._points):
            tangent = self._material_tangent(
                point, self._committed.material_states[cell],
                material.epsilon[:, :, cell] - self._committed.epsilon[:, :, cell],
                material.material_states[cell],
            )
            elastic = point.material.elastic_tensor.to_numpy(copy=False)[:2, :2, :2, :2]
            blocks.append(np.einsum("aij,ijkl,bkl->ab", MANDEL_BASIS, elastic - tangent[:2, :2], MANDEL_BASIS))
        derivative = sps.csr_array(sps.block_diag(blocks, format="csr"))
        derivative.eliminate_zeros()
        if not np.all(np.isfinite(derivative.data)):
            raise ValueError("Retained-face material tangent must be finite.")
        return derivative

    def face_jacobian(self, z: NDArray[np.float64], evaluation: RetainedFaceResidual) -> sps.csr_array:
        """d delta_t/dz for ALL one-sided tractions, shape (2*num_sides,num_dofs)."""
        matrix = sps.csr_array(self.T + self.P_side @ self.plastic_tangent(z, evaluation) @ self.B)
        matrix.eliminate_zeros()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("One-sided traction Jacobian must be finite.")
        return matrix

    def __call__(self, z: NDArray[np.float64], evaluation: RetainedFaceResidual) -> sps.csr_array:
        """Return the sparse unscaled derivative of all retained-face residual rows."""
        matrix = sps.csr_array(self.L + self.P @ self.plastic_tangent(z, evaluation) @ self.B)
        matrix.eliminate_zeros()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Retained-face Jacobian must be finite.")
        return matrix


class RetainedFiniteDifferenceJacobian(RetainedAnalyticalJacobian):
    """Same sparse chain rule with the existing local forward-difference tangent.

    step is an absolute dimensionless gradient perturbation, default 1e-10.
    Each local perturbation starts from the fixed committed state. Global
    operators are differentiated exactly; no global finite differences occur.
    """

    def __init__(
        self, operators: RetainedFaceOperators, material_points: Sequence[MaterialPoint],
        committed: TpsaState, *, step: float = 1e-10,
    ) -> None:
        _validate_step(step)
        self.step = step
        super().__init__(operators, material_points, committed)

    def _material_tangent(
        self, point: MaterialPoint, committed: MaterialState,
        increment: NDArray[np.float64], returned: MaterialState,
    ) -> NDArray[np.float64]:
        return finite_difference_material_tangent(point, committed, increment, step=self.step)
