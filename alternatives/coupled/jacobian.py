"""Exact sparse monolithic chain rule for the complementary-weight TPSA residual."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray

from alternatives.coupled.residual import CoupledOperators
from utils.reconstruction import MANDEL_BASIS
from material.tangent import analytical_material_tangent, finite_difference_material_tangent
from alternatives.original.residual import TpsaOperators, TpsaResidual
from utils.state import TpsaState
from utils.transfer import CellToFaceTransfer
from material.material_state import MaterialPoint, MaterialState


class CoupledAnalyticalJacobian:
    """Equation (7.2): [[A, H D_Q], [-B, I-H_epsilon D_Q]].

    Strain is an independent Newton unknown, encoded in cell-major Mandel form.
    D_Q = blockdiag(C_e-C_alg) uses the consistent 3D plane-strain return map.
    H includes momentum, rotation and pressure plastic-source maps. Geometry,
    elastic coefficients, complementary transfer and history are held fixed.
    """

    def __init__(
        self, operators: TpsaOperators, transfer: CellToFaceTransfer,
        material_points: Sequence[MaterialPoint], committed: TpsaState,
    ) -> None:
        if not isinstance(operators, CoupledOperators):
            raise ValueError("Coupled Jacobians require CoupledOperators.")
        operators.validate_transfer(transfer)
        operators.validate_committed(committed, material_points)
        self._operators = operators
        self._points = tuple(material_points)
        self._committed = committed.copy()

    def _material_tangent(
        self, point: MaterialPoint, committed: MaterialState,
        increment: NDArray[np.float64], returned: MaterialState,
    ) -> NDArray[np.float64]:
        return analytical_material_tangent(point, committed, increment, returned=returned)

    def plastic_tangent(self, z: NDArray[np.float64], evaluation: TpsaResidual) -> sps.csr_array:
        """Return dQ/d epsilon in matching Mandel stress/strain coordinates."""
        ops, state = self._operators, evaluation.trial
        if (z.shape != (7 * ops.num_cells,) or not np.all(np.isfinite(z))
                or not np.allclose(z, ops.unknowns(state), rtol=1e-14, atol=1e-15)):
            raise ValueError("Coupled Jacobian requires a candidate matching the residual evaluation.")
        blocks = []
        for cell, point in enumerate(self._points):
            material = self._material_tangent(point, self._committed.material_states[cell],
                state.epsilon[:, :, cell] - self._committed.epsilon[:, :, cell], state.material_states[cell])
            elastic = point.material.elastic_tensor.to_numpy(copy=False)[:2, :2, :2, :2]
            blocks.append(np.einsum("aij,ijkl,bkl->ab", MANDEL_BASIS,
                                     elastic - material[:2, :2], MANDEL_BASIS))
        derivative = sps.csr_array(sps.block_diag(blocks, format="csr"))
        if not np.all(np.isfinite(derivative.data)):
            raise ValueError("Coupled material tangent must be finite.")
        return derivative

    def face_jacobian(self, z: NDArray[np.float64], evaluation: TpsaResidual) -> sps.csr_array:
        """All integrated reaction/interior tractions versus all [u,r,p,epsilon] DOFs."""
        ops = self._operators
        return sps.csr_array(sps.hstack((ops.T, ops.H_t @ self.plastic_tangent(z, evaluation)), format="csr"))

    def __call__(self, z: NDArray[np.float64], evaluation: TpsaResidual) -> sps.csr_array:
        ops = self._operators
        dq = self.plastic_tangent(z, evaluation)
        matrix = sps.csr_array(sps.block_array([
            [ops.A, ops.H @ dq],
            [-ops.B, sps.eye(3 * ops.num_cells, format="csr") - ops.H_epsilon @ dq],
        ], format="csr"))
        matrix.eliminate_zeros()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Coupled Jacobian must be finite.")
        return matrix


class CoupledFiniteDifferenceJacobian(CoupledAnalyticalJacobian):
    """Same monolithic chain rule, with local forward-difference material tangents.

    The four in-plane gradient perturbations are converted to Mandel strain
    columns with the same symmetric basis as the analytical derivative. This
    preserves the existing finite-difference step convention (default 1e-10).
    """

    def __init__(
        self, operators: TpsaOperators, transfer: CellToFaceTransfer,
        material_points: Sequence[MaterialPoint], committed: TpsaState, *, step: float = 1e-10,
    ) -> None:
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Finite-difference step must be finite and positive.")
        self.step = step
        super().__init__(operators, transfer, material_points, committed)

    def _material_tangent(
        self, point: MaterialPoint, committed: MaterialState,
        increment: NDArray[np.float64], returned: MaterialState,
    ) -> NDArray[np.float64]:
        return finite_difference_material_tangent(point, committed, increment, step=self.step)
