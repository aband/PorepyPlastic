"""Analytical and finite-difference material tangents with coupled TPSA assembly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray

from coupling.residual import TpsaOperators, TpsaResidual, TpsaState
from coupling.transfer import CellToFaceTransfer
from material_state import MaterialPoint, MaterialState
from tensor import strain, stress


def _validate_step(step: float) -> None:
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Finite-difference step must be finite and positive.")


def finite_difference_material_tangent(
    point: MaterialPoint,
    committed: MaterialState,
    strain_increment: NDArray[np.float64],
    *,
    step: float = 1e-10,
) -> NDArray[np.float64]:
    """Return d sigma[i,j] / d G[k,l], shape (3,3,2,2), by forward differences.

    G is the in-plane displacement gradient. Perturbing G[k,l] by step changes
    total strain by step * sym(e_k outer e_l); shear perturbations therefore
    change BOTH symmetric strain entries by step/2. The full 3D return map keeps
    out-of-plane stress and plastic history while varying only in-plane strain.

    Each of five local updates (baseline plus four perturbations) starts from
    the supplied committed history. point supplies only model/parameters; its
    committed/trial slots and all inputs remain unchanged, including on failure.
    step is an absolute dimensionless gradient perturbation, defaulting to the
    paper's 1e-10. At a yield switch this is a one-sided numerical derivative.
    """
    _validate_step(step)
    increment = np.array(strain_increment, dtype=np.float64, copy=True)
    if increment.shape != (3, 3) or not np.all(np.isfinite(increment)):
        raise ValueError("Expected a finite strain increment with shape (3, 3).")
    local = MaterialPoint(
        material=point.material, parameters=point.parameters, model=point.model,
        K_law=point.K_law, H_law=point.H_law, committed=committed,
    )

    def stress_at(values: NDArray[np.float64]) -> NDArray[np.float64]:
        history, _ = local.update(strain(values))
        stress = np.asarray(history.stress.to_numpy(), dtype=np.float64)
        if stress.shape != (3, 3) or not np.all(np.isfinite(stress)):
            raise ValueError("Material update returned a non-finite or invalid stress.")
        return stress

    baseline = stress_at(increment)
    tangent = np.empty((3, 3, 2, 2))
    for k in range(2):
        for l in range(2):
            basis = np.zeros((3, 3))
            basis[k, l] += 0.5
            basis[l, k] += 0.5
            perturbed = increment + step * basis
            if not np.all(np.isfinite(perturbed)) or np.array_equal(perturbed, increment):
                raise ValueError("Finite-difference step cannot produce a finite, distinct perturbation.")
            tangent[:, :, k, l] = (stress_at(perturbed) - baseline) / step
    if not np.all(np.isfinite(tangent)):
        raise ValueError("Finite-difference material tangent must be finite.")
    return tangent


def _gradient_matrix(operators: TpsaOperators) -> sps.csr_array:
    """Exact d vec(G_cell) / d x at fixed Dirichlet values; cell-major xx,xy,yx,yy."""
    grid = operators.grid
    nc, nf = operators.num_cells, operators.num_faces
    normal = grid.face_normals[:2] / grid.face_areas
    tangent = np.vstack((normal[1], -normal[0]))
    active = np.ones(nf)
    active[operators.boundary_faces] = 0.0  # Prescribed face displacement has zero derivative.
    rotation = operators.face_discretization[2 * nf:3 * nf]
    mass = operators.face_discretization[3 * nf:]
    average = sps.diags(1 / grid.cell_volumes) @ grid.cell_faces.T
    components = []
    for i in range(2):
        face = (
            sps.diags(active * normal[i] / grid.face_areas) @ mass
            + sps.diags(active * tangent[i] / grid.face_areas) @ rotation
        )
        for j in range(2):
            components.append(average @ sps.diags(grid.face_normals[j]) @ face)
    grouped = sps.vstack(components, format="csr")
    order = np.arange(4 * nc).reshape((4, nc)).T.ravel()
    return sps.csr_array(grouped[order])


def analytical_material_tangent(
    point: MaterialPoint,
    committed: MaterialState,
    strain_increment: NDArray[np.float64],
    *,
    returned: MaterialState | None = None,
) -> NDArray[np.float64]:
    """Return exact J2 d sigma[i,j]/d G[k,l], with shape (3,3,2,2).

    Use the same 3D radial return and scalar hardening laws as MaterialPoint.
    Optionally reuse returned from that SAME increment and committed history;
    otherwise perform one isolated local update. No finite differences are used.
    The supplied point's committed/trial slots and all input states are preserved.
    Both shear columns act on sym(delta_G), with no engineering-shear factor.
    Out-of-plane stress rows are retained. At a yield switch use the branch
    selected by the return map, rather than claiming a unique smooth derivative.
    """
    increment = np.asarray(strain_increment, dtype=np.float64)
    if (increment.shape != (3, 3) or not np.all(np.isfinite(increment))
            or not np.allclose(increment, increment.T, rtol=1e-10, atol=1e-14)):
        raise ValueError("Expected a finite symmetric strain increment with shape (3, 3).")
    if returned is None:
        local = MaterialPoint(
            point.material, point.parameters, point.model, point.K_law, point.H_law,
            committed=committed,
        )
        returned, _ = local.update(strain(increment))
    # Recover the multiplier in THIS implementation's unit-flow convention.
    dgamma = (returned.alpha - committed.alpha) / np.sqrt(2 / 3)
    elastic = point.material.elastic_tensor.to_numpy(copy=False)
    predictor = committed.stress.to_numpy() + np.einsum("ijkl,kl->ij", elastic, increment)
    tangent = point.model.consistent_tangent(
        stress(predictor), committed.backstress, returned.alpha, float(dgamma),
        point.material, point.K_law, point.H_law, point.parameters,
    )
    return tangent.to_numpy()[:, :, :2, :2]


class _TpsaJacobian(ABC):
    """Shared exact sparse chain rule; subclasses supply only the material tangent."""

    def __init__(
        self,
        operators: TpsaOperators,
        transfer: CellToFaceTransfer,
        material_points: Sequence[MaterialPoint],
        committed: TpsaState,
    ) -> None:
        nc, nf = operators.num_cells, operators.num_faces
        if (transfer.num_cells, transfer.num_faces) != (nc, nf):
            raise ValueError("Transfer must match the operators' cell and face counts.")
        if len(material_points) != nc or len(committed.material_states) != nc:
            raise ValueError("Expected one material point and committed history per cell.")
        if committed.epsilon.shape != (3, 3, nc) or not np.all(np.isfinite(committed.epsilon)):
            raise ValueError("Expected finite committed strain with shape (3, 3, nc).")
        self._num_cells = nc
        self._points = tuple(material_points)
        self._committed = committed.copy()
        self._gradient = _gradient_matrix(operators)
        faces = np.arange(nf)
        rows = np.concatenate((2 * faces, 2 * faces, 2 * faces + 1, 2 * faces + 1))
        cols = np.concatenate((4 * faces, 4 * faces + 1, 4 * faces + 2, 4 * faces + 3))
        nx, ny = operators.grid.face_normals[:2]
        normal = sps.csr_array(
            (np.concatenate((nx, ny, nx, ny)), (rows, cols)), shape=(2 * nf, 4 * nf),
        )
        self._normal_transfer = normal @ sps.kron(transfer.weights, sps.eye(4), format="csr")
        self._traction = operators.T.copy()
        self._divergence = operators.D_u.copy()
        self._auxiliary = operators.A[2 * nc:].copy()

    @abstractmethod
    def _material_tangent(
        self, point: MaterialPoint, committed: MaterialState,
        increment: NDArray[np.float64], returned: MaterialState,
    ) -> NDArray[np.float64]:
        """Return d sigma/d G with shape (3,3,2,2) at fixed committed history."""

    def face_jacobian(self, x: NDArray[np.float64], evaluation: TpsaResidual) -> sps.csr_array:
        """Derivative of all integrated tractions, shape (2*nf,4*nc).

        evaluation must correspond to x, this callback's committed history and
        fixed data. Include interior and Dirichlet reaction faces; fixed boundary
        displacement has zero variation, while the boundary traction does not.
        """
        nc = self._num_cells
        if x.shape != (4 * nc,) or not np.all(np.isfinite(x)) or not np.array_equal(x, evaluation.trial.x):
            raise ValueError("Jacobian requires a finite candidate x matching evaluation.trial.x.")
        epsilon = evaluation.trial.epsilon
        if epsilon.shape != (3, 3, nc) or not np.all(np.isfinite(epsilon)):
            raise ValueError("Expected finite trial strain with shape (3, 3, nc).")
        if len(evaluation.trial.material_states) != nc:
            raise ValueError("Expected one returned material history per cell.")
        blocks = []
        for cell, point in enumerate(self._points):
            material = self._material_tangent(
                point, self._committed.material_states[cell],
                epsilon[:, :, cell] - self._committed.epsilon[:, :, cell],
                evaluation.trial.material_states[cell],
            )
            # Minor symmetry already accounts for sym(G), including half-shear.
            elastic = point.material.elastic_tensor.to_numpy(copy=False)[:2, :2, :2, :2]
            blocks.append((elastic - material[:2, :2]).reshape((4, 4)))
        derivative = sps.block_diag(blocks, format="csr")
        face = sps.csr_array(self._traction - self._normal_transfer @ derivative @ self._gradient)
        face.eliminate_zeros()
        if not np.all(np.isfinite(face.data)):
            raise ValueError("Assembled face Jacobian must be finite.")
        return face

    def __call__(self, x: NDArray[np.float64], evaluation: TpsaResidual) -> sps.csr_array:
        """Assemble all [u,r,p] rows; divergence carries orientation, not extra area."""
        momentum = self._divergence @ self.face_jacobian(x, evaluation)
        matrix = sps.csr_array(sps.vstack((momentum, self._auxiliary), format="csr"))
        matrix.eliminate_zeros()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Assembled Jacobian must be finite.")
        return matrix


class AnalyticalJacobian(_TpsaJacobian):
    """Analytical coupled version of Plasticity_Dirichlet_marked.pdf (13), (24), D13.

    With Q = sigma_n + C_e:(epsilon-epsilon_n) - sigma_return, assemble
        J_face = T - N (W kron I_4) blockdiag(C_e-C_alg) B,
        J = [D_u J_face; A_r; A_p].
    B differentiates the Green-Gauss gradient with respect to ALL [u,r,p].
    Interior pressure stabilization and Dirichlet reaction derivatives are kept;
    fixed boundary displacement has zero derivative. Area-weighted normals occur
    exactly once. The elastic trial term T is never cancelled against C_e B.

    The document's condensed displacement Jacobian is the Schur complement of
    this coupled matrix: J_uu + J_ur R + J_up P, with A_rr R=-A_ru and
    A_pp P=-A_pu. Newton continues to solve u,r,p together, without inner
    auxiliary solves or dense inverses. The local tangent differentiates the
    existing 3D J2 return map and supports its differentiable hardening laws.

    Construct once per load step with the SAME operators, weights, models and
    committed state used by the residual. History is copied; geometry, material
    parameters and transfer weights must remain fixed. The current returned
    histories are reused without further return-map calls or state mutations.
    Derivative consistency on a smooth branch does not establish mesh convergence.
    """

    def _material_tangent(
        self, point: MaterialPoint, committed: MaterialState,
        increment: NDArray[np.float64], returned: MaterialState,
    ) -> NDArray[np.float64]:
        return analytical_material_tangent(point, committed, increment, returned=returned)


class FiniteDifferenceJacobian(_TpsaJacobian):
    """Optional local forward differences with the same exact TPSA chain rule.

    Construct once per step with the residual's fixed operators, transfer, models
    and committed history. step is an absolute dimensionless gradient increment.
    Every cell tangent uses five local updates from that same history. All global
    [u,r,p] columns and auxiliary rows are retained. No global finite differences
    or auxiliary solves are performed. Arguments and point histories are preserved.
    Near yield switches this one-sided numerical derivative need not be smooth.
    """

    def __init__(
        self,
        operators: TpsaOperators,
        transfer: CellToFaceTransfer,
        material_points: Sequence[MaterialPoint],
        committed: TpsaState,
        *,
        step: float = 1e-10,
    ) -> None:
        _validate_step(step)
        self.step = step
        super().__init__(operators, transfer, material_points, committed)

    def _material_tangent(
        self, point: MaterialPoint, committed: MaterialState,
        increment: NDArray[np.float64], returned: MaterialState,
    ) -> NDArray[np.float64]:
        return finite_difference_material_tangent(point, committed, increment, step=self.step)
