"""Monolithic plane-strain TPSA plasticity, Eqs. (4.7), (6.7), (7.2) of the
8 October 2026 complementary-weights formulation.

Unknown ordering: [u(2*nc), r(nc), p(nc), epsilon_M(3*nc)]. Symmetric tensors
use cell-major Mandel entries [xx, yy, sqrt(2)*xy]. Material histories remain
full 3D. State.x retains [u,r,p]; state.epsilon and state.face_displacement store
the additional unknowns and consistent shared face history.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.structured import TensorGrid

from alternatives.original.residual import TpsaOperators, TpsaResidual
from utils.state import TpsaState
from utils.material_evaluation import evaluate_material_at_strain
from utils.transfer import CellToFaceTransfer
from material.material_state import MaterialPoint

from utils.reconstruction import (
    SQRT2 as SQRT2,
    MANDEL_BASIS as MANDEL_BASIS,
    mandel as mandel,
    from_mandel as from_mandel,
)

if TYPE_CHECKING:
    from alternatives.plane_strain import PlaneStrainTpsa

class CoupledOperators(TpsaOperators):
    """Fixed Cartesian TPSA maps, complementary transfer, and plastic feedback.

    All normals keep PorePy's orientation, including inward-oriented boundary
    normals. cell_faces supplies the outward signs. Unit normals enter H_f;
    area-weighted normals enter integrated fluxes exactly once. No tangent
    moduli enter the weights. Rebuild after any geometry/material/type change.
    """

    def __init__(self, case: PlaneStrainTpsa) -> None:
        super().__init__(case)
        g, nc, nf = self.grid, self.num_cells, self.num_faces
        if not isinstance(g, TensorGrid):
            raise ValueError("Coupled reconstruction requires a Cartesian tensor-product grid.")
        self.mu = g.cell_volumes / self.accum.diagonal()[2 * nc:3 * nc]
        self.lam = g.cell_volumes / self.accum.diagonal()[3 * nc:]
        self.normal = g.face_normals[:2] / g.face_areas
        cf = sps.csr_array(g.cell_faces)
        counts = np.diff(cf.indptr)
        if not np.all(np.isin(counts, [1, 2])):
            raise ValueError("Each coupled face must have one or two physical neighbors.")
        # Centroid/orthogonality checks enforce the reconstruction's stated scope.
        cn, fn = g.cell_nodes().tocsc(), g.face_nodes.tocsc()
        for points, connectivity, expected_count in ((g.cell_centers, cn, 4), (g.face_centers, fn, 2)):
            for index in range(points.shape[1]):
                nodes = g.nodes[:, connectivity.indices[connectivity.indptr[index]:connectivity.indptr[index + 1]]]
                if nodes.shape[1] != expected_count or not np.allclose(points[:, index], nodes.mean(axis=1), rtol=1e-12, atol=1e-12):
                    raise ValueError("Coupled reconstruction requires cell and face centroids.")
        if (not np.allclose(g.nodes[2], 0, atol=1e-12)
                or not np.allclose(np.sum(np.abs(self.normal), axis=0), 1, atol=1e-12)):
            raise ValueError("Coupled reconstruction requires Cartesian faces in the xy plane.")

        U = sps.lil_array((2 * nf, 4 * nc))
        Hf = sps.lil_array((2 * nf, 3 * nc))
        W = sps.lil_array((nf, nc))
        G = sps.lil_array((3 * nc, 2 * nf))
        self.delta_mu = np.zeros(nf)
        for face in range(nf):
            start, end = cf.indptr[face:face + 2]
            cells, signs = cf.indices[start:end], cf.data[start:end]
            n = self.normal[:, face]
            offsets = g.face_centers[:2, face, None] - g.cell_centers[:2, cells]
            distances = np.sum(offsets * n[:, None], axis=0) * signs
            if (np.any(distances <= 0) or not np.all(np.isfinite(distances))
                    or not np.allclose(offsets, n[:, None] * (signs * distances), rtol=1e-10, atol=1e-12)):
                raise ValueError("Coupled reconstruction requires positive orthogonal cell-to-face distances.")
            N = np.array([[n[0], 0, n[1] / SQRT2], [0, n[1], n[0] / SQRT2]])
            a = 2 * self.mu[cells] / distances
            if len(cells) == 2:
                self.delta_mu[face] = 1 / np.sum(a)
                for cell, sign, weight in zip(cells, signs, a / np.sum(a), strict=True):
                    U[2 * face:2 * face + 2, 2 * cell:2 * cell + 2] = weight * np.eye(2)
                    U[2 * face:2 * face + 2, 3 * nc + cell] = (-self.delta_mu[face] * sign * n)[:, None]
                    Hf[2 * face:2 * face + 2, 3 * cell:3 * cell + 3] = self.delta_mu[face] * sign * N
                    W[face, cell] = 1 - weight
            else:
                W[face, cells[0]] = 1.0
            for cell, sign in zip(cells, signs, strict=True):
                nx, ny = sign * g.face_normals[:2, face] / g.cell_volumes[cell]
                G[3 * cell:3 * cell + 3, 2 * face:2 * face + 2] = np.array([
                    [nx, 0], [0, ny], [ny / SQRT2, nx / SQRT2],
                ])
        self.U, self.H_f, self.G = U.tocsr(), Hf.tocsr(), G.tocsr()
        boundary_mask = np.zeros(nf)
        boundary_mask[self.boundary_faces] = 1
        self.U_g = sps.diags(np.repeat(boundary_mask, 2), format="csr")
        self.transfer = CellToFaceTransfer(g, W.tocsr())
        self.B = sps.csr_array(self.G @ self.U)
        self.H_epsilon = sps.csr_array(self.G @ self.H_f)
        # Rotation/mass fluxes from one shared displacement per face.
        rows = np.repeat(np.arange(nf), 2)
        cols = np.arange(2 * nf)
        nx, ny = g.face_normals[:2]
        self.rotation_flux = sps.csr_array((np.vstack((ny, -nx)).ravel("F"), (rows, cols)), shape=(nf, 2 * nf))
        self.mass_flux = sps.csr_array((np.vstack((nx, ny)).ravel("F"), (rows, cols)), shape=(nf, 2 * nf))
        normals = [np.array([[x, 0, y / SQRT2], [0, y, x / SQRT2]]) for x, y in zip(nx, ny, strict=True)]
        self.H_t = sps.csr_array(-sps.block_diag(normals, format="csr") @ sps.kron(self.transfer.weights, sps.eye(3)))
        self.D = sps.csr_array(g.cell_faces.T)
        self.H = sps.csr_array(sps.vstack((self.D_u @ self.H_t,
            self.D @ self.rotation_flux @ self.H_f, self.D @ self.mass_flux @ self.H_f)))
        # Verify that the reference blocks really implement the chosen alpha=0 law.
        for map_, reference, boundary_reference in (
            (self.rotation_flux, self.face_discretization[2 * nf:3 * nf], self.rhs_matrix[2 * nf:3 * nf]),
            (self.mass_flux, self.face_discretization[3 * nf:], self.rhs_matrix[3 * nf:]),
        ):
            for actual, expected in ((map_ @ self.U, reference), (map_ @ self.U_g, boundary_reference)):
                difference = sps.csr_array(actual - expected)
                error = np.max(np.abs(difference.data), initial=0.0)
                scale = np.max(np.abs(expected.data), initial=0.0)
                if error > 1e-12 + 1e-10 * scale:
                    raise ValueError("Reference TPSA auxiliary maps do not match the coupled alpha=0 formulation.")

    def unknowns(self, state: TpsaState) -> NDArray[np.float64]:
        """Pack mechanical unknowns and independent total strain."""
        return np.asarray(np.r_[state.x, mandel(state.epsilon)], dtype=np.float64)

    def validate_committed(self, state: TpsaState, points: Sequence[MaterialPoint]) -> None:
        """Check consistent stored face history and matching elastic coefficients."""
        if state.face_displacement is None:
            raise ValueError("Coupled states require committed face_displacement; use TpsaState.zeros(grid, store_faces=True).")
        if state.face_displacement.shape != (2, self.num_faces) or not np.all(np.isfinite(state.face_displacement)):
            raise ValueError("Invalid committed face_displacement.")
        if not np.allclose(state.face_displacement[:, self.boundary_faces], state.bc_values[:, self.boundary_faces], rtol=1e-10, atol=1e-12):
            raise ValueError("Committed face displacement must match its Dirichlet data.")
        if not np.allclose(self.G @ state.face_displacement.ravel("F"), mandel(state.epsilon), rtol=1e-8, atol=1e-10):
            raise ValueError("Committed strain must match the stored face displacement reconstruction.")
        if len(points) != self.num_cells or len(state.material_states) != self.num_cells:
            raise ValueError("Expected one material point and history per cell.")
        for cell, point in enumerate(points):
            if not np.allclose([point.material.isotropic_shear_modulus, point.material.lame_parameter],
                               [self.mu[cell], self.lam[cell]], rtol=1e-12, atol=0):
                raise ValueError("Material elastic coefficients must match the coupled TPSA operators.")

    def validate_transfer(self, transfer: CellToFaceTransfer) -> None:
        if (transfer.num_cells, transfer.num_faces) != (self.num_cells, self.num_faces):
            raise ValueError("Transfer must match coupled operators.")
        diff = transfer.weights - self.transfer.weights
        if np.max(np.abs(diff.data), initial=0.0) > 1e-12:
            raise ValueError("Coupled formulation requires its complementary TPSA transfer weights.")


def assemble_coupled_residual(
    operators: CoupledOperators, state: TpsaState, *, sources: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Absolute balances plus reconstruction, including any committed imbalance."""
    nc = operators.num_cells
    load = np.zeros(4 * nc) if sources is None else np.asarray(sources, dtype=np.float64)
    if load.shape != (4 * nc,) or not np.all(np.isfinite(load)):
        raise ValueError(f"Expected finite sources with shape ({4 * nc},).")
    if state.face_displacement is None:
        raise ValueError("Coupled assembly requires face_displacement.")
    face = state.face_displacement.ravel("F")
    return np.asarray(np.r_[
        operators.D_u @ state.traction.ravel("F") - load[:2 * nc],
        operators.D @ (operators.rotation_flux @ face) - operators.accum.diagonal()[2 * nc:3 * nc] * state.x[2 * nc:3 * nc] - load[2 * nc:3 * nc],
        operators.D @ (operators.mass_flux @ face) - operators.accum.diagonal()[3 * nc:] * state.x[3 * nc:] - load[3 * nc:],
        mandel(state.epsilon) - operators.G @ face,
    ], dtype=np.float64)


def evaluate_coupled_residual(
    operators: CoupledOperators, transfer: CellToFaceTransfer,
    material_points: Sequence[MaterialPoint], committed: TpsaState,
    z: NDArray[np.float64], bc_values: NDArray[np.float64], *,
    sources: NDArray[np.float64] | None = None,
) -> TpsaResidual:
    """Evaluate all four equations from independent strain and fixed history.

    Face displacement and traction are updated incrementally from stored values.
    MaterialPoint committed/trial slots are never modified. Boundary data are
    prescribed exactly; face reactions remain functions of the candidate state.
    """
    nc, nf = operators.num_cells, operators.num_faces
    if z.shape != (7 * nc,) or not np.all(np.isfinite(z)):
        raise ValueError(f"Expected finite coupled unknowns with shape ({7 * nc},).")
    operators.validate_transfer(transfer)
    if committed.face_displacement is None:
        raise ValueError("Coupled evaluation requires committed face_displacement.")
    # Reuse the reference validation and mask unused interior boundary entries.
    operators.assemble_rhs(bc_values, sources=sources)
    bf = operators.boundary_faces
    boundary = np.zeros((2, nf))
    boundary[:, bf] = bc_values[:, bf]
    x, epsilon = z[:4 * nc], from_mandel(z[4 * nc:])
    material = evaluate_material_at_strain(operators, material_points, committed, epsilon)
    q = mandel(material.stress_correction)
    dx, dg = x - committed.x, (boundary - committed.bc_values).ravel("F")
    face = (committed.face_displacement.ravel("F") + operators.U @ dx + operators.U_g @ dg + operators.H_f @ q).reshape((2, nf), order="F")
    face[:, bf] = boundary[:, bf]
    traction = (committed.traction.ravel("F") + operators.T @ dx + operators.T_g @ dg + operators.H_t @ q).reshape((2, nf), order="F")
    trial = TpsaState(x, boundary, epsilon, material.material_states, traction, face_displacement=face)
    return TpsaResidual(assemble_coupled_residual(operators, trial, sources=sources), trial, material.stress_correction)
