"""TPSA face displacement and Green–Gauss plane-strain reconstruction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.grid import Grid

from utils.reconstruction import (
    gradient_green_gauss as gradient_green_gauss,
    plane_strain_from_gradient as plane_strain_from_gradient,
    _validate_reconstruction_inputs as _validate_reconstruction_inputs,
)


if TYPE_CHECKING:
    from alternatives.plane_strain import PlaneStrainTpsa


class TpsaPostprocessing:
    """Data for reconstructing a solved 2D TPSA case with Dirichlet boundaries.

    Grid, matrices, boundary conditions, and solution arrays are shared with the
    supplied case. u_cell (2, nc), r (nc,), and p (nc,) are views of x;
    u_boundary and bc_values (2, nf) share the flattened bound_vec storage.
    Create a new instance after solving the case again. Reconstruction methods
    refresh u_face, grad_u, and epsilon from the shared inputs; these results are
    None until computed and must be recomputed after an in-place input change.
    """

    def __init__(self, case: PlaneStrainTpsa) -> None:
        x = case.x
        if x is None:
            raise RuntimeError("Call case.solve() before creating postprocessing.")
        self.grid = case.grid
        nc = self.grid.num_cells
        if x.shape != (4 * nc,) or not np.all(np.isfinite(x)):
            raise ValueError(f"Expected a finite TPSA solution with shape ({4 * nc},).")

        self.x = x
        self.d = case.d
        self.matrices = case.matrices
        self.face_discretization = case.face_discretization
        self.rhs_matrix = case.rhs_matrix
        self.div = case.div
        self.accum = case.accum
        self.bc = case.bc
        self.bc_values = case.bc_values
        self.bound_vec = case.bound_vec
        self.boundary_faces = self.grid.get_all_boundary_faces()
        self.u_cell = x[:2 * nc].reshape((2, nc), order="F")
        self.r = x[2 * nc:3 * nc]
        self.p = x[3 * nc:]
        self.u_boundary = self.bc_values

        self._validate_state()
        self.u_face: NDArray[np.float64] | None = None
        self.grad_u: NDArray[np.float64] | None = None
        self.epsilon: NDArray[np.float64] | None = None

    def reconstruct_face_displacement(self) -> NDArray[np.float64]:
        """Recover (2, nf) displacement using TPSA's integrated rotation/mass fluxes.

        With unit normal n and tangent t = (n_y, -n_x), u_f = (v*n + tau*t)/A.
        This retains the material-weighted average and pressure-jump correction.
        Dirichlet faces use the prescribed values. Each call recomputes from x
        and boundary data, and clears the previous gradient and strain results.
        """
        self.u_face = self.grad_u = self.epsilon = None
        self._validate_state()
        self.u_face = reconstruct_face_displacement(
            self.grid, self.face_discretization, self.rhs_matrix, self.x, self.u_boundary,
        )
        return self.u_face

    def gradient_green_gauss(self) -> NDArray[np.float64]:
        """Recompute G[i, j, cell] = du_i/dx_j, with shape (2, 2, nc)."""
        u_face = self.reconstruct_face_displacement()
        self.grad_u = gradient_green_gauss(self.grid, u_face)
        return self.grad_u

    def strain_green_gauss(self) -> NDArray[np.float64]:
        """Recompute total plane strain as (3, 3, nc), using tensor shear strain.

        The zz, xz, and yz components are zero. This is total strain; a material
        update later needs its increment from the last converged load step.
        """
        gradient = self.gradient_green_gauss()
        self.epsilon = plane_strain_from_gradient(gradient)
        return self.epsilon

    def _validate_state(self) -> None:
        """Check the current shared inputs before reconstruction."""
        if not np.all(np.isfinite(self.x)):
            raise ValueError("The TPSA solution must contain only finite values.")
        if not np.all(self.bc.is_dir[:, self.boundary_faces]):
            raise ValueError("Postprocessing requires Dirichlet data on every boundary face.")
        _validate_reconstruction_inputs(self.grid, self.u_cell, self.u_boundary)


def reconstruct_face_displacement(
    grid: Grid,
    face_discretization: sps.spmatrix | sps.sparray,
    rhs_matrix: sps.spmatrix | sps.sparray,
    x: NDArray[np.float64],
    bc_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Recover face displacement from a candidate [u, r, p] without a solved case.

    Assumes full Dirichlet boundaries. Only boundary entries of bc_values are
    used. Matrices must correspond to the supplied grid and boundary types.
    The result owns its storage; all inputs remain unchanged.
    """
    nc, nf = grid.num_cells, grid.num_faces
    if x.shape != (4 * nc,) or not np.all(np.isfinite(x)):
        raise ValueError(f"Expected a finite TPSA solution with shape ({4 * nc},).")
    _validate_reconstruction_inputs(
        grid, x[:2 * nc].reshape((2, nc), order="F"), bc_values,
    )
    if face_discretization.shape != (4 * nf, 4 * nc) or rhs_matrix.shape != (4 * nf, 2 * nf):
        raise ValueError("TPSA reconstruction matrices must match the grid.")
    bf = grid.get_all_boundary_faces()
    boundary = np.zeros((2, nf))
    boundary[:, bf] = bc_values[:, bf]
    # Skip traction rows; normals already contain face measures.
    flux = np.asarray(
        face_discretization[2 * nf:] @ x + rhs_matrix[2 * nf:] @ boundary.ravel(order="F"),
        dtype=np.float64,
    )
    tau, normal_flux = flux[:nf], flux[nf:]
    normal = grid.face_normals[:2] / grid.face_areas
    tangent = np.vstack((normal[1], -normal[0]))
    u_face = np.asarray(
        (normal * normal_flux + tangent * tau) / grid.face_areas, dtype=np.float64,
    )
    u_face[:, bf] = boundary[:, bf]
    return u_face
