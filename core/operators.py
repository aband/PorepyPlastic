"""Retained-face layout, kinematics, and fixed traction/accumulation coefficients."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.grid import Grid
from porepy.grids.structured import TensorGrid
from porepy.params.bc import BoundaryConditionVectorial

from utils.reconstruction import _validate_reconstruction_inputs, gradient_green_gauss, plane_strain_from_gradient
from utils.transfer import CellToFaceTransfer


class RetainedFaceOperators:
    """Index independent cell/face increments on a full-Dirichlet 2D tensor grid.

    z = [delta_u(2*nc), delta_r(nc), delta_p(nc), d_free(2*n_free)]. Vector
    components are interleaved by cell/face, as in PorePy (order="F"). The
    free faces are the interior faces, sorted by PorePy face number. Prescribed
    boundary faces have no Newton DOFs; their increments are target minus old
    boundary displacement. All entries in z are increments from the committed
    load step, not absolute fields or corrections from the previous iteration.

    Strain is evaluated from the face increments, independently of z's cell
    fields and material state. Supply a positive scalar or cellwise mu to build
    fixed elastic traction coefficients and the complementary rotation transfer.
    Without mu the object supports kinematics only. Supply lam as well for
    global residual assembly: rotation_accumulation and pressure_accumulation
    store V/mu and V/lam. Both moduli are fixed positive reference elastic
    coefficients, not algorithmic tangents. No material updates occur here.

    Side arrays enumerate cell-face pairs in increasing face order, then cell
    order within each face. side_faces and side_cells map the last axis of a
    (2,num_sides) traction array. side_normals are outward UNIT normals;
    side_areas supplies the face measure exactly once. Traction coefficients
    require orthogonal cell-to-face offsets; rotated rectangular grids are valid.

    Supply computed geometry and the existing BoundaryConditionVectorial, e.g.
    RetainedFaceOperators(case.grid, case.bc) after case.prepare_simulation().
    The grid is shared; coefficients are snapshots. Keep geometry, mu, lam, and
    boundary types fixed and rebuild after changes. Boundary values are supplied
    per call.
    Returned arrays own their storage and inputs are preserved.
    """

    def __init__(
        self, grid: Grid, bc: BoundaryConditionVectorial,
        *, mu: float | NDArray[np.float64] | None = None,
        lam: float | NDArray[np.float64] | None = None,
    ) -> None:
        nc, nf = grid.num_cells, grid.num_faces
        _validate_reconstruction_inputs(grid, np.zeros((2, nc)), np.zeros((2, nf)))
        if not isinstance(grid, TensorGrid):
            raise ValueError("Retained-face kinematics currently requires a tensor-product grid.")
        if not np.allclose(grid.nodes[2], 0.0, rtol=0, atol=1e-12):
            raise ValueError("Retained-face plane strain requires a grid in the xy plane.")
        boundary = np.asarray(grid.get_all_boundary_faces(), dtype=np.int64)
        if bc.is_dir.shape != (2, nf) or not np.all(bc.is_dir[:, boundary]):
            raise ValueError("Retained-face kinematics requires Dirichlet data on every boundary face.")

        free = np.setdiff1d(np.arange(nf, dtype=np.int64), boundary)
        incidence = sps.csr_array(grid.cell_faces, copy=True)
        incidence.sum_duplicates()
        incidence.eliminate_zeros()
        incidence.sort_indices()
        if incidence.shape != (nf, nc):
            raise ValueError("Face-cell connectivity must match the grid.")
        neighbors = np.diff(incidence.indptr)
        if (not np.all(neighbors[boundary] == 1) or not np.all(neighbors[free] == 2)
                or not np.all(np.isin(incidence.data, [-1, 1]))
                or not np.all(np.asarray(incidence.sum(axis=1)).ravel()[free] == 0)):
            raise ValueError("Expected one boundary neighbor and two oppositely oriented interior neighbors.")

        self.grid = grid
        self.num_cells, self.num_faces = nc, nf
        self.boundary_faces = boundary.copy()
        self.free_faces = free
        self.num_free_faces = free.size
        self.num_dofs = 4 * nc + 2 * self.num_free_faces
        self.u_slice = slice(0, 2 * nc)
        self.r_slice = slice(2 * nc, 3 * nc)
        self.p_slice = slice(3 * nc, 4 * nc)
        self.cell_slice = slice(0, 4 * nc)
        self.face_slice = slice(4 * nc, self.num_dofs)
        # face_dofs[component, face] is its global z index, or -1 if prescribed.
        self.face_dofs = np.full((2, nf), -1, dtype=np.int64)
        self.face_dofs[:, free] = np.arange(4 * nc, self.num_dofs, dtype=np.int64).reshape(
            (2, self.num_free_faces), order="F",
        )

        self.side_faces = np.repeat(np.arange(nf, dtype=np.int64), neighbors)
        self.side_cells = np.array(incidence.indices, dtype=np.int64, copy=True)
        self.side_signs = np.array(incidence.data, dtype=np.float64, copy=True)
        self.num_sides = self.side_faces.size
        self.side_areas = grid.face_areas[self.side_faces].copy()
        self.side_normals = grid.face_normals[:2, self.side_faces] * self.side_signs / self.side_areas
        if not np.allclose(np.linalg.norm(self.side_normals, axis=0), 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError("Face normals must have magnitude equal to the face areas.")
        offsets = grid.face_centers[:2, self.side_faces] - grid.cell_centers[:2, self.side_cells]
        self.side_distances = np.sum(offsets * self.side_normals, axis=0)
        if not np.all(np.isfinite(self.side_distances)) or np.any(self.side_distances <= 0):
            raise ValueError("Expected positive outward cell-to-face distances.")

        self.mu: NDArray[np.float64] | None = None
        self.lam: NDArray[np.float64] | None = None
        self.rotation_accumulation: NDArray[np.float64] | None = None
        self.pressure_accumulation: NDArray[np.float64] | None = None
        self.side_stiffness: NDArray[np.float64] | None = None
        self.rotation_transfer: CellToFaceTransfer | None = None
        if mu is not None:
            if not np.allclose(offsets, self.side_normals * self.side_distances, rtol=1e-10, atol=1e-12):
                raise ValueError("Retained-face tractions require orthogonal cell-to-face offsets.")
            self._set_traction_coefficients(mu, incidence)
        if lam is not None:
            if mu is None:
                raise ValueError("Supply mu together with lam for global residual assembly.")
            self.lam = self._cell_coefficient("lam", lam)
            self.pressure_accumulation = self._cell_accumulation(self.lam)

    def _cell_coefficient(
        self, name: str, value: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Validate and copy a positive scalar or cellwise elastic coefficient."""
        values = np.asarray(value, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(self.num_cells, float(values))
        if values.shape != (self.num_cells,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} must be a finite positive scalar or array with shape (num_cells,).")
        return values.copy()

    def _cell_accumulation(self, modulus: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute finite positive cell-volume/modulus accumulation coefficients."""
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            values = self.grid.cell_volumes / modulus
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Cell accumulation coefficients must be finite and positive.")
        return values

    def _set_traction_coefficients(
        self, mu: float | NDArray[np.float64], incidence: sps.csr_array,
    ) -> None:
        """Snapshot a_ka = 2*mu_a/delta_ak and the shared face rotation weights."""
        values = self._cell_coefficient("mu", mu)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            stiffness = 2.0 * values[self.side_cells] / self.side_distances
        if not np.all(np.isfinite(stiffness)) or np.any(stiffness <= 0):
            raise ValueError("Cell-to-face stiffness coefficients must be finite and positive.")
        weights = np.ones(self.num_sides)
        for face in self.free_faces:
            start, end = incidence.indptr[face:face + 2]
            scaled = stiffness[start:end] / np.max(stiffness[start:end])
            # Weight of r_i is a_j/(a_i+a_j), not a_i/(a_i+a_j).
            weights[start:end] = scaled[::-1] / np.sum(scaled)
        matrix = sps.csr_array(
            (weights, self.side_cells, incidence.indptr), shape=incidence.shape,
        )
        self.mu = values
        self.rotation_accumulation = self._cell_accumulation(values)
        self.side_stiffness = stiffness
        self.rotation_transfer = CellToFaceTransfer(self.grid, matrix)

    def pack(
        self, cell_increment: NDArray[np.float64], free_face_increment: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Pack [delta_u,delta_r,delta_p] (4*nc,) and d_free (2,n_free)."""
        if cell_increment.shape != (4 * self.num_cells,) or not np.all(np.isfinite(cell_increment)):
            raise ValueError(f"Expected finite cell_increment with shape ({4 * self.num_cells},).")
        if free_face_increment.shape != (2, self.num_free_faces) or not np.all(np.isfinite(free_face_increment)):
            raise ValueError(f"Expected finite free_face_increment with shape (2, {self.num_free_faces}).")
        return np.concatenate((cell_increment, free_face_increment.ravel(order="F")))

    def unpack(self, z: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return owned cell increments (4*nc,) and free-face increments (2,n_free)."""
        self._validate_unknowns(z)
        return (z[self.cell_slice].copy(),
                z[self.face_slice].reshape((2, self.num_free_faces), order="F").copy())

    def face_displacement_increment(
        self, z: NDArray[np.float64], bc_values: NDArray[np.float64],
        committed_bc_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Form d (2,nf), inserting target-minus-committed Dirichlet values.

        Both boundary arrays have shape (2,nf) and hold absolute displacements.
        Only their boundary entries are used; unused interior entries may be NaN.
        """
        self._validate_unknowns(z)
        bf = self.boundary_faces
        for name, values in (("bc_values", bc_values), ("committed_bc_values", committed_bc_values)):
            if values.shape != (2, self.num_faces) or not np.all(np.isfinite(values[:, bf])):
                raise ValueError(f"Expected {name} (2, {self.num_faces}), finite on boundary faces.")
        d = np.zeros((2, self.num_faces))
        d[:, self.free_faces] = z[self.face_slice].reshape((2, self.num_free_faces), order="F")
        with np.errstate(over="ignore", invalid="ignore"):
            d[:, bf] = bc_values[:, bf] - committed_bc_values[:, bf]
        if not np.all(np.isfinite(d)):
            raise ValueError("Face displacement increments must remain finite.")
        return d

    def strain_increment(
        self, z: NDArray[np.float64], bc_values: NDArray[np.float64],
        committed_bc_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Reconstruct delta_epsilon (3,3,nc) with the existing Green–Gauss helpers."""
        d = self.face_displacement_increment(z, bc_values, committed_bc_values)
        return plane_strain_from_gradient(gradient_green_gauss(self.grid, d))

    def reconstruct_strain(
        self, z: NDArray[np.float64], bc_values: NDArray[np.float64],
        committed_bc_values: NDArray[np.float64], committed_epsilon: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return total strain epsilon_n + delta_epsilon, preserving fixed history.

        committed_epsilon is the previous total plane strain, including elastic
        and plastic contributions. Tensor shear is used; zz, xz, yz are zero.
        """
        if (committed_epsilon.shape != (3, 3, self.num_cells)
                or not np.all(np.isfinite(committed_epsilon))
                or not np.allclose(committed_epsilon, committed_epsilon.swapaxes(0, 1), rtol=1e-10, atol=1e-14)
                or not np.allclose(committed_epsilon[2], 0, atol=1e-14)
                or not np.allclose(committed_epsilon[:, 2], 0, atol=1e-14)):
            raise ValueError("Expected finite symmetric committed total plane strain with shape (3, 3, nc).")
        increment = self.strain_increment(z, bc_values, committed_bc_values)
        with np.errstate(over="ignore", invalid="ignore"):
            total = committed_epsilon + increment
        if not np.all(np.isfinite(total)):
            raise ValueError("Reconstructed total strain must remain finite.")
        return total

    def _validate_unknowns(self, z: NDArray[np.float64]) -> None:
        if z.shape != (self.num_dofs,) or not np.all(np.isfinite(z)):
            raise ValueError(f"Expected finite z with shape ({self.num_dofs},).")
