import numpy as np
import porepy as pp
import scipy.sparse as sps

from core.layout import DofLayout, FloatArray, IntArray

from utils.reconstruction import(
    from_mandel,
    mandel_basis,
    validate_reconstruction_geometry,
)

from utils.transfer import CellToFaceTransfer

class RetainedFaceOperators:
    """Fixed geometry and kinematic operators for retained face unknowns."""

    def __init__(
            self, 
            grid: pp.Grid, 
            layout: DofLayout,
            *,
            mu: FloatArray | None=None,
        ) -> None:
        if (
            grid.dim != layout.dim
            or grid.num_cells != layout.num_cells
            or grid.num_faces != layout.num_faces
        ):
            raise ValueError("Grid and DOF layout are incompatible.")

        self.grid = grid
        self.layout = layout

        # Build and cache the Green–Gauss reconstruction operators.
        self.B = self._build_strain_matrix()

        # Expose B for free and prescribed dofs
        # Components = (0,1) or (0,1,2) for 2D and 3D respectively
        # Adding components selecting all components, since components 
        # are indxed in order
        # None expand dimension for broadcasting use
        components = np.arange(layout.dim, dtype=np.int64)[:, None]

        free_columns = (
            layout.dim * layout.free_faces[None, :] + components
        ).ravel(order="F")

        dirichlet_columns = (
            layout.dim * layout.dirichlet_faces[None, :] + components
        ).ravel(order="F")

        self.B_free      = self.B[:, free_columns]
        self.B_dirichlet = self.B[:, dirichlet_columns]

        self._build_side_geometry()

        self.mu: FloatArray | None = None
        self.side_stiffness: FloatArray | None = None
        self.rotation_transfer: CellToFaceTransfer | None = None

        if mu is not None:
            self._build_traction_coefficients(mu)

    def face_displacement_increment(
        self,
        z: FloatArray,
        dirichlet_increment: FloatArray,
    ) -> FloatArray:
        """Assemble displacement increments on all faces.

        Parameters:
            z:
                Packed load-step increments [delta_u, delta_r, delta_p, d].
                Shape: (layout.size,).

            dirichlet_increment:
                Prescribed displacement increments, ordered according to
                layout.dirichlet_faces.
                Shape: (dim, number of Dirichlet faces).

        Returns:
            Face displacement increments with shape (dim, num_faces).
            Interior and Neumann face values come from the packed increments z.
            Dirichlet face values come from dirichlet_increment.

        All increments are measured from the last accepted load step.
        Inputs are preserved; the returned array owns its storage.
        """

        # Load a 2D or 3D layout
        layout = self.layout

        if np.iscomplexobj(z) or np.iscomplexobj(dirichlet_increment):
            raise ValueError("Displacement increments must be real.")

        # Load and check incremental data
        z = np.asarray(z, dtype=np.float64)
        prescribed = np.asarray(dirichlet_increment, dtype=np.float64)

        if z.shape != (layout.size,):
            raise ValueError(
                f"Solution vector must have shape ({layout.size},)."
            )

        expected_shape = (layout.dim, layout.dirichlet_faces.size)
        if prescribed.shape != expected_shape:
            raise ValueError(
                f"Dirichlet increments must have shape {expected_shape}."
            )

        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(prescribed)):
            raise ValueError("Increments must be finite.")

        # Use a view of the face block; avoid copying unused cell variables.
        free_increment = z[layout.face_slice].reshape(
            (layout.dim, layout.free_faces.size), order="F"
        )

        # The layout partitions all faces into free and prescribed groups.
        face_increment = np.empty(
            (layout.dim, layout.num_faces),
            dtype=np.float64,
            order="F",
        )
        face_increment[:, layout.free_faces] = free_increment
        face_increment[:, layout.dirichlet_faces] = prescribed

        return face_increment

    def _build_strain_matrix(self) -> sps.csr_array:
        """
        Map face displacement increments to strain in Mandel form.

        Input : face_increment.ravel(order="F")
        Output: The reconstruction matrix B
                (3*num_cells, 2*num_faces) for 2D, 
                (6*num_cells, 3*num_faces) for 3D in Mandel form.
        """

        grid = self.grid
        validate_reconstruction_geometry(grid)

        # Create mandel basis
        nd = grid.dim
        basis = mandel_basis(nd)
        num_components = basis.shape[0]

        # Identify one face, adjacent cell and orientation
        incidence = grid.cell_faces.tocoo()
        faces = np.asarray(incidence.row, dtype=np.int64)
        cells = np.asarray(incidence.col, dtype=np.int64)

        # Outward area-weighted normals divided by cell volumes
        # Porepy face normals already include face areas
        weights = grid.face_normals[:nd, faces] * (
            incidence.data / grid.cell_volumes[cells]
        )[None, :]

        # E[a,i,j]: Mandel component a, displacement direction i,
        # and normal direction j.
        a, i, j = np.nonzero(basis)

        values = basis[a, i, j][:, None] * weights[j]
        rows = num_components * cells[None, :] + a[:, None]
        cols = nd * faces[None, :] + i[:, None]

        matrix = sps.coo_array(
            (
                values.ravel(),
                (rows.ravel(), cols.ravel()),
            ),
            shape=(
                num_components * grid.num_cells,
                nd * grid.num_faces,
            ),
        ).tocsr()

        matrix.eliminate_zeros()
        return matrix

    def _build_side_geometry(
            self
        ) -> None:

        """Create one-sided geometry view."""
        grid = self.grid
        nd, nf = self.layout.dim, self.layout.num_faces

        # Extract incidence
        incidence = sps.csr_array(grid.cell_faces, copy=True)
        incidence.sum_duplicates()
        incidence.eliminate_zeros()
        incidence.sort_indices()

        neighbors = np.diff(incidence.indptr)

        # Mark all geometric boundary faces
        boundary = np.zeros(nf, dtype=bool)
        boundary[grid.get_all_boundary_faces()] = True
        interior = ~boundary

        sign_sum = np.asarray(incidence.sum(axis=1)).ravel()
        if (
            # Check all values are either -1 or 1
            not np.all(np.isin(incidence.data, [-1,1]))
            or not np.all(neighbors[boundary] == 1)
            or not np.all(neighbors[interior] == 2)
            or not np.all(sign_sum[interior] == 0)
        ): 
            raise ValueError(
                "Expected one side per boundary faces and two for interior faces."
                "Interior faces are installed with correct orientation."
            )
        
        self.side_faces = np.repeat(
            np.arange(nf, dtype=np.int64), neighbors
        )
        self.side_cells = incidence.indices.astype(np.int64, copy=True)
        self.side_signs = incidence.data.astype(np.float64, copy=True)
        self.num_sides = self.side_faces.size

        # Create area and weighted normal array
        self.side_areas = grid.face_areas[self.side_faces]
        self.side_normals = (
            grid.face_normals[:nd, self.side_faces]
            * self.side_signs[None, :]
            / self.side_areas[None, :]
        )

        if not np.allclose(
            np.linalg.norm(self.side_normals, axis=0),
            1.0,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "Face-noraml magnitudes must equal face measures."
            )

        offsets = (
            grid.face_centers[:nd, self.side_faces]
            - grid.cell_centers[:nd, self.side_cells]
        )

        # Match Porepy's convention
        self.side_distances = np.abs(
            np.sum(offsets * self.side_normals, axis=0)
        )

        if(
            not np.all(np.isfinite(self.side_distances))
            or np.any(self.side_distances<=0)
        ):
            raise ValueError(
                "Projected cell-to-face distances must be finite and positive."
            )

    def _build_traction_coefficients(
            self,
            mu:FloatArray
    )->None:
        """Build the fixed elastic coefficients and rotationa-transfer weights."""
        nc, nf = self.layout.num_cells, self.layout.num_faces

        if np.iscomplexobj(mu):
            raise ValueError("Shear modulus must be real.")

        values = np.asarray(mu, dtype=np.float64, copy=True)
        if values.shape != (nc,):
            raise ValueError(f"Shear modulus must have the shape ({nc},).")

        if not np.all(np.isfinite(values)) or np.any(values<=0):
            raise ValueError("Stiffness must be finite and positive.")

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            stiffness=(
                2.0*values[self.side_cells]/self.side_distances
            )

        if not np.all(np.isfinite(stiffness)) or np.any(stiffness<=0):
            raise ValueError("Stiffness must be finite and positive.")

        # Follow the side geometry view
        counts = np.bincount(self.side_faces, minlength=nf)
        indptr = np.empty(nf+1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])

        # Boundary Faces use single adjacent cell with weight one
        weights = np.ones(self.num_sides, dtype=np.float64)

        # Locate two sides of every interior face.
        first  = indptr[:-1][counts==2]
        second = first + 1

        # Scale before summing to avoid overflow
        scale = np.maximum(stiffness[first], stiffness[second])
        a_first = stiffness[first] / scale
        a_second = stiffness[second] / scale
        denominator = a_first + a_second

        # Complemenary weights, \tilde{\Xi}
        weights[first] = a_second / denominator
        weights[second]= a_first / denominator

        matrix = sps.csr_array(
            (weights, self.side_cells, indptr),
            shape=(nf, nc),
        )
        transfer = CellToFaceTransfer(self.grid, matrix)

        self.mu = values
        self.side_stiffness = stiffness
        self.rotation_transfer = transfer

    def green_gauss_reconstruction(
        self, 
        face_increment:FloatArray
    ) -> FloatArray:
        """Reconstruct strain from face displacement d."""

        if np.iscomplexobj(face_increment):
            raise ValueError("Face displacement increment must be real value.")

        values = np.asarray(face_increment, dtype=np.float64)
        expected_shape = (self.layout.dim, self.layout.num_faces)

        if values.shape != expected_shape:
            raise ValueError(
                f"Face displacement increments must have shape "
                f"{expected_shape}; got {values.shape} instead."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError("Face displacement increments must be finite.")

        strain_increment = self.B @ values.ravel(order="F") 
        
        return from_mandel(strain_increment, dim=self.layout.dim)

    def elastic_one_sided_traction(
            self,
            z: FloatArray,
            dirichlet_increment: FloatArray
    )->FloatArray:
        """Return outward integrated traction increments for at each loading."""

        stiffness = self.side_stiffness
        transfer  = self.rotation_transfer

        if stiffness is None or transfer is None:
            raise ValueError(
                "Supply my when constructing operators ot evaluate traction."
            )

        face_increment = self.face_displacement_increment(
            z,
            dirichlet_increment
        )
        values = np.asarray(z, dtype=np.float64)

        nd, nc = self.layout.dim, self.layout.num_cells

        displacement = values[self.layout.u_slice].reshape(
            (nd, nc), order="F"
        )

        rotation = values[self.layout.r_slice].reshape(
            (self.layout.rotation_dim, nc), order="F"
        )

        pressure = values[self.layout.p_slice]

        # Complementary interior weights
        face_rotation = transfer.apply(rotation)

        faces = self.side_faces
        cells = self.side_cells
        normals = self.side_normals
        side_rotation = face_rotation[:, faces]

        with np.errstate(over="ignore", invalid="ignore"):
            if nd == 2:
                rotation_traction = (
                    np.vstack((-normals[1], normals[0]))
                    * side_rotation[0][None,:]
                )
            else:
                rotation_traction = np.cross(
                    side_rotation, normals, axis=0
                )

            traction = self.side_areas[None, :] * (
                stiffness[None, :]
                * (face_increment[:, faces] - displacement[:, cells])
                + rotation_traction
                + pressure[cells][None, :] * normals
            )

        if not np.all(np.isfinite(traction)):
            raise ValueError("Elastic traction increments must be finite.")

        return np.asarray(traction, dtype=np.float64)

