from dataclasses import dataclass

import numpy as np
import porepy as pp
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class DofLayout:
    """DOF layout for cell variables and retained face displacements."""

    dim: int
    rotation_dim: int
    num_cells: int
    num_faces: int

    cell_size: int
    size: int

    u_slice: slice
    r_slice: slice
    p_slice: slice
    face_slice: slice

    dirichlet_faces: IntArray
    free_faces: IntArray

    # Global indices in z; prescribed face components have index -1.
    face_dofs: IntArray

    @property
    def cell_slice(self) -> slice:
        return slice(0, self.cell_size)

    @classmethod
    def from_grid(
        cls,
        grid: pp.Grid,
        bc: pp.BoundaryConditionVectorial,
    ) -> "DofLayout":
        """Build numbering for full-vector Dirichlet/Neumann conditions."""
        if grid.dim not in (2, 3):
            raise ValueError("Only 2D and 3D grids are supported.")

        if bc.dim != grid.dim or bc.num_faces != grid.num_faces:
            raise ValueError("Boundary conditions do not match the grid.")

        nd = grid.dim
        nr = 1 if nd == 2 else 3
        nc, nf = grid.num_cells, grid.num_faces

        boundary = np.asarray(
            grid.get_all_boundary_faces(), dtype=np.int64
        )

        dir_flags = bc.is_dir[:, boundary]
        neu_flags = bc.is_neu[:, boundary]

        full_dir = np.all(dir_flags & ~neu_flags, axis=0)
        full_neu = np.all(neu_flags & ~dir_flags, axis=0)

        if (
            np.any(bc.is_rob[:, boundary])
            or not np.all(full_dir | full_neu)
        ):
            raise ValueError(
                "Only full-vector Dirichlet and Neumann conditions "
                "are supported."
            )

        dirichlet_faces = np.sort(boundary[full_dir])

        # Interior and Neumann faces retain displacement unknowns.
        free_faces = np.setdiff1d(
            np.arange(nf, dtype=np.int64), dirichlet_faces
        )

        u_stop = nd * nc
        r_stop = u_stop + nr * nc
        cell_size = r_stop + nc
        size = cell_size + nd * free_faces.size

        face_dofs = np.full((nd, nf), -1, dtype=np.int64)
        face_dofs[:, free_faces] = np.arange(
            cell_size, size, dtype=np.int64
        ).reshape((nd, free_faces.size), order="F")

        return cls(
            dim=nd,
            rotation_dim=nr,
            num_cells=nc,
            num_faces=nf,
            cell_size=cell_size,
            size=size,
            u_slice=slice(0, u_stop),
            r_slice=slice(u_stop, r_stop),
            p_slice=slice(r_stop, cell_size),
            face_slice=slice(cell_size, size),
            dirichlet_faces=dirichlet_faces,
            free_faces=free_faces,
            face_dofs=face_dofs,
        )

    def pack(
        self,
        cell_values: FloatArray,
        face_values: FloatArray,
    ) -> FloatArray:
        """Pack x=[u,r,p] and free-face values into z=[x,d].

        face_values has shape (dim, number of free faces), with
        columns ordered according to self.free_faces.
        """
        if cell_values.shape != (self.cell_size,):
            raise ValueError(
                f"Cell values must have shape ({self.cell_size},)."
            )

        face_shape = (self.dim, self.free_faces.size)
        if face_values.shape != face_shape:
            raise ValueError(
                f"Free-face values must have shape {face_shape}."
            )

        return np.concatenate((
            cell_values,
            face_values.ravel(order="F"),
        ))

    def unpack(self, z: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return independent copies of cell and free-face values."""
        if z.shape != (self.size,):
            raise ValueError(
                f"Solution vector must have shape ({self.size},)."
            )

        cell_values = z[self.cell_slice].copy()
        face_values = z[self.face_slice].reshape(
            (self.dim, self.free_faces.size), order="F"
        ).copy()

        return cell_values, face_values