"""Green–Gauss plane-strain reconstruction and in-plane Mandel conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from porepy.grids.grid import Grid


def gradient_green_gauss(grid: Grid, u_face: NDArray[np.float64]) -> NDArray[np.float64]:
    """Integrate finite face displacement on computed geometry; return (2, 2, nc)."""
    if u_face.shape != (2, grid.num_faces) or not np.all(np.isfinite(u_face)):
        raise ValueError("Expected finite face displacement with shape (2, nf).")
    gradient = np.empty((2, 2, grid.num_cells))
    for i in range(2):
        for j in range(2):
            gradient[i, j] = (
                grid.cell_faces.T @ (u_face[i] * grid.face_normals[j])
            ) / grid.cell_volumes
    return gradient


def plane_strain_from_gradient(gradient: NDArray[np.float64]) -> NDArray[np.float64]:
    """Embed sym(grad u) in 3D with zero out-of-plane total strain and tensor shear."""
    if gradient.ndim != 3 or gradient.shape[:2] != (2, 2) or not np.all(np.isfinite(gradient)):
        raise ValueError("Expected a finite displacement gradient with shape (2, 2, nc).")
    epsilon = np.zeros((3, 3, gradient.shape[2]))
    epsilon[:2, :2] = 0.5 * (gradient + gradient.swapaxes(0, 1))
    return epsilon


def _validate_reconstruction_inputs(
    grid: Grid,
    u_cell: NDArray[np.float64],
    u_boundary: NDArray[np.float64],
) -> None:
    """Check computed geometry and (2, n) displacements; raise ValueError if invalid.

    Assumes Dirichlet values on every boundary face; ignores interior entries
    of u_boundary. Inputs are unchanged.
    """
    nc, nf = grid.num_cells, grid.num_faces
    if grid.dim != 2 or nc < 1 or nf < 1:
        raise ValueError("Expected a nonempty 2D grid.")

    for name, shape in {
        "cell_volumes": (nc,),
        "cell_centers": (3, nc),
        "face_areas": (nf,),
        "face_centers": (3, nf),
        "face_normals": (3, nf),
    }.items():
        values = getattr(grid, name, None)
        if (
            not isinstance(values, np.ndarray)
            or values.shape != shape
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                f"Grid {name} must be finite with shape {shape}; check compute_geometry()."
            )
    if np.any(grid.cell_volumes <= 0) or np.any(grid.face_areas <= 0):
        raise ValueError("Cell areas and face measures must be positive.")

    if u_cell.shape != (2, nc) or u_boundary.shape != (2, nf):
        raise ValueError(f"Expected u_cell (2, {nc}) and u_boundary (2, {nf}).")
    if not np.all(np.isfinite(u_cell)):
        raise ValueError("u_cell must contain only finite values.")
    if not np.all(np.isfinite(u_boundary[:, grid.get_all_boundary_faces()])):
        raise ValueError("u_boundary must be finite on all boundary faces.")


SQRT2 = np.sqrt(2.0)
MANDEL_BASIS = np.zeros((3, 2, 2))
MANDEL_BASIS[0, 0, 0] = MANDEL_BASIS[1, 1, 1] = 1.0
MANDEL_BASIS[2, 0, 1] = MANDEL_BASIS[2, 1, 0] = 1 / SQRT2


def mandel(tensors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pack the in-plane symmetric block of (3,3,nc) tensors, cell by cell."""
    return np.asarray(np.vstack((tensors[0, 0], tensors[1, 1], SQRT2 * tensors[0, 1]))
                      .ravel(order="F"), dtype=np.float64)


def from_mandel(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Embed cell-major Mandel strain in 3D plane strain."""
    components = values.reshape((-1, 3)).T
    epsilon = np.zeros((3, 3, components.shape[1]))
    epsilon[0, 0], epsilon[1, 1] = components[:2]
    epsilon[0, 1] = epsilon[1, 0] = components[2] / SQRT2
    return epsilon
