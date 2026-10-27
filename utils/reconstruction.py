"""Green–Gauss reconstruction and Mandel tensor conventions in 2D and 3D."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from porepy.grids.grid import Grid


def gradient_green_gauss(grid: Grid, u_face: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map face displacements (dim,nf) to gradients (dim,dim,nc).

    Values must be finite on every face, including interior and Neumann faces.
    Supply face averages for the exact surface integral; evaluating an affine
    field at face centers gives these averages. Area-weighted face normals and
    signed cell-face connectivity supply the outward surface measure once.
    The 2D case uses global x/y components on a grid parallel to the xy plane.
    Inputs are preserved and the returned gradient owns its storage.
    """
    validate_reconstruction_geometry(grid)
    nd = grid.dim
    if np.iscomplexobj(u_face):
        raise ValueError("Face displacement must be real.")
    u_face = np.asarray(u_face, dtype=np.float64)
    if u_face.shape != (nd, grid.num_faces) or not np.all(np.isfinite(u_face)):
        raise ValueError(f"Expected finite face displacement with shape ({nd}, nf).")
    gradient = np.empty((nd, nd, grid.num_cells))
    divergence = grid.cell_faces.T
    for i in range(nd):
        for j in range(nd):
            gradient[i, j] = (
                divergence @ (u_face[i] * grid.face_normals[j])
            ) / grid.cell_volumes
    return gradient


def strain_from_gradient(gradient: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetrize (dim,dim,nc) gradients into owned (3,3,nc) tensors.

    Infer dim=2 or dim=3 from the input. For dim=2, embed plane strain with
    zero zz/xz/yz components; dim=3 retains all components. Shears are tensor
    strains, epsilon_ij = (gradient_ij + gradient_ji)/2.
    """
    if np.iscomplexobj(gradient):
        raise ValueError("Displacement gradient must be real.")
    gradient = np.asarray(gradient, dtype=np.float64)
    if (
        gradient.ndim != 3
        or gradient.shape[:2] not in ((2, 2), (3, 3))
        or not np.all(np.isfinite(gradient))
    ):
        raise ValueError(
            "Expected a finite displacement gradient with shape (2, 2, nc) or (3, 3, nc)."
        )
    nd = gradient.shape[0]
    epsilon = np.zeros((3, 3, gradient.shape[2]))
    epsilon[:nd, :nd] = 0.5 * (gradient + gradient.swapaxes(0, 1))
    return epsilon


def plane_strain_from_gradient(gradient: NDArray[np.float64]) -> NDArray[np.float64]:
    """Preserve the 2D-only interface for plane-strain reconstruction."""
    if gradient.ndim != 3 or gradient.shape[:2] != (2, 2):
        raise ValueError("Expected a finite displacement gradient with shape (2, 2, nc).")
    return strain_from_gradient(gradient)


def validate_reconstruction_geometry(grid: Grid) -> None:
    """Check computed 2D/3D geometry without requiring boundary conditions.

    The 2D coordinate convention requires a grid parallel to the xy plane.
    No data are modified. Cached-operator builders can call this once during
    initialization, then rebuild their operators if the geometry changes.
    """
    nc, nf = grid.num_cells, grid.num_faces
    if grid.dim not in (2, 3) or nc < 1 or nf < 1:
        raise ValueError("Expected a nonempty 2D or 3D grid.")
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
            or np.iscomplexobj(values)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                f"Grid {name} must be real and finite with shape {shape}; check compute_geometry()."
            )
    if np.any(grid.cell_volumes <= 0) or np.any(grid.face_areas <= 0):
        raise ValueError("Cell areas and face measures must be positive.")
    incidence = getattr(grid, "cell_faces", None)
    if getattr(incidence, "shape", None) != (nf, nc):
        raise ValueError("Grid cell_faces must have shape (num_faces, num_cells).")
    if np.iscomplexobj(grid.cell_faces.data) or not np.all(np.isfinite(grid.cell_faces.data)):
        raise ValueError("Grid cell_faces must contain real finite values.")
    if grid.dim == 2 and np.any(np.abs(grid.face_normals[2]) > 1e-12 * grid.face_areas):
        raise ValueError("2D reconstruction requires a grid parallel to the xy plane.")


def _validate_reconstruction_inputs(
    grid: Grid,
    u_cell: NDArray[np.float64],
    u_boundary: NDArray[np.float64],
) -> None:
    """Preserve validation of 2D cell and prescribed-boundary displacements.

    Only boundary entries of u_boundary are used; interior entries may be
    nonfinite. Retained-face reconstruction instead requires a full face field.
    """
    nc, nf = grid.num_cells, grid.num_faces
    if grid.dim != 2 or nc < 1 or nf < 1:
        raise ValueError("Expected a nonempty 2D grid.")
    validate_reconstruction_geometry(grid)
    if u_cell.shape != (2, nc) or u_boundary.shape != (2, nf):
        raise ValueError(f"Expected u_cell (2, {nc}) and u_boundary (2, {nf}).")
    if not np.all(np.isfinite(u_cell)):
        raise ValueError("u_cell must contain only finite values.")
    if not np.all(np.isfinite(u_boundary[:, grid.get_all_boundary_faces()])):
        raise ValueError("u_boundary must be finite on all boundary faces.")


SQRT2 = np.sqrt(2.0)


def _mandel_indices(dim: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Tensor entries in the shared 2D/3D Mandel ordering."""
    if dim == 2:
        return (0, 1, 0), (0, 1, 1)
    if dim == 3:
        return (0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)
    raise ValueError("Mandel dimension must be 2 or 3.")


def mandel(
    tensors: NDArray[np.float64], *, dim: int = 2,
) -> NDArray[np.float64]:
    """Pack symmetric tensors cell by cell into a flat Mandel vector.

    Input shape is (3,3,nc); (2,2,nc) is also accepted for dim=2.
    The default dim=2 preserves [xx, yy, sqrt(2)*xy], ignoring out-of-plane
    entries. dim=3 uses [xx, yy, zz, sqrt(2)*yz, sqrt(2)*xz, sqrt(2)*xy],
    matching material.tensor. Symmetry is assumed; upper-triangle entries
    are used. The returned float64 array is independent of the input.
    """
    rows, cols = _mandel_indices(dim)
    if np.iscomplexobj(tensors):
        raise ValueError("Mandel tensors must be real.")
    tensors = np.asarray(tensors, dtype=np.float64)
    if tensors.ndim != 3 or tensors.shape[:2] not in ((dim, dim), (3, 3)):
        raise ValueError(
            f"Expected tensors with shape (3, 3, nc) or ({dim}, {dim}, nc)."
        )
    components = tensors[rows, cols]
    components[dim:] *= SQRT2
    return components.ravel(order="F")


def from_mandel(
    values: NDArray[np.float64], *, dim: int = 2,
) -> NDArray[np.float64]:
    """Unpack a flat cell-major Mandel vector into (3,3,nc) tensors.

    dim=2 expects three components per cell, [xx, yy, sqrt(2)*xy], and
    leaves out-of-plane entries zero. dim=3 expects six components per cell,
    [xx, yy, zz, sqrt(2)*yz, sqrt(2)*xz, sqrt(2)*xy]. The output is symmetric,
    stores tensor shear, and is independent of the input.
    """
    rows, cols = _mandel_indices(dim)
    if np.iscomplexobj(values):
        raise ValueError("Mandel values must be real.")
    values = np.asarray(values, dtype=np.float64)
    num_components = len(rows)
    if values.ndim != 1 or values.size % num_components:
        raise ValueError(
            f"Expected a flat Mandel vector with length {num_components} * nc."
        )
    components = values.reshape((-1, num_components)).T
    epsilon = np.zeros((3, 3, components.shape[1]))
    epsilon[rows[:dim], cols[:dim]] = components[:dim]
    shear = components[dim:] / SQRT2
    epsilon[rows[dim:], cols[dim:]] = shear
    epsilon[cols[dim:], rows[dim:]] = shear
    return epsilon


def mandel_basis(dim: int = 2) -> NDArray[np.float64]:
    """Return an owned orthonormal symmetric basis of shape (m,dim,dim).

    m=3 for dim=2 and m=6 for dim=3. Component ordering and shear scaling
    are shared with mandel()/from_mandel(). Build once when assembling B.
    """
    rows, _ = _mandel_indices(dim)
    tensors = from_mandel(np.eye(len(rows)).ravel(order="F"), dim=dim)
    return np.moveaxis(tensors, 2, 0)[:, :dim, :dim].copy()


# Preserve the original 2D constant for existing solver imports.
MANDEL_BASIS = mandel_basis()
