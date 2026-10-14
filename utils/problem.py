"""Shared plane-strain geometry, material points, and reference Dirichlet data."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from porepy.grids.grid import Grid
from porepy.grids.structured import CartGrid
from porepy.params.bc import BoundaryConditionVectorial

from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import HardeningParameters, isotropic_hardening_K, kinematic_hardening_H
from material.tensor import strain

BoundaryDisplacement: TypeAlias = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class PlaneStrainProblem:
    """Unit-square problem data without elastic TPSA matrix assembly.

    prepare_simulation creates a fresh grid, virgin material histories, and
    full-Dirichlet data at load factor one. The default affine displacement has
    an elastic reference solution. A custom boundary_displacement callback is
    evaluated once per preparation and supplies boundary data only.
    """

    def __init__(
        self,
        *,
        cells_per_axis: int = 8,
        displacement_gradient: ArrayLike | None = None,
        translation: ArrayLike | None = None,
        boundary_displacement: BoundaryDisplacement | None = None,
        young_modulus: float = 210.0e9,
        poisson_ratio: float = 0.3,
    ) -> None:
        if (
            isinstance(cells_per_axis, bool)
            or not isinstance(cells_per_axis, (int, np.integer))
            or cells_per_axis < 1
        ):
            raise ValueError("cells_per_axis must be a positive integer.")
        if not np.isfinite(young_modulus) or young_modulus <= 0:
            raise ValueError("young_modulus must be finite and positive.")
        if not np.isfinite(poisson_ratio) or not 0 < poisson_ratio < 0.5:
            raise ValueError("This baseline requires 0 < poisson_ratio < 0.5.")

        if boundary_displacement is not None:
            if not callable(boundary_displacement):
                raise ValueError("boundary_displacement must be callable or None.")
            if displacement_gradient is not None or translation is not None:
                raise ValueError(
                    "boundary_displacement cannot be combined with displacement_gradient or translation."
                )
        self._boundary_displacement = boundary_displacement

        self.displacement_gradient: NDArray[np.float64] = np.array(
            [[1.0e-4, 0.0], [0.0, 0.0]]
            if displacement_gradient is None else displacement_gradient,
            dtype=np.float64,
            copy=True,
        )
        self.translation: NDArray[np.float64] = np.array(
            [0.0, 0.0] if translation is None else translation,
            dtype=np.float64,
            copy=True,
        )
        if self.displacement_gradient.shape != (2, 2) or not np.all(
            np.isfinite(self.displacement_gradient)
        ):
            raise ValueError("displacement_gradient must be a finite 2-by-2 array.")
        if self.translation.shape != (2,) or not np.all(np.isfinite(self.translation)):
            raise ValueError("translation must be a finite two-component vector.")

        self.x: NDArray[np.float64] | None = None
        self.cells_per_axis = int(cells_per_axis)
        self.material = _materialProperty()
        self.material.young_modulus = young_modulus
        self.material.poisson_ratio = poisson_ratio
        self.hardening_parameters = HardeningParameters(
            sigma_y=250.0e6,
            sigma_u=250.0e6,
            H_bar=1.0e9,
            theta=0.4,
            delta=0.0,
        )

    def set_geometry(self) -> None:
        """Create the square grid and one initially undeformed point per cell."""
        self.x = None
        self.grid = CartGrid(
            np.array([self.cells_per_axis, self.cells_per_axis]),
            physdims=np.array([1.0, 1.0]),
        )
        self.grid.compute_geometry()
        self.material_points: list[MaterialPoint] = [
            MaterialPoint(
                material=self.material,
                parameters=self.hardening_parameters,
                model=vonMisesModel("J2"),
                K_law=isotropic_hardening_K,
                H_law=kinematic_hardening_H,
            )
            for _ in range(self.grid.num_cells)
        ]

    def bc_values_displacement(self, grid: Grid) -> NDArray[np.float64]:
        """Return reference Dirichlet data as a flat vector in PorePy face ordering.

        Uses a 2D grid with computed geometry, rather than the BoundaryGrid used
        by PorePy's model interface. The callback receives a copy of face centers
        in get_all_boundary_faces() order, shape (3, nb), and returns (2, nb).
        Values are displacements in metres, without face-area scaling. Interior
        entries are zero; ravel("F") interleaves x/y components face by face.
        Every call returns fresh storage independent of the callback's arrays.
        """
        if grid.dim != 2:
            raise ValueError("Boundary displacement requires a 2D grid.")
        bf = grid.get_all_boundary_faces()
        coordinates = grid.face_centers[:, bf].copy()
        raw_values = (
            self.displacement_gradient @ coordinates[:2] + self.translation[:, None]
            if self._boundary_displacement is None else self._boundary_displacement(coordinates)
        )
        if np.iscomplexobj(raw_values):
            raise ValueError("Boundary displacement values must be real.")
        values = np.asarray(raw_values, dtype=np.float64)
        if values.shape != (2, bf.size) or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected finite boundary displacement with shape (2, {bf.size}).")
        bc_values = np.zeros((grid.dim, grid.num_faces))
        bc_values[:, bf] = values
        return bc_values.ravel(order="F")

    def _require_affine_reference(self) -> None:
        """Reject exact-solution queries when only custom boundary data are known."""
        if self._boundary_displacement is not None:
            raise ValueError("Affine reference solutions are unavailable with a custom boundary_displacement.")

    def reference_displacement(
        self, coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Exact affine displacement, shaped (2, number of points), in metres."""
        self._require_affine_reference()
        return self.displacement_gradient @ coordinates[:2] + self.translation[:, None]

    def reference_strain(self) -> strain:
        """Exact 3D total strain for this affine plane-strain problem.

        This analytical reference is not a reconstruction from numerical DOFs.
        Off-diagonal entries are tensor shear strains, not engineering shear.
        """
        self._require_affine_reference()
        value = np.zeros((3, 3))
        gradient = self.displacement_gradient
        value[:2, :2] = 0.5 * (gradient + gradient.T)
        return strain(value)

    def reference_stress(self) -> NDArray[np.float64]:
        """Exact 3D elastic Cauchy stress, including sigma_zz, in pascals."""
        epsilon = np.asarray(self.reference_strain().to_numpy(), dtype=np.float64)
        return (
            2.0 * self.material.isotropic_shear_modulus * epsilon
            + self.material.lame_parameter * float(np.trace(epsilon)) * np.eye(3)
        )

    def reference_total_pressure(self) -> float:
        """TPSA pressure lambda * div(u), not the mean Cauchy stress."""
        self._require_affine_reference()
        return float(
            self.material.lame_parameter * np.trace(self.displacement_gradient)
        )

    def reference_rotation_stress(self) -> float:
        """PorePy's 2D rotation stress mu * (du_x/dy - du_y/dx).

        The scalar orientation follows the 2D maps in Tpsa.discretize; it is
        minus mu times the usual z component of curl(u).
        """
        self._require_affine_reference()
        gradient = self.displacement_gradient
        return float(
            self.material.isotropic_shear_modulus * (gradient[0, 1] - gradient[1, 0])
        )


    def prepare_simulation(self) -> None:
        """Prepare geometry, material points, and boundaries without solving."""
        self.set_geometry()
        faces = self.grid.get_all_boundary_faces()
        self.bc = BoundaryConditionVectorial(self.grid, faces=faces, cond=["dir"] * faces.size)
        self.bound_vec = self.bc_values_displacement(self.grid)
        self.bc_values = self.bound_vec.reshape((self.grid.dim, self.grid.num_faces), order="F")
