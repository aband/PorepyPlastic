"""Elastic Cartesian TPSA baseline for subsequent cell-wise plasticity coupling.

Run with 'python -m coupling.plane_strain' from the repository root.
Lengths are in metres and stresses/moduli in pascals. The global solve uses
PorePy's Tpsa discretization and the helper structure from test_tpsa.py;
material points are initialized but not updated by the solve. Postprocessing
recovers total plane strain with a Green–Gauss reconstruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
# This environment has SciPy but no scipy-stubs; scope the exception to its import.
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.grids.grid import Grid
from porepy.grids.structured import CartGrid
from porepy.numerics.fv.tpsa import Tpsa
from porepy.params.bc import BoundaryConditionVectorial
from porepy.params.tensor import FourthOrderTensor
from porepy.utils.common_constants import DISCRETIZATION_MATRICES, PARAMETERS

from coupling.postprocessing import TpsaPostprocessing
from coupling.visualization import export_strain_png, export_vtk
from J2 import _materialProperty, vonMisesModel
from material_state import MaterialPoint
from scalar_hardening import (
    HardeningParameters,
    isotropic_hardening_K,
    kinematic_hardening_H,
)
from tensor import strain


# The helper names, argument order, block names, and solution ordering follow
# porepy/tests/numerics/fv/test_tpsa.py. The discretization itself remains PorePy's
# Tpsa from porepy/numerics/fv/tpsa.py.
KEYWORD = "mechanics"
SparseMatrix: TypeAlias = sps.spmatrix | sps.sparray
TpsaData = dict[str, Any]
DiscretizationMatrices = dict[str, SparseMatrix]


def _set_uniform_parameters(
    g: Grid,
    val: float = 1.0,
    *,
    mu: float | None = None,
    lmbda: float | None = None,
) -> TpsaData:
    """Set uniform elastic parameters, using the upstream val default.

    Optional mu and lmbda allow the plane-strain material to supply distinct
    Lamé parameters; otherwise both equal val, as in test_tpsa.py.
    """
    C = FourthOrderTensor(
        mu=np.full(g.num_cells, val if mu is None else mu),
        lmbda=np.full(g.num_cells, val if lmbda is None else lmbda),
    )
    return {
        PARAMETERS: {KEYWORD: {"fourth_order_tensor": C}},
        DISCRETIZATION_MATRICES: {KEYWORD: {}},
    }


def _set_uniform_bc(
    grid: Grid,
    d: TpsaData,
    bc_type: Literal["dir", "neu", "rob"],
) -> None:
    """Assign one boundary-condition type on all external faces."""
    if bc_type not in ("dir", "neu", "rob"):
        raise ValueError(f"Unknown boundary condition type {bc_type}")
    face_ind = grid.get_all_boundary_faces()
    d[PARAMETERS][KEYWORD]["bc"] = BoundaryConditionVectorial(
        grid, faces=face_ind, cond=[bc_type] * face_ind.size
    )


def _discretize_get_matrices(grid: Grid, d: TpsaData) -> DiscretizationMatrices:
    """Discretize with Tpsa and return its original matrix dictionary."""
    discr = Tpsa(KEYWORD)
    discr.discretize(grid, d)
    return cast(DiscretizationMatrices, d[DISCRETIZATION_MATRICES][KEYWORD])


def _assemble_matrices(
    matrices: DiscretizationMatrices, g: Grid, d: TpsaData
) -> tuple[sps.sparray, sps.sparray, sps.sparray, sps.sparray]:
    """Build face_discretization, rhs_matrix, div, and accum as in test_tpsa.

    Block rows correspond to stress, rotation, and solid mass flux. Columns
    correspond to cell displacement u, rotation stress r, and total pressure p.
    Face stresses are integrated tractions (force per thickness in 2D).
    """
    C = cast(FourthOrderTensor, d[PARAMETERS][KEYWORD]["fourth_order_tensor"])
    rot_dim = g.dim if g.dim == 3 else 1
    n_rot_face = g.num_faces * rot_dim
    n_rot_cell = g.num_cells * rot_dim

    face_discretization = sps.block_array(
        [
            [
                matrices["stress"],
                matrices["stress_rotation"],
                matrices["stress_total_pressure"],
            ],
            [
                matrices["rotation_displacement"],
                matrices["rotation_rotation"],
                sps.csr_array((n_rot_face, g.num_cells)),
            ],
            [
                matrices["solid_mass_displacement"],
                sps.csr_array((g.num_faces, n_rot_cell)),
                matrices["solid_mass_total_pressure"],
            ],
        ],
        format="csr",
    )
    rhs_matrix = sps.block_array(
        [
            [matrices["bound_stress"]],
            [matrices["bound_rotation_displacement"]],
            [matrices["bound_mass_displacement"]],
        ],
        format="csr",
    )
    div = sps.csr_array(sps.block_diag(
        [g.divergence(dim=g.dim), g.divergence(dim=rot_dim), g.divergence(dim=1)],
        format="csr",
    ))
    accum = sps.csr_array(sps.block_diag(
        [
            sps.csr_array((g.num_cells * g.dim, g.num_cells * g.dim)),
            sps.diags(np.repeat(g.cell_volumes / C.mu, rot_dim)),
            sps.diags(g.cell_volumes / C.lmbda),
        ],
        format="csr",
    ))
    return face_discretization, rhs_matrix, div, accum


def _solve(
    face_discretization: sps.sparray,
    rhs_matrix: sps.sparray,
    div: sps.sparray,
    accum: sps.sparray,
    bound_vec: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble and solve A x = b, returning x in the upstream [u, r, p] order."""
    b = -div @ rhs_matrix @ bound_vec
    A = div @ face_discretization - accum
    return np.asarray(sps.linalg.spsolve(A, b), dtype=np.float64)


class PlaneStrainTpsa:
    """Small-strain, static elasticity on an unfractured unit square.

    Prescribe u(x) = displacement_gradient @ x + translation on every boundary.
    With homogeneous moduli and zero body force, this is also the exact interior
    solution. The default is constrained extension in x, not uniaxial stress:
    both sigma_yy and sigma_zz can be nonzero.

    TPSA solves for two displacement components, one rotation stress, and one
    total pressure per cell. Each cell owns an independent 3D material history,
    indexed by PorePy cell number in material_points. Plane strain sets
    epsilon_zz = epsilon_xz = epsilon_yz = 0, not sigma_zz = 0.

    This case uses SI units without rescaling and solves the explicit sparse
    TPSA system. Call solve() to prepare and solve a fresh elastic case.
    """

    def __init__(
        self,
        *,
        cells_per_axis: int = 8,
        displacement_gradient: ArrayLike | None = None,
        translation: ArrayLike | None = None,
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

    def prepare_simulation(self) -> None:
        """Set parameters and boundaries, discretize, and assemble TPSA blocks."""
        self.set_geometry()
        g = self.grid
        self.d = _set_uniform_parameters(
            g,
            mu=self.material.isotropic_shear_modulus,
            lmbda=self.material.lame_parameter,
        )
        _set_uniform_bc(g, self.d, "dir")
        self.bc = cast(BoundaryConditionVectorial, self.d[PARAMETERS][KEYWORD]["bc"])
        self.bound_vec = np.zeros(g.dim * g.num_faces)
        self.bc_values = self.bound_vec.reshape((g.dim, g.num_faces), order="F")
        bf = g.get_all_boundary_faces()
        self.bc_values[:, bf] = self.reference_displacement(g.face_centers[:, bf])
        self.matrices = _discretize_get_matrices(g, self.d)
        (
            self.face_discretization,
            self.rhs_matrix,
            self.div,
            self.accum,
        ) = _assemble_matrices(self.matrices, g, self.d)

    def solve(self) -> NDArray[np.float64]:
        """Run the elastic case; material histories remain at their initial values.

        Each call prepares a fresh case and stores the returned vector in self.x:
        all displacement DOFs, then rotation, then total pressure, as in TPSA tests.
        """
        self.prepare_simulation()
        self.x = _solve(
            self.face_discretization, self.rhs_matrix, self.div, self.accum,
            self.bound_vec,
        )
        return self.x

    def reference_displacement(
        self, coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Exact displacement, shaped (2, number of points), in metres."""
        return self.displacement_gradient @ coordinates[:2] + self.translation[:, None]

    def reference_strain(self) -> strain:
        """Exact 3D total strain for this affine plane-strain problem.

        This analytical reference is not a reconstruction from numerical DOFs.
        Off-diagonal entries are tensor shear strains, not engineering shear.
        """
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
        return float(
            self.material.lame_parameter * np.trace(self.displacement_gradient)
        )

    def reference_rotation_stress(self) -> float:
        """PorePy's 2D rotation stress mu * (du_x/dy - du_y/dx).

        The scalar orientation follows the 2D maps in Tpsa.discretize; it is
        minus mu times the usual z component of curl(u).
        """
        gradient = self.displacement_gradient
        return float(
            self.material.isotropic_shear_modulus * (gradient[0, 1] - gradient[1, 0])
        )


def main() -> None:
    """Solve, compare with the affine reference, and export the numerical fields."""
    parser = argparse.ArgumentParser(
        description="Solve the elastic TPSA plane-strain example and export VTK and strain PNG files."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="VTK and strain PNG output directory (default: results).",
    )
    args = parser.parse_args()
    case = PlaneStrainTpsa()
    case.solve()
    post = TpsaPostprocessing(case)
    g, x = post.grid, post.x
    u, r, p = post.u_cell, post.r, post.p
    epsilon = post.strain_green_gauss()
    epsilon_ex = np.repeat(
        case.reference_strain().to_numpy()[:, :, None], g.num_cells, axis=2,
    )

    u_ex = case.reference_displacement(g.cell_centers)
    r_ex = np.full(g.num_cells, case.reference_rotation_stress())
    p_ex = np.full(g.num_cells, case.reference_total_pressure())
    generalized_flux = (
        post.face_discretization @ x + post.rhs_matrix @ post.bound_vec
    )
    # TPSA's stress block contains face-integrated tractions; normals already
    # include face measures. Forces are per unit out-of-plane thickness in 2D.
    stress = generalized_flux[:g.dim * g.num_faces].reshape((g.dim, -1), order="F")
    stress_ex = case.reference_stress()[:2, :2] @ g.face_normals[:2]
    # Absolute discrete L2 norms use cell volumes for cell fields and PorePy's
    # face dual volumes for integrated forces. In 2D the square root of these
    # measures contributes one metre to the norm's units. Vector components
    # must be interleaved by cell/face for the integration weights to align.
    comparisons = {
        "displacement [m^2]": (u_ex.ravel(order="F"), u.ravel(order="F"), True),
        "total pressure [Pa m]": (p_ex, p, True),
        "rotation stress [Pa m]": (r_ex, r, True),
        "Green-Gauss strain [m]": (
            epsilon_ex.ravel(order="F"), epsilon.ravel(order="F"), True,
        ),
        "face force per thickness [N]": (
            stress_ex.ravel(order="F"), stress.ravel(order="F"), False,
        ),
    }
    print(
        f"Cartesian plane strain: {g.num_cells} cells, "
        f"{Tpsa(KEYWORD).ndof(g)} global unknowns"
    )
    print(f"Material points: {len(case.material_points)} (initialized 3D histories)")
    print("Absolute discrete L2 errors against the affine elastic solution (norm units):")
    for name, (exact, numerical, is_cc) in comparisons.items():
        error = ConvergenceAnalysis.lp_error(
            grid=g, true_array=exact, approx_array=numerical,
            is_cc=is_cc, p=2, relative=False,
        )
        print(f"  {name}: {error:.3e}")
    print(f"Reference sigma_zz: {case.reference_stress()[2, 2]:.6e} Pa")
    pvd = export_vtk(g, x, epsilon=epsilon, folder_name=args.output_dir)
    pngs = export_strain_png(g, epsilon, folder_name=args.output_dir)
    print(f"VTK output: {pvd} (open in ParaView)")
    for name, path in pngs.items():
        print(f"Strain PNG ({name}): {path}")


if __name__ == "__main__":
    main()
