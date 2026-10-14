"""Elastic Cartesian TPSA baseline for subsequent cell-wise plasticity coupling.

Run with 'python -m examples.elastic_plane_strain' from the repository root.
Lengths are in metres and stresses/moduli in pascals. The global solve uses
PorePy's Tpsa discretization and the helper structure from test_tpsa.py;
material points are initialized but not updated by the solve. Postprocessing
recovers total plane strain with a Green–Gauss reconstruction.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, cast

import numpy as np
# This environment has SciPy but no scipy-stubs; scope the exception to its import.
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.grid import Grid
from porepy.numerics.fv.tpsa import Tpsa
from porepy.params.bc import BoundaryConditionVectorial
from porepy.params.tensor import FourthOrderTensor
from porepy.utils.common_constants import DISCRETIZATION_MATRICES, PARAMETERS


from utils.problem import BoundaryDisplacement as BoundaryDisplacement, PlaneStrainProblem


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


class PlaneStrainTpsa(PlaneStrainProblem):
    """Small-strain, static elasticity on an unfractured unit square.

    By default prescribe u(x) = displacement_gradient @ x + translation on every
    boundary. With homogeneous moduli and zero body force, this is also the exact
    interior solution. The default is constrained extension in x, not uniaxial
    stress: both sigma_yy and sigma_zz can be nonzero.

    Alternatively, boundary_displacement maps boundary-face centers (3, nb) to
    displacements (2, nb), both in metres. It is evaluated once per preparation
    and cannot be combined with explicit displacement_gradient or translation.
    These values represent load factor 1; LoadStepController scales the entire
    array for proportional loading. Custom callbacks specify boundary data only,
    so the affine reference_* methods are unavailable for these cases.

    TPSA solves for two displacement components, one rotation stress, and one
    total pressure per cell. Each cell owns an independent 3D material history,
    indexed by PorePy cell number in material_points. Plane strain sets
    epsilon_zz = epsilon_xz = epsilon_yz = 0, not sigma_zz = 0.

    This case uses SI units without rescaling and solves the explicit sparse
    TPSA system. Call solve() to prepare and solve a fresh elastic case.
    """

    def prepare_simulation(self) -> None:
        """Set parameters and boundaries, discretize, and assemble TPSA blocks."""
        super().prepare_simulation()
        g = self.grid
        self.d = _set_uniform_parameters(
            g,
            mu=self.material.isotropic_shear_modulus,
            lmbda=self.material.lame_parameter,
        )
        self.d[PARAMETERS][KEYWORD]["bc"] = self.bc
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
