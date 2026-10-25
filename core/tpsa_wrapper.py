from dataclasses import dataclass
from typing import Any,cast

import numpy as np
import porepy as pp
import scipy.sparse as sps
from numpy.typing import NDArray

Coefficient = float | NDArray[np.float64]
SparseMatrix = sps.spmatrix | sps.sparray

@dataclass(eq=False) 
class TpsaMatrices:
    """Storing elastic TPSA matrix formulation."""

    raw:dict[str, SparseMatrix]
    face_discretization: sps.csr_array
    rhs_matrix: sps.csr_array
    div: sps.csr_array
    accum: sps.csr_array
    cell_matrix: sps.csr_array

class ElasticTpsa:
    """Prepare ans expose elastic TPSA operators for 2D or 3D grid."""
    def __init__(self,
                 grid: pp.Grid,
                 bc: pp.BoundaryConditionVectorial,
                 mu: Coefficient,
                 lam: Coefficient,
                 *,
                 keyword: str="mechanics",
                 )->None:
        if grid.dim not in (2,3):
            raise ValueError("Not admissible dimension.")

        self.grid = grid
        self.bc = bc
        self.keyword = keyword

        self.data = self._prepare_data(mu, lam)
        self.discretization = pp.Tpsa(keyword)
        self.discretization.discretize(grid, self.data)
        self.matrices = self._assemble_matrices()

    def _prepare_data(
            self, mu: Coefficient, lam: Coefficient,
    )->dict[str, Any]: 
        """Validate inputs and create PorePy's parameter dictionaries."""

        nc= self.grid.num_cells
        if nc < 1:
            raise ValueError("The grid must contain at least one cell.")

        def cell_values(
                name: str, value: Coefficient,
        )-> NDArray[np.float64]:
            if np.iscomplexobj(value):
                raise ValueError(f"{name} must be real.")

            values = np.array(value, dtype=np.float64, copy=True)

            if values.ndim == 0:
                values = np.full(nc, values.item(), dtype=np.float64)

            if values.shape != (nc,):
                raise ValueError(f"{name} must be a scalar or have shape ({nc},).")

            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"{name} must contain finite positive values.")

            return values   

        if (
            self.bc.dim != self.grid.dim
            or self.bc.num_faces != self.grid.num_faces
        ):
            raise ValueError("Boundary-condition dimensions do not match the grid.")

        stiffness = pp.FourthOrderTensor(
            mu=cell_values("mu", mu),
            lmbda=cell_values("lam", lam),
        )

        return {
            pp.PARAMETERS: {
                self.keyword: {
                    "fourth_order_tensor": stiffness,
                    "bc": self.bc,
                },
            },
            pp.DISCRETIZATION_MATRICES: {
                self.keyword: {},
            },
        }

    def _assemble_matrices(self) -> TpsaMatrices:
        """Assemble the elastic blocks from Porepy's matrix dictionaries."""
        g = self.grid
        nc, nf = g.num_cells, g.num_faces
        rot_dim = 1 if g.dim == 2 else 3
        n_rot_cell = rot_dim * nc
        n_rot_face = rot_dim * nf

        raw = cast(
            dict[str, SparseMatrix],
            self.data[pp.DISCRETIZATION_MATRICES][self.keyword],
        )
        stiffness = cast(
            pp.FourthOrderTensor,
            self.data[pp.PARAMETERS][self.keyword]["fourth_order_tensor"],
        )

        # Assemble TPSA elasticity matrix
        # Follow the routine in porepy/src/numerics/fv/tpsa.py
        face_discretization = sps.block_array(
            [
                [
                    raw["stress"],
                    raw["stress_rotation"],
                    raw["stress_total_pressure"],
                ],
                [
                    raw["rotation_displacement"],
                    raw["rotation_rotation"],
                    sps.csr_array((n_rot_face, nc)),
                ],
                [
                    raw["solid_mass_displacement"],
                    sps.csr_array((nf, n_rot_cell)),
                    raw["solid_mass_total_pressure"],
                ],
            ],
            format="csr",
        )

        # Boundary values contribute to the same three flux groups.
        # A compact definition of the original rhs_matrix in porepy/src/numerics/fv/tpsa.py
        rhs_matrix = sps.csr_array(
            sps.vstack(
                [
                    raw["bound_stress"],
                    raw["bound_rotation_displacement"],
                    raw["bound_mass_displacement"],
                ],
                format="csr",
            )
        )

        div = sps.csr_array(
            sps.block_diag(
                [
                    g.divergence(dim=g.dim),
                    g.divergence(dim=rot_dim),
                    g.divergence(dim=1),
                ],
                format="csr",
            )
        )

        accum = sps.csr_array(
            sps.block_diag(
                [
                    sps.csr_array((g.dim * nc, g.dim * nc)),
                    sps.diags(
                        np.repeat(g.cell_volumes / stiffness.mu, rot_dim)
                    ),
                    sps.diags(g.cell_volumes / stiffness.lmbda),
                ],
                format="csr",
            )
        )

        cell_matrix = sps.csr_array(
            div @ face_discretization - accum
        )

        return TpsaMatrices(
            raw=raw,
            face_discretization=sps.csr_array(face_discretization),
            rhs_matrix=rhs_matrix,
            div=div,
            accum=accum,
            cell_matrix=cell_matrix,
        )

    def boundary_rhs(
            self, boundary_values:NDArray[np.float64],
    )->NDArray[np.float64]:
        """Return the elastic boundary contribution."""
        if np.iscomplexobj(boundary_values):
            raise ValueError("Boundary values must be real number.")

        values = np.asarray(boundary_values,dtype=np.float64)
        expected_size = self.grid.dim * self.grid.num_faces

        if values.shape != (expected_size,):
            raise ValueError(
                f"Boundary values must have shape ({expected_size},)."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Boundary values must be finite.")

        matrices = self.matrices
        rhs = matrices.div @ (matrices.rhs_matrix @ values)
        return np.asarray(-rhs, dtype=np.float64)