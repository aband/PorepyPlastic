import copy
from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import porepy as pp

import scipy.sparse as sps

from porepy.applications.md_grids.domains import nd_cube_domain

KEYWORD = "mechanics"

def _assemble_matrices(
    matrices: dict, g: pp.Grid, d: dict
) -> tuple[sps.sparray, sps.sparray, sps.sparray, sps.sparray]:
    """Helper method to assemble discretization matrices derived from a Tpsa
    discretization into global matrices.

    Parameters:
        matrices: Dictionary containing the discretization matrices.
        g: Grid object.

    Returns:
        sps.sparray: Discretization of the face terms as a block matrix. The first block
            row contains the stress terms, the second the rotation terms, and the third
            the solid mass 'flux'.
        sps.sparray: Discretization of the boundary conditions, as a map from numerical
            values for the boundary condition to stresses, rotations and solid mass
            fluxes on the boundary faces.
        sps.sparray: Divergence matrix for the face terms.
        sps.sparray: Accumulation matrix for the cell center terms.

    """
    C = d[pp.PARAMETERS][KEYWORD]["fourth_order_tensor"]

    # Deal with the different dimensions of the rotation variable.
    rot_dim = g.dim if g.dim == 3 else 1

    n_rot_face = g.num_faces * rot_dim
    n_rot_cell = g.num_cells * rot_dim
    div_rot = g.divergence(dim=rot_dim)

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
    )

    rhs_matrix = sps.block_array(
        [
            [matrices["bound_stress"]],
            [matrices["bound_rotation_displacement"]],
            [matrices["bound_mass_displacement"]],
        ]
    )

    div = sps.block_diag(
        [
            g.divergence(dim=g.dim),
            div_rot,
            g.divergence(dim=1),
        ],
        format="csr",
    )

    accum = sps.block_diag(
        [
            sps.csr_array((g.num_cells * g.dim, g.num_cells * g.dim)),
            sps.eye(n_rot_cell),
            sps.eye(g.num_cells),
        ],
        format="csr",
    )
    accum = sps.block_diag(
        [
            sps.csr_array((g.num_cells * g.dim, g.num_cells * g.dim)),
            sps.dia_matrix(
                (np.repeat(g.cell_volumes / C.mu, rot_dim), 0),
                shape=(n_rot_cell, n_rot_cell),
            ),
            sps.dia_matrix(
                (g.cell_volumes / C.lmbda, 0), shape=(g.num_cells, g.num_cells)
            ),
        ],
        format="csr",
    )
    return face_discretization, rhs_matrix, div, accum

def _discretize_get_matrices(grid: pp.Grid, d: dict):
    """Helper function to discretize with Tpsa and return the dictionary of
    discretization matrices.

    Parameters:
        grid: Grid to discretize.
        d: Dictionary with parameters.

    Returns:
        Dictionary of discretization matrices.

    """
    discr = pp.Tpsa(KEYWORD)
    discr.discretize(grid, d)
    return d[pp.DISCRETIZATION_MATRICES][KEYWORD]

def _solve(
    face_discretization: sps.sparray,
    rhs_matrix: sps.sparray,
    div: sps.sparray,
    accum: sps.sparray,
    bound_vec: np.ndarray,
) -> np.ndarray:
    """Assemble the Tpsa problem and solve.

    Parameters:
        face_discretization: Discretization of the face terms as a block matrix.
        rhs_matrix: Discretization of the boundary conditions.
        div: Divergence matrix for the face terms.
        accum: Accumulation matrix for the cell center terms.
        bound_vec: Array of boundary condition values.

    Returns:
        np.ndarray: Array of cell center values.
    """
    b = -div @ rhs_matrix @ bound_vec

    # Assemble and solve. The minus sign on accum follows from the definition of the
    # governing equations in the paper.
    A = div @ face_discretization - accum
    print(A.shape[0])
    x = sps.linalg.spsolve(A, b)
    return x

# run script
if __name__ == "__main__":

    n = 5
    g = pp.CartGrid([n,n])
    g.compute_geometry()

    # Create stiffness matrix
    # Values not defined yet
    lam = np.ones(g.num_cells)
    mu = np.ones(g.num_cells)
    C = pp.FourthOrderTensor(mu, lam)

    # Define boundary type
    dirich = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))  # Bottom?
    bound = pp.BoundaryConditionVectorial(g, dirich, ["dir"] * dirich.size)

    top_faces = np.ravel(np.argwhere(g.face_centers[1] > n - 1e-10))
    bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))

    u_b = np.zeros((g.dim, g.num_faces))
    u_b[1, top_faces] = -1 * g.face_areas[top_faces]
    u_b[:, bot_faces] = 0

    u_b = u_b.ravel("F")

    # Discretize with Tpsa scheme
    disc = pp.Tpsa(KEYWORD)

    f = np.zeros(g.dim * g.num_cells)

    specified_parameters = {
        "fourth_order_tensor": C,
        "source": f,
        "bc": bound,
        "bc_values": u_b,
    }

    data = pp.initialize_data({}, KEYWORD, specified_parameters)
    #data = {pp.PARAMETERS: {KEYWORD: {'fourth_order_tensor': C, 'bc': bound}}}
    #disc.discretize(g, data)

    # Assemble_matrix_rhs function not implemented

    matrices = _discretize_get_matrices(g,data)
    face_discretization, rhs_matrix, div, accum = _assemble_matrices(matrices, g, data)

    # Set up and solve the system.
    x = _solve(face_discretization, rhs_matrix, div, accum, u_b.ravel("F"))

    # Extract stress
    #stress = data[pp.DISCRETIZATION_MATRICES][KEYWORD][
    #disc.stress_displacement_matrix_key]

    #bound_stress = data[pp.DISCRETIZATION_MATRICES][KEYWORD][
    #disc.bound_stress_matrix_key]

    # Displacement
    u = x[:g.dim * g.num_cells] 

    # Plot grids
    pp.plot_grid(g, None, figsize=(15, 12), alpha=0, plot_2d=True)
