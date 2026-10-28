"""Nested material trials, one-sided tractions, and retained-face TPSA residuals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from utils.material_evaluation import MaterialTrial, evaluate_material_at_strain
from utils.state import TpsaState
from material.material_state import MaterialPoint
from prototype.operators import RetainedFaceOperators


def evaluate_one_sided_traction(
    operators: RetainedFaceOperators,
    z: NDArray[np.float64],
    bc_values: NDArray[np.float64],
    committed_bc_values: NDArray[np.float64],
    stress_correction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return outward integrated traction INCREMENTS with shape (dim,num_sides).

    Implements Eqs. (5)-(7) of Plasticity_revised.pdf:
        delta_t_ka = A_k [a_ka*(d_k-delta_u_a) + (S*delta_r_k)*n_ka
                         + delta_p_a*n_ka - Q_a*n_ka].
    Side order is operators.side_faces / operators.side_cells. Both sides of
    an interior face are evaluated independently, without enforcing balance or
    applying another incidence sign. Boundary outputs are reaction increments.

    z contains load-step increments [delta_u,delta_r,delta_p,d_free]. Boundary
    arrays contain absolute displacements; d on the boundary is target minus
    committed. The 2D TPSA convention is (S*r)*n = r*(-n_y,n_x). Shared rotation
    uses complementary a weights internally and the cell value on the boundary.

    Q is a finite symmetric (3,3,nc) cell stress correction supplied by the
    caller, e.g. sigma_n + C_e:delta_epsilon - sigma_returned. The active spatial
    block enters traction; it is never transferred between cells. Lamé lambda
    enters through delta_p and is not needed here. Material updates, committed
    traction addition, and global residual assembly are separate operations.
    Inputs remain unchanged; the returned array owns its storage.
    """
    stiffness, transfer = operators.side_stiffness, operators.rotation_transfer
    if stiffness is None or transfer is None:
        raise ValueError("Supply mu when constructing RetainedFaceOperators to evaluate traction.")
    nc, nd, nr = operators.num_cells, operators.dim, operators.rotation_dim
    if (stress_correction.shape != (3, 3, nc)
            or not np.all(np.isfinite(stress_correction))
            or not np.allclose(stress_correction, stress_correction.swapaxes(0, 1), rtol=1e-10, atol=1e-14)):
        raise ValueError("Expected finite symmetric stress_correction with shape (3, 3, num_cells).")
    d = operators.face_displacement_increment(z, bc_values, committed_bc_values)
    u = z[operators.u_slice].reshape((nd, nc), order="F")
    rotation = transfer.apply(z[operators.r_slice].reshape((nr, nc), order="F"))
    pressure = z[operators.p_slice]
    faces, cells = operators.side_faces, operators.side_cells
    n = operators.side_normals
    rotation_traction = (
        np.vstack((-n[1], n[0])) * rotation[0, faces]
        if nd == 2 else np.cross(rotation[:, faces].T, n.T).T
    )
    with np.errstate(over="ignore", invalid="ignore"):
        traction = operators.side_areas * (
            stiffness * (d[:, faces] - u[:, cells])
            + rotation_traction
            + pressure[cells] * n
            - np.einsum("ijs,js->is", stress_correction[:nd, :nd, cells], n)
        )
    if not np.all(np.isfinite(traction)):
        raise ValueError("One-sided traction increments must remain finite.")
    return np.asarray(traction, dtype=np.float64)


def assemble_global_residual(
    operators: RetainedFaceOperators,
    z: NDArray[np.float64],
    bc_values: NDArray[np.float64],
    committed_bc_values: NDArray[np.float64],
    traction_increment: NDArray[np.float64],
    *,
    body_force_increment: NDArray[np.float64] | None = None,
    neumann_increment: NDArray[np.float64] | None = None,
    committed_residual: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Assemble integrated [R_u,R_r,R_p,R_f], ordered exactly like z.

    Implements Eqs. (8)-(11) of Plasticity_revised.pdf, with zero auxiliary
    sources and the convention div(sigma) = body_force:
        R_u,i = sum_k delta_t_ki - delta_b_i,
        R_r,i = sum_k A_k*(d_k x n_ki) - (V_i/mu_i)*delta_r_i,
        R_p,i = sum_k A_k*(d_k . n_ki) - (V_i/lam_i)*delta_p_i,
        R_f,k = delta_t_ki + delta_t_kj  (interior),
        R_f,k = delta_t_ki - prescribed_delta_t_k  (Neumann boundary).
    In 2D, d x n = d_x*n_y - d_y*n_x. Tractions are already outward and
    integrated: no additional incidence signs or face areas multiply them.

    traction_increment has shape (dim,num_sides), from the same candidate and
    load as z. Boundary arrays contain absolute displacements; prescribed d
    is target minus committed. body_force_increment has shape (dim,num_cells)
    and contains INTEGRATED force increments, not densities or a boundary-
    shifted right-hand side. Omitted body force means zero load increment.

    neumann_increment has shape (dim, num_neumann_faces), ordered by
    operators.neumann_faces; values are OUTWARD integrated forces, not stresses.
    None means zero Neumann load increment.

    Omitting committed_residual assumes an equilibrated previous step. Supply
    its known unscaled residual (num_dofs,) to add that imbalance once; it must
    use the same geometry, ordering, and sign convention. Dirichlet reactions
    enter cell momentum sums but have no face-balance equations.

    Plastic corrections enter through supplied tractions only; no correction
    is subtracted again in either auxiliary block. This function does not
    evaluate materials, solve for any unknowns, or commit a candidate. Inputs
    remain unchanged and the returned vector owns its storage.
    """
    rotation_accumulation = operators.rotation_accumulation
    pressure_accumulation = operators.pressure_accumulation
    if rotation_accumulation is None or pressure_accumulation is None:
        raise ValueError("Supply mu and lam when constructing RetainedFaceOperators for residual assembly.")
    nc, nf, nd = operators.num_cells, operators.num_faces, operators.dim
    for name, values, shape in (
        ("traction_increment", traction_increment, (nd, operators.num_sides)),
        ("body_force_increment", body_force_increment, (nd, nc)),
        ("neumann_increment", neumann_increment, (nd, operators.neumann_faces.size)),
        ("committed_residual", committed_residual, (operators.num_dofs,)),
    ):
        if values is not None and (values.shape != shape or not np.all(np.isfinite(values))):
            raise ValueError(f"Expected finite {name} with shape {shape}.")
    d = operators.face_displacement_increment(z, bc_values, committed_bc_values)
    cells, faces = operators.side_cells, operators.side_faces
    n, area = operators.side_normals, operators.side_areas
    side_d = d[:, faces]
    result = np.empty(operators.num_dofs)
    with np.errstate(over="ignore", invalid="ignore"):
        momentum = np.vstack([
            np.bincount(cells, weights=traction_increment[component], minlength=nc)
            for component in range(nd)
        ], dtype=np.float64)
        if body_force_increment is not None:
            momentum -= body_force_increment
        result[operators.u_slice] = momentum.ravel(order="F")
        rotation_flux = area * (
            (side_d[0] * n[1] - side_d[1] * n[0])[None, :]
            if nd == 2 else np.cross(side_d.T, n.T).T
        )
        mass_flux = area * np.sum(side_d * n, axis=0)
        result[operators.r_slice] = (
            np.vstack([
                np.bincount(cells, weights=component, minlength=nc)
                for component in rotation_flux
            ]).ravel(order="F")
            - np.repeat(rotation_accumulation, operators.rotation_dim) * z[operators.r_slice]
        )
        result[operators.p_slice] = (
            np.bincount(cells, weights=mass_flux, minlength=nc)
            - pressure_accumulation * z[operators.p_slice]
        )
        face_balance = np.vstack([
            np.bincount(faces, weights=traction_increment[component], minlength=nf)
            for component in range(nd)
        ], dtype=np.float64)
        if neumann_increment is not None:
            face_balance[:, operators.neumann_faces] -= neumann_increment
        result[operators.face_slice] = face_balance[:, operators.free_faces].ravel(order="F")
        if committed_residual is not None:
            result += committed_residual
    if not np.all(np.isfinite(result)):
        raise ValueError("Global residual must remain finite.")
    return result


@dataclass(eq=False)
class RetainedFaceResidual:
    """Owned results of one candidate evaluated from fixed committed history.

    residual is unscaled [R_u,R_r,R_p,R_f], ordered like z. material_trial
    reuses MaterialTrial: total strain, full 3D material histories, and Q.
    traction_increment is (dim,num_sides), outward and integrated on each side.
    z and bc_values snapshot the matching candidate and absolute boundary data
    for Jacobian validation; unused interior boundary entries are stored as zero.
    These are trial data, not an accepted global state or total face tractions.
    No arrays or material histories alias the evaluation's inputs.
    """

    residual: NDArray[np.float64]
    material_trial: MaterialTrial
    traction_increment: NDArray[np.float64]
    z: NDArray[np.float64]
    bc_values: NDArray[np.float64]


def evaluate_global_residual(
    operators: RetainedFaceOperators,
    material_points: Sequence[MaterialPoint],
    committed: TpsaState,
    z: NDArray[np.float64],
    bc_values: NDArray[np.float64],
    *,
    body_force_increment: NDArray[np.float64] | None = None,
    neumann_increment: NDArray[np.float64] | None = None,
    committed_residual: NDArray[np.float64] | None = None,
) -> RetainedFaceResidual:
    """Evaluate the nested residual, rerunning local return maps for this z.

    Reconstruct epsilon = epsilon_n + GreenGauss(d), then reuse the existing
    evaluate_material_at_strain / MaterialPoint.update with fixed H_n to obtain
    Q = sigma_n + C_e:(epsilon-epsilon_n) - sigma_returned. Insert that same Q
    into the one-sided traction law and assemble all four global balance blocks.

    z contains physical load-step increments; bc_values is the absolute target
    boundary displacement. The existing TpsaState supplies epsilon_n, material
    histories, and old boundary values. Its cell unknowns and stored tractions
    are not needed for incremental evaluation; any old equilibrium imbalance
    must be supplied as committed_residual. Load conventions match
    assemble_global_residual. Both mu and lam must match the cell materials.

    Each call, including a line-search trial, starts from the same committed
    argument. Material points supply models, parameters, and hardening laws;
    their own committed/trial slots are untouched. Local failures propagate
    without partial acceptance. Only the face increments determine strain;
    the supplied cell u,r,p remain independent unconverged unknowns.

    No tangent, Newton step, history commit, or total traction update is done.
    Returned histories can be reused by the subsequent consistent Jacobian.
    """
    mu, lam = operators.mu, operators.lam
    if mu is None or lam is None:
        raise ValueError("Supply mu and lam when constructing RetainedFaceOperators for residual evaluation.")
    nc = operators.num_cells
    if len(material_points) != nc or len(committed.material_states) != nc:
        raise ValueError("Expected one material point and committed history per cell.")
    for cell, point in enumerate(material_points):
        if not np.allclose(
            [point.material.isotropic_shear_modulus, point.material.lame_parameter],
            [mu[cell], lam[cell]], rtol=1e-12, atol=0,
        ):
            raise ValueError(f"Material elastic coefficients in cell {cell} must match the retained-face operators.")

    epsilon = operators.reconstruct_strain(z, bc_values, committed.bc_values, committed.epsilon)
    material_trial = evaluate_material_at_strain(operators, material_points, committed, epsilon, dim=operators.dim)
    traction = evaluate_one_sided_traction(
        operators, z, bc_values, committed.bc_values, material_trial.stress_correction,
    )
    residual = assemble_global_residual(
        operators, z, bc_values, committed.bc_values, traction,
        body_force_increment=body_force_increment, neumann_increment=neumann_increment,
        committed_residual=committed_residual,
    )
    boundary = np.zeros((operators.dim, operators.num_faces))
    boundary[:, operators.dirichlet_faces] = bc_values[:, operators.dirichlet_faces]
    return RetainedFaceResidual(residual, material_trial, traction, z.copy(), boundary)
