"""State, trial evaluation, and fully coupled global residual for TPSA."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
# This environment has SciPy but no scipy-stubs.
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.grids.grid import Grid

from coupling.postprocessing import (
    gradient_green_gauss, plane_strain_from_gradient, reconstruct_face_displacement,
)
from coupling.transfer import CellToFaceTransfer
from material_state import MaterialPoint, MaterialState
from tensor import strain

if TYPE_CHECKING:
    from coupling.plane_strain import PlaneStrainTpsa

@dataclass(eq=False)
class TpsaState:
    """Owned data for one committed state or one independent trial candidate.

    x stores [u, r, p], with the two displacement components interleaved by cell.
    bc_values stores the boundary data belonging to this state, not the next load.
    epsilon contains total 3D plane strain; material_states retain full 3D stress
    and plastic history. traction contains integrated numerical face tractions in
    the grid's fixed normal orientation, distinct from cell material stresses.

    Construction copies all arrays and each material history. Keep the committed
    object unchanged and use copy() for a mutable trial. Validation checks storage
    and tensor conventions; equilibrium and constitutive compatibility require the
    residual evaluator. This container does not solve, update, or commit a load.
    """

    x: NDArray[np.float64]                 # (4 * nc,)
    bc_values: NDArray[np.float64]         # (2, nf), finite, unused entries zero
    epsilon: NDArray[np.float64]           # (3, 3, nc), tensor shear strain
    material_states: list[MaterialState]  # one independent history per cell
    traction: NDArray[np.float64]          # (2, nf), force per thickness in 2D

    def __post_init__(self) -> None:
        nc = len(self.material_states)
        if nc < 1 or self.bc_values.ndim != 2 or self.bc_values.shape[1] < 1:
            raise ValueError("Expected a nonempty cell state and boundary array (2, nf).")
        nf = self.bc_values.shape[1]
        for name, shape in {
            "x": (4 * nc,), "bc_values": (2, nf),
            "epsilon": (3, 3, nc), "traction": (2, nf),
        }.items():
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}.")
            setattr(self, name, value)
        if not np.allclose(
            self.epsilon, self.epsilon.swapaxes(0, 1), rtol=1e-10, atol=1e-14,
        ):
            raise ValueError("Total strain must be symmetric.")
        if not np.allclose(self.epsilon[2], 0.0, atol=1e-14) or not np.allclose(
            self.epsilon[:, 2], 0.0, atol=1e-14,
        ):
            raise ValueError("Total strain must satisfy plane strain (zz = xz = yz = 0).")
        for state in self.material_states:
            if not np.isfinite(state.alpha) or state.alpha < 0:
                raise ValueError("Material alpha must be finite and nonnegative.")
            for name in ("stress", "plastic_strain", "backstress"):
                value = getattr(state, name).to_numpy(copy=False)
                if (
                    value.shape != (3, 3) or not np.all(np.isfinite(value))
                    or not np.allclose(value, value.T, rtol=1e-10, atol=1e-14)
                ):
                    raise ValueError(f"Material {name} must be a finite symmetric 3D tensor.")
        self.material_states = [state.copy() for state in self.material_states]

    @classmethod
    def zeros(cls, grid: Grid) -> TpsaState:
        """Create an undeformed, unstressed state with zero old boundary data.

        Assumes zero initial sources and an admissible unstressed material.
        Geometry need not be computed. Nonzero initial states must be supplied
        with compatible strains, histories, boundary data, and face tractions.
        """
        nc, nf = grid.num_cells, grid.num_faces
        if grid.dim != 2 or nc < 1 or nf < 1:
            raise ValueError("Expected a nonempty 2D grid.")
        return cls(
            x=np.zeros(4 * nc), bc_values=np.zeros((2, nf)),
            epsilon=np.zeros((3, 3, nc)),
            material_states=[MaterialState() for _ in range(nc)],
            traction=np.zeros((2, nf)),
        )

    def copy(self) -> TpsaState:
        """Copy all arrays and material histories for an independent candidate."""
        return deepcopy(self)


class TpsaOperators:
    """Fixed reference TPSA blocks and reusable full-Dirichlet auxiliary solves.

    Build after case.prepare_simulation(); an elastic solution is not required.
    The grid is shared, while sparse matrices and boundary indices are copied.
    Treat the grid and these operators as fixed. Rebuild this object after changing
    geometry, elastic coefficients, boundary types, or discretization matrices.
    Changing boundary values or integrated sources only requires assemble_rhs().

    This stage supports finite positive elastic coefficients, with uncoupled
    rotation/pressure equations. It does not impose boundary values on cell DOFs
    or evaluate material trials. All cells retain both displacement unknowns.
    """

    def __init__(self, case: PlaneStrainTpsa) -> None:
        if not hasattr(case, "face_discretization"):
            raise RuntimeError("Call case.prepare_simulation() before creating operators.")
        self.grid = case.grid
        nc, nf = self.grid.num_cells, self.grid.num_faces
        if self.grid.dim != 2 or nc < 1 or nf < 1:
            raise ValueError("Expected a nonempty 2D grid.")
        self.num_cells, self.num_faces = nc, nf
        self.boundary_faces = self.grid.get_all_boundary_faces().copy()
        if case.bc.is_dir.shape != (2, nf) or not np.all(
            case.bc.is_dir[:, self.boundary_faces],
        ):
            raise ValueError("Operators require Dirichlet data on every boundary face.")

        self.face_discretization = sps.csr_array(case.face_discretization, copy=True)
        self.rhs_matrix = sps.csr_array(case.rhs_matrix, copy=True)
        self.div = sps.csr_array(case.div, copy=True)
        self.accum = sps.csr_array(case.accum, copy=True)
        for name, shape in {
            "face_discretization": (4 * nf, 4 * nc),
            "rhs_matrix": (4 * nf, 2 * nf),
            "div": (4 * nc, 4 * nf), "accum": (4 * nc, 4 * nc),
        }.items():
            matrix = getattr(self, name)
            if matrix.shape != shape or not np.all(np.isfinite(matrix.data)):
                raise ValueError(f"{name} must be finite with shape {shape}; prepare the case again.")
        if np.any(self.accum.diagonal()[2 * nc:] <= 0):
            raise ValueError("Auxiliary accumulation requires finite positive elastic coefficients.")

        self.A = (self.div @ self.face_discretization - self.accum).tocsr()
        self.A.eliminate_zeros()
        u, r, p = slice(0, 2 * nc), slice(2 * nc, 3 * nc), slice(3 * nc, 4 * nc)
        if self.A[r, p].nnz or self.A[p, r].nnz:
            raise ValueError("Separate auxiliary solves require zero rotation-pressure coupling.")
        self.A_uu, self.A_ur, self.A_up = self.A[u, u], self.A[u, r], self.A[u, p]
        self.A_ru, self.A_rr = self.A[r, u], self.A[r, r]
        self.A_pu, self.A_pp = self.A[p, u], self.A[p, p]
        self.T = self.face_discretization[:2 * nf]
        self.T_g = self.rhs_matrix[:2 * nf]
        self.D_u = self.div[:2 * nc, :2 * nf]
        self._boundary_rhs = -(self.div @ self.rhs_matrix)
        try:
            self._rotation_factor = sps.linalg.splu(self.A_rr.tocsc())
            self._pressure_factor = sps.linalg.splu(self.A_pp.tocsc())
        except RuntimeError as error:
            raise ValueError("The rotation and pressure blocks must be nonsingular.") from error

    def assemble_rhs(
        self,
        bc_values: NDArray[np.float64],
        *,
        sources: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return f = b - div @ rhs_matrix @ g for one target load.

        bc_values has shape (2, nf); only boundary entries are used and must be
        finite. sources is the integrated [body force, rotation source, pressure
        source] vector with shape (4 * nc,), defaulting to zero. Displacement
        components are interleaved, as in x. Inputs are preserved.

        Reuse the returned right-hand side during a fixed load's trial evaluations.
        Its displacement block includes boundary contributions; the force
        residual div(complete traction) - body_force must use the original body
        source rather than subtracting that shifted block again.
        """
        values = np.asarray(bc_values, dtype=np.float64)
        if values.shape != (2, self.num_faces) or not np.all(
            np.isfinite(values[:, self.boundary_faces]),
        ):
            raise ValueError(f"Expected bc_values (2, {self.num_faces}), finite on boundary faces.")
        boundary = np.zeros_like(values)
        boundary[:, self.boundary_faces] = values[:, self.boundary_faces]
        rhs = np.zeros(4 * self.num_cells) if sources is None else np.array(
            sources, dtype=np.float64, copy=True,
        )
        if rhs.shape != (4 * self.num_cells,) or not np.all(np.isfinite(rhs)):
            raise ValueError(f"Expected finite sources with shape ({4 * self.num_cells},).")
        rhs += self._boundary_rhs @ boundary.ravel(order="F")
        return rhs

    def solve_auxiliary(
        self, u: NDArray[np.float64], rhs: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Recover r and p for a candidate u and an assembled target-load rhs.

        Solve A_rr r = f_r - A_ru u and A_pp p = f_p - A_pu u using the stored
        factorizations. u has shape (2 * nc,), rhs has shape (4 * nc,), and the
        two returned arrays each have shape (nc,). The full pressure coupling is
        retained. No input array, case solution, or material history is changed.
        """
        nc = self.num_cells
        displacement = np.asarray(u, dtype=np.float64)
        load = np.asarray(rhs, dtype=np.float64)
        for name, value, size in [("u", displacement, 2 * nc), ("rhs", load, 4 * nc)]:
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Expected finite {name} with shape ({size},).")
        r = self._rotation_factor.solve(load[2 * nc:3 * nc] - self.A_ru @ displacement)
        p = self._pressure_factor.solve(load[3 * nc:] - self.A_pu @ displacement)
        return np.asarray(r, dtype=np.float64), np.asarray(p, dtype=np.float64)


@dataclass(eq=False)
class TpsaMaterialTrial:
    """Cell results from one trial evaluation, with independent arrays and histories.

    epsilon and stress_correction have shape (3, 3, nc). The correction is the
    elastic predictor minus returned stress for the current load increment.
    Numerical face tractions and force equilibrium are evaluated separately.
    """

    epsilon: NDArray[np.float64]
    material_states: list[MaterialState]
    stress_correction: NDArray[np.float64]


def evaluate_material_trial(
    operators: TpsaOperators,
    material_points: Sequence[MaterialPoint],
    committed: TpsaState,
    x: NDArray[np.float64],
    bc_values: NDArray[np.float64],
) -> TpsaMaterialTrial:
    """Reconstruct a candidate and return cell material updates without committing.

    x stores the current [u, r, p], including unconverged coupled candidates.
    No auxiliary equation is solved or required to be satisfied here.
    Reconstruct total Green–Gauss strain, then update each material with
    epsilon - committed.epsilon. Material points supply the fixed constitutive
    models and parameters; their stored committed/trial histories are not used.
    Their elastic coefficients must match those used to build the TPSA operators.

    The full 3D stress correction is sigma_n + C_e : delta_epsilon - sigma_trial;
    it is zero for elastic increments, including unloading from a plastic state.
    Every call starts from committed, even after a rejected candidate or local
    failure. A failed local update propagates without modifying input histories.
    This function neither solves force equilibrium nor updates face tractions.
    """
    nc = operators.num_cells
    if len(material_points) != nc or len(committed.material_states) != nc:
        raise ValueError("Expected one material point and committed history per cell.")
    if committed.epsilon.shape != (3, 3, nc) or not np.all(np.isfinite(committed.epsilon)):
        raise ValueError("Expected finite committed total strain with shape (3, 3, nc).")
    u_face = reconstruct_face_displacement(
        operators.grid, operators.face_discretization, operators.rhs_matrix, x, bc_values,
    )
    epsilon = plane_strain_from_gradient(gradient_green_gauss(operators.grid, u_face))
    histories: list[MaterialState] = []
    correction = np.empty_like(epsilon)
    for cell, point in enumerate(material_points):
        old = committed.material_states[cell]
        increment = strain(epsilon[:, :, cell] - committed.epsilon[:, :, cell])
        # A local point copies history and isolates its mutable trial slot.
        local = MaterialPoint(
            material=point.material, parameters=point.parameters, model=point.model,
            K_law=point.K_law, H_law=point.H_law, committed=old,
        )
        history, _ = local.update(increment)
        elastic_increment = point.material.elastic_tensor.double_contract(increment)
        correction[:, :, cell] = (
            old.stress.to_numpy(copy=False) + elastic_increment.to_numpy(copy=False)
            - history.stress.to_numpy(copy=False)
        )
        histories.append(history)
    return TpsaMaterialTrial(epsilon, histories, correction)


def evaluate_face_traction(
    operators: TpsaOperators,
    transfer: CellToFaceTransfer,
    committed: TpsaState,
    x: NDArray[np.float64],
    bc_values: NDArray[np.float64],
    material_trial: TpsaMaterialTrial,
) -> NDArray[np.float64]:
    """Return integrated numerical traction (2, nf) for a coupled trial [u, r, p].

    t = t_n + T @ (x - x_n) + T_g @ (g - g_n) - Q_face @ N_face,
    with vector blocks interleaved by face and Q_face = W @ Q_cell componentwise.
    N_face is the grid's area-weighted normal in its fixed orientation. Contract
    only the in-plane stress block; retain full 3D material histories unchanged.
    Divergence supplies cell-orientation signs later, including on boundaries.

    The transfer must use the same grid and cell/face ordering as the operators.
    material_trial must have been evaluated for these same x, bc_values, and
    committed history. Only boundary entries of both old and new bc_values are
    used. Boundary Q_face uses its single adjacent cell, as enforced by transfer.

    No auxiliary solve, constitutive update, force residual, or commit is done
    here. The result owns its data; all inputs are preserved. In particular, t_n
    is supplied numerical history and is never rebuilt from cell stresses.
    """
    nc, nf = operators.num_cells, operators.num_faces
    if (transfer.num_cells, transfer.num_faces) != (nc, nf):
        raise ValueError("Transfer must match the operators' cell and face counts.")
    correction = material_trial.stress_correction
    normals = operators.grid.face_normals[:2]
    for name, values, shape in (
        ("x", x, (4 * nc,)),
        ("committed.x", committed.x, (4 * nc,)),
        ("committed.traction", committed.traction, (2, nf)),
        ("stress_correction", correction, (3, 3, nc)),
        ("face_normals", normals, (2, nf)),
    ):
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected finite {name} with shape {shape}.")
    bf = operators.boundary_faces
    for name, values in (("bc_values", bc_values), ("committed.bc_values", committed.bc_values)):
        if values.shape != (2, nf) or not np.all(np.isfinite(values[:, bf])):
            raise ValueError(f"Expected {name} with shape (2, {nf}), finite on boundary faces.")
    boundary_increment = np.zeros((2, nf))
    boundary_increment[:, bf] = bc_values[:, bf] - committed.bc_values[:, bf]
    elastic_increment = np.asarray(
        operators.T @ (x - committed.x)
        + operators.T_g @ boundary_increment.ravel(order="F"), dtype=np.float64,
    ).reshape((2, nf), order="F")
    face_correction = transfer.apply(correction)
    plastic_force = np.einsum("ijf,jf->if", face_correction[:2, :2], normals)
    return np.asarray(committed.traction + elastic_increment - plastic_force, dtype=np.float64)


@dataclass(eq=False)
class TpsaResidual:
    """One evaluated candidate, with owned arrays and independent material history.

    residual is the unscaled integrated [R_u, R_r, R_p] vector of length 4 * nc.
    trial stores the matching x, boundary data, strains, histories, and traction;
    it is not automatically accepted. stress_correction stores this increment's
    cell correction (3, 3, nc), available for the later Jacobian.
    """

    residual: NDArray[np.float64]
    trial: TpsaState
    stress_correction: NDArray[np.float64]


def assemble_global_residual(
    operators: TpsaOperators,
    x: NDArray[np.float64],
    bc_values: NDArray[np.float64],
    traction: NDArray[np.float64],
    *,
    sources: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Assemble the fully coupled [R_u, R_r, R_p] from supplied numerical tractions.

    R_u = D_u @ t - b_u,
    R_r = A_ru @ u + A_rr @ r - f_r,
    R_p = A_pu @ u + A_pp @ p - f_p,
    where f = b - div @ rhs_matrix @ g includes target boundary contributions.

    x has shape (4 * nc,), bc_values and integrated traction have shape (2, nf).
    Only boundary entries of bc_values are used. sources is the target integrated
    [body force, rotation source, pressure source] vector (4 * nc,), default zero.
    It is neither a density nor a load increment nor a boundary-shifted rhs.
    In particular, R_u subtracts b_u, never f_u: t already includes boundary data.

    All cell unknowns remain in the system, including cells adjacent to Dirichlet
    boundaries. No auxiliary solve, row elimination, scaling, or mutation is done.
    The caller must supply traction evaluated for this same candidate and load.
    """
    nc, nf = operators.num_cells, operators.num_faces
    for name, values, shape in (("x", x, (4 * nc,)), ("traction", traction, (2, nf))):
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected finite {name} with shape {shape}.")
    source_values = np.zeros(4 * nc) if sources is None else np.asarray(sources, dtype=np.float64)
    # assemble_rhs validates source shape/finiteness and masks unused boundary data.
    rhs = operators.assemble_rhs(bc_values, sources=source_values)
    u, r, p = x[:2 * nc], x[2 * nc:3 * nc], x[3 * nc:]
    result = np.empty(4 * nc)
    result[:2 * nc] = operators.D_u @ traction.ravel(order="F") - source_values[:2 * nc]
    result[2 * nc:3 * nc] = operators.A_ru @ u + operators.A_rr @ r - rhs[2 * nc:3 * nc]
    result[3 * nc:] = operators.A_pu @ u + operators.A_pp @ p - rhs[3 * nc:]
    return result


def evaluate_global_residual(
    operators: TpsaOperators,
    transfer: CellToFaceTransfer,
    material_points: Sequence[MaterialPoint],
    committed: TpsaState,
    x: NDArray[np.float64],
    bc_values: NDArray[np.float64],
    *,
    sources: NDArray[np.float64] | None = None,
) -> TpsaResidual:
    """Evaluate one coupled candidate from fixed committed history, without solving.

    Reconstruct strain, evaluate return maps, update face tractions, and assemble
    all three residual blocks using the supplied current [u, r, p]. Rotation and
    pressure need not satisfy their equations yet. sources has the integrated,
    absolute target-load convention of assemble_global_residual().

    Return an independent candidate state together with the residual and cell
    stress correction. Unused boundary entries are stored as zero in the result.
    Inputs, case material-point histories, and committed state remain unchanged,
    including if evaluation fails. No Newton iteration or acceptance is performed.
    Operators/transfer use the same fixed grid ordering and material coefficients.
    Residual blocks have different physical units; convergence scaling is a later
    solver choice, not part of this raw assembly.
    """
    material_trial = evaluate_material_trial(operators, material_points, committed, x, bc_values)
    traction = evaluate_face_traction(operators, transfer, committed, x, bc_values, material_trial)
    residual = assemble_global_residual(operators, x, bc_values, traction, sources=sources)
    boundary = np.zeros((2, operators.num_faces))
    bf = operators.boundary_faces
    boundary[:, bf] = bc_values[:, bf]
    trial = TpsaState(
        x=x, bc_values=boundary, epsilon=material_trial.epsilon,
        material_states=material_trial.material_states, traction=traction,
    )
    return TpsaResidual(residual, trial, material_trial.stress_correction)
