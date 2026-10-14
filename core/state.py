"""Owned total histories for incremental TPSA with retained face displacement."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from utils.state import TpsaState
from core.operators import RetainedFaceOperators
from core.residual import assemble_global_residual

from utils.state import (
    as_tpsa_state as as_tpsa_state,
)


class RetainedState:
    """Reuse TpsaState and retain BOTH numerical traction histories per interface.

    tpsa holds total cell u,r,p, face displacement, total strain, and 3D material
    histories. side_traction (2,num_sides) stores total outward integrated forces
    in operators.side order. body_force (2,num_cells) stores the absolute
    integrated load belonging to this state. All inputs are copied and validated.

    tpsa.traction is regenerated as the average of signed side tractions in the
    grid's fixed normal orientation. It is a compatibility/visualization field;
    only side_traction is used for retained-face equilibrium. Averaging must not
    erase a small accepted interior imbalance from subsequent load steps.

    Reconstruct this container with the SAME operators on restart. Geometry,
    side ordering and reference elasticity remain fixed throughout the history.
    Validation checks storage and kinematics, not equilibrium or yield admissibility.
    """

    def __init__(
        self, operators: RetainedFaceOperators, tpsa: TpsaState,
        side_traction: NDArray[np.float64], body_force: NDArray[np.float64],
    ) -> None:
        nc, nf = operators.num_cells, operators.num_faces
        side = np.array(side_traction, dtype=np.float64, copy=True)
        body = np.array(body_force, dtype=np.float64, copy=True)
        for name, values, shape in (
            ("side_traction", side, (2, operators.num_sides)),
            ("body_force", body, (2, nc)),
        ):
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"Expected finite {name} with shape {shape}.")
        if len(tpsa.material_states) != nc or tpsa.bc_values.shape != (2, nf):
            raise ValueError("Committed cell and face counts must match the retained operators.")
        if tpsa.face_displacement is None:
            raise ValueError("Retained states require total face_displacement.")
        faces = operators.side_faces
        count = np.bincount(faces, minlength=nf)
        oriented = np.vstack([
            np.bincount(faces, weights=component * operators.side_signs, minlength=nf) / count
            for component in side
        ])
        state = TpsaState(tpsa.x, tpsa.bc_values, tpsa.epsilon, tpsa.material_states,
                          oriented, face_displacement=tpsa.face_displacement)
        assert state.face_displacement is not None
        bf = operators.boundary_faces
        if not np.allclose(state.face_displacement[:, bf], state.bc_values[:, bf], rtol=1e-10, atol=1e-12):
            raise ValueError("Total face displacement must match the committed Dirichlet data.")
        state.bc_values[:, operators.free_faces] = 0.0
        total = operators.pack(state.x, state.face_displacement[:, operators.free_faces])
        reconstructed = operators.strain_increment(total, state.bc_values, np.zeros((2, nf)))
        if not np.allclose(state.epsilon, reconstructed, rtol=1e-8, atol=1e-10):
            raise ValueError("Committed total strain must match the stored face displacement.")
        self.tpsa, self.side_traction, self.body_force = state, side, body

    @classmethod
    def zeros(cls, operators: RetainedFaceOperators) -> RetainedState:
        """Undeformed, unstressed state with zero loads and total face histories."""
        return cls(operators, TpsaState.zeros(operators.grid, store_faces=True),
                   np.zeros((2, operators.num_sides)), np.zeros((2, operators.num_cells)))

    def copy(self) -> RetainedState:
        """Independent copy of every array and material history; no shared grid."""
        return deepcopy(self)


def assemble_state_residual(operators: RetainedFaceOperators, state: RetainedState) -> NDArray[np.float64]:
    """Measure TOTAL [R_u,R_r,R_p,R_f] using stored side tractions and loads.

    Reuse the linear balance assembler with total fields measured from zero.
    No return map is run and no stress-derived traction replaces the numerical
    history. This is the known old imbalance passed into the next Newton step.
    """
    faces = state.tpsa.face_displacement
    if faces is None:
        raise ValueError("Retained states require total face_displacement.")
    total = operators.pack(state.tpsa.x, faces[:, operators.free_faces])
    return assemble_global_residual(
        operators, total, state.tpsa.bc_values, np.zeros_like(state.tpsa.bc_values),
        state.side_traction, body_force_increment=state.body_force,
    )
