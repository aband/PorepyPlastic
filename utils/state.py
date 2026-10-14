"""Shared total fields and independent material histories for TPSA formulations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from porepy.grids.grid import Grid

from material.material_state import MaterialState


@dataclass(eq=False)
class TpsaState:
    """Owned data for one committed state or one independent trial candidate.

    x stores [u, r, p], with the two displacement components interleaved by cell.
    bc_values stores the boundary data belonging to this state, not the next load.
    epsilon contains total 3D plane strain; material_states retain full 3D stress
    and plastic history. traction contains integrated numerical face tractions in
    the grid's fixed normal orientation, distinct from cell material stresses.
    face_displacement retains the shared face history for the updated coupled
    formulation; it is optional for the original formulation. Coupled Newton
    packs epsilon independently without changing the layout of x.

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

    face_displacement: NDArray[np.float64] | None = None  # Coupled formulation history (2, nf).

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
        if self.face_displacement is not None:
            face = np.array(self.face_displacement, dtype=np.float64, copy=True)
            if face.shape != (2, nf) or not np.all(np.isfinite(face)):
                raise ValueError("face_displacement must be finite with shape (2, nf).")
            self.face_displacement = face
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
    def zeros(cls, grid: Grid, *, store_faces: bool = False) -> TpsaState:
        """Create an undeformed, unstressed state with zero old boundary data.

        Assumes zero initial sources and an admissible unstressed material.
        Set store_faces=True for the coupled formulation. Geometry need not be
        computed. Nonzero initial states must be supplied
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
            face_displacement=np.zeros((2, nf)) if store_faces else None,
        )

    def copy(self) -> TpsaState:
        """Copy all arrays and material histories for an independent candidate."""
        return deepcopy(self)


class StateWithTpsa(Protocol):
    """A formulation-specific history exposing its common total fields."""

    @property
    def tpsa(self) -> TpsaState: ...


def as_tpsa_state(state: TpsaState | StateWithTpsa) -> TpsaState:
    """View common total fields without copying or discarding solver histories.

    Use the complete formulation-specific state for equilibrium and restart.
    """
    return state if isinstance(state, TpsaState) else state.tpsa
