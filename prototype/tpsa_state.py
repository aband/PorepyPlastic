"""Dimension-aware state for prototype, preserving the shared 2D state API."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from porepy.grids.grid import Grid

from material.material_state import MaterialState
from utils.state import TpsaState as PlaneStrainState


class TpsaState(PlaneStrainState):
    """Owned [u,r,p] and face histories in 2D plane strain or spatial 3D.

    Dimension is inferred from bc_values. The inherited constructor and fields
    preserve compatibility with existing 2D callers and as_tpsa_state(). In 3D,
    x has seven entries per cell (three u, three r, one p, in field blocks);
    vector fields have three rows and total strain is a full symmetric tensor.
    """

    @property
    def dim(self) -> int:
        return int(self.bc_values.shape[0])

    def __post_init__(self) -> None:
        boundary = np.asarray(self.bc_values)
        if boundary.ndim != 2 or boundary.shape[0] not in (2, 3):
            raise ValueError("Expected boundary array (dim, num_faces), dim=2 or 3.")
        # Check real inputs before either branch converts to float.
        for name in ("x", "bc_values", "epsilon", "traction", "face_displacement"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).dtype.kind not in ("f", "i", "u"):
                raise ValueError(f"{name} must contain real numeric values.")
        if boundary.shape[0] == 2:
            super().__post_init__()
            return

        nc, nf = len(self.material_states), boundary.shape[1]
        if nc < 1 or nf < 1:
            raise ValueError("Expected nonempty cell and face histories.")
        for name, shape in {
            "x": (7 * nc,), "bc_values": (3, nf),
            "epsilon": (3, 3, nc), "traction": (3, nf),
            "face_displacement": (3, nf),
        }.items():
            raw = getattr(self, name)
            if name == "face_displacement" and raw is None:
                continue
            value = np.array(raw, dtype=np.float64, copy=True)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}.")
            setattr(self, name, value)
        if not np.allclose(self.epsilon, self.epsilon.swapaxes(0, 1), rtol=1e-10, atol=1e-14):
            raise ValueError("Total strain must be symmetric.")
        for state in self.material_states:
            if not np.isfinite(state.alpha) or state.alpha < 0:
                raise ValueError("Material alpha must be finite and nonnegative.")
            for name in ("stress", "plastic_strain", "backstress"):
                tensor = getattr(state, name).to_numpy(copy=False)
                if (tensor.shape != (3, 3) or not np.all(np.isfinite(tensor))
                        or not np.allclose(tensor, tensor.T, rtol=1e-10, atol=1e-14)):
                    raise ValueError(f"Material {name} must be a finite symmetric 3D tensor.")
        self.material_states = [state.copy() for state in self.material_states]

    @classmethod
    def zeros(cls, grid: Grid, *, store_faces: bool = False) -> TpsaState:
        """Create a virgin 2D/3D state, optionally storing total face displacement."""
        nd, nc, nf = grid.dim, grid.num_cells, grid.num_faces
        if nd not in (2, 3) or nc < 1 or nf < 1:
            raise ValueError("Expected a nonempty 2D or 3D grid.")
        nr = 1 if nd == 2 else 3
        return cls(
            x=np.zeros((nd + nr + 1) * nc), bc_values=np.zeros((nd, nf)),
            epsilon=np.zeros((3, 3, nc)),
            material_states=[MaterialState() for _ in range(nc)],
            traction=np.zeros((nd, nf)),
            face_displacement=np.zeros((nd, nf)) if store_faces else None,
        )

    def copy(self) -> TpsaState:
        return deepcopy(self)
