from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.layout import DofLayout, FloatArray
from material.material_state import MaterialState

@dataclass(eq=False)
class MaterialHistory:
    """Owned material history at a converged load step.

    epsilon: total strain, shape (3, 3, num_cells).
    material_states: one full 3D material state per cell.

    Keep this object unchanged during Newton iterations.
    """

    dim: int
    epsilon: FloatArray
    material_states: list[MaterialState]

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError("Expected dimension 2 or 3.")

        nc = len(self.material_states)
        values = np.asarray(self.epsilon)

        if nc < 1 or values.shape != (3, 3, nc):
            raise ValueError(
                "Expected strain with shape (3, 3, num_cells)."
            )

        if values.dtype.kind not in ("f", "i", "u"):
            raise ValueError("Strain must contain real numeric values.")

        if not np.all(np.isfinite(values)):
            raise ValueError("Strain must contain finite values.")

        if not np.allclose(
            values, values.swapaxes(0, 1),
            rtol=0.0, atol=1e-14,
        ):
            raise ValueError("Strain must be symmetric.")

        if self.dim == 2 and (
            not np.allclose(values[2, :, :], 0.0, atol=1e-14)
            or not np.allclose(values[:, 2, :], 0.0, atol=1e-14)
        ):
            raise ValueError("2D total strain must satisfy plane strain.")

        self.epsilon = np.array(values, dtype=np.float64, copy=True)
        self.material_states = [
            state.copy() for state in self.material_states
        ]

    @classmethod
    def zeros(cls, layout: DofLayout) -> MaterialHistory:
        """Undeformed, unstressed initial material history."""
        return cls(
            dim=layout.dim,
            epsilon=np.zeros((3, 3, layout.num_cells)),
            material_states=[
                MaterialState() for _ in range(layout.num_cells)
            ],
        )

    def copy(self) -> MaterialHistory:
        return MaterialHistory(
            self.dim, self.epsilon, self.material_states
        )