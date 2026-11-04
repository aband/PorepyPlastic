from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core.layout import DofLayout, FloatArray
from material import tensor
from material.J2_batch import ReturnMapBatch
from material.material_state import MaterialState

@dataclass(eq=False)
class MaterialHistory:
    """Owned cellwise material history at a converged load step.

    epsilon: total strain, shape (3, 3, num_cells).
    stress, plastic_strain, backstress: full 3D tensors, shape (3, 3, num_cells).
    alpha: accumulated equivalent plastic strain, shape (num_cells,).

    Construction validates and copies every array. Keep this object
    unchanged during Newton iterations.
    """

    dim: int
    epsilon: FloatArray
    stress: FloatArray
    plastic_strain: FloatArray
    backstress: FloatArray
    alpha: FloatArray

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError("Expected dimension 2 or 3.")

        values = np.asarray(self.epsilon)
        nc = values.shape[2] if values.ndim == 3 else 0

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

        for name in ("stress", "plastic_strain", "backstress"):
            history = np.asarray(getattr(self, name))
            if (
                history.shape != (3, 3, nc)
                or history.dtype.kind not in ("f", "i", "u")
                or not np.all(np.isfinite(history))
                or not np.allclose(
                    history, history.swapaxes(0, 1), rtol=1e-10, atol=1e-14,
                )
            ):
                raise ValueError(
                    f"Material {name} must be finite symmetric with shape (3, 3, {nc})."
                )
            setattr(self, name, np.array(history, dtype=np.float64, copy=True))

        alpha = np.asarray(self.alpha)
        if (
            alpha.shape != (nc,)
            or alpha.dtype.kind not in ("f", "i", "u")
            or not np.all(np.isfinite(alpha))
            or np.any(alpha < 0)
        ):
            raise ValueError(
                f"Material alpha must be finite and nonnegative with shape ({nc},)."
            )
        self.alpha = np.array(alpha, dtype=np.float64, copy=True)

    @property
    def num_cells(self) -> int:
        return self.alpha.size

    @classmethod
    def zeros(cls, layout: DofLayout) -> MaterialHistory:
        """Undeformed, unstressed initial material history."""
        tensors = np.zeros((3, 3, layout.num_cells))
        return cls(
            dim=layout.dim,
            epsilon=tensors,
            stress=tensors,
            plastic_strain=tensors,
            backstress=tensors,
            alpha=np.zeros(layout.num_cells),
        )

    @classmethod
    def from_states(
        cls,
        dim: int,
        epsilon: FloatArray,
        material_states: Sequence[MaterialState],
    ) -> MaterialHistory:
        """Stack one MaterialState per cell into the array layout."""
        if len(material_states) < 1:
            raise ValueError("Expected one material state per cell.")

        def stacked(name: str) -> FloatArray:
            return np.stack(
                [getattr(state, name).to_numpy(copy=False) for state in material_states],
                axis=2,
            )

        return cls(
            dim=dim,
            epsilon=epsilon,
            stress=stacked("stress"),
            plastic_strain=stacked("plastic_strain"),
            backstress=stacked("backstress"),
            alpha=np.array([state.alpha for state in material_states], dtype=np.float64),
        )

    @property
    def material_states(self) -> list[MaterialState]:
        """Independent per-cell copies, for inspection and per-cell reference code."""
        return [
            MaterialState(
                stress=tensor.stress(self.stress[:, :, cell]),
                plastic_strain=tensor.strain(self.plastic_strain[:, :, cell]),
                backstress=tensor.stress(self.backstress[:, :, cell]),
                alpha=float(self.alpha[cell]),
            )
            for cell in range(self.num_cells)
        ]

    def copy(self) -> MaterialHistory:
        return MaterialHistory(
            self.dim,
            self.epsilon,
            self.stress,
            self.plastic_strain,
            self.backstress,
            self.alpha,
        )

@dataclass(eq=False)
class MaterialTrial:
    """Cellwise constitutive trial for one candidate; nothing is committed.

    history: trial total strain and returned history, ready to commit.
    stress_correction: elastic predictor minus returned stress, (3, 3, nc).
    return_map: the batched return-map quantities reused by the Jacobian.
    """

    history: MaterialHistory
    stress_correction: FloatArray
    return_map: ReturnMapBatch

    @property
    def epsilon(self) -> FloatArray:
        return self.history.epsilon

    @property
    def material_states(self) -> list[MaterialState]:
        return self.history.material_states

    @property
    def elastic_predictor(self) -> FloatArray:
        return self.return_map.elastic_predictor

@dataclass(eq=False)
class SolutionState:
    """Owned snapshot of the accepted total solution.

    Face and side ordering must match the residual assembler.
    Keep this state unchanged during Newton iterations.
    """

    # Total cell unknowns [u, r, p].
    x: FloatArray                       # (layout.cell_size,)

    # Total displacement on ALL faces, including Dirichlet faces.
    face_displacement: FloatArray       # (dim, num_faces)

    # Total outward INTEGRATED traction for each cell-face side.
    side_traction: FloatArray           # (dim, num_sides)

    # Total strain and committed constitutive states.
    material: MaterialHistory

    # Accepted outward INTEGRATED Neumann tractions.
    # Columns follow assembler.neumann_faces.
    neumann_values: FloatArray          # (dim, num_neumann_faces)

    # Accepted integrated sources [b_u, b_r, b_p].
    cell_source: FloatArray             # (layout.cell_size,)

    def __post_init__(self) -> None:
        """Own the supplied arrays and material history."""
        for name in (
            "x",
            "face_displacement",
            "side_traction",
            "neumann_values",
            "cell_source",
        ):
            values = np.asarray(getattr(self, name))

            if values.dtype.kind not in ("f", "i", "u"):
                raise ValueError(f"{name} must contain real numeric values.")

            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain finite values.")

            setattr(
                self,
                name,
                np.array(values, dtype=np.float64, copy=True),
            )

        self.material = self.material.copy()

    @classmethod
    def zeros(
        cls,
        layout: DofLayout,
        *,
        num_sides: int,
        num_neumann_faces: int,
    ) -> SolutionState:
        """Undeformed, unstressed state with zero applied loads."""
        return cls(
            x=np.zeros(layout.cell_size),
            face_displacement=np.zeros(
                (layout.dim, layout.num_faces)
            ),
            side_traction=np.zeros((layout.dim, num_sides)),
            material=MaterialHistory.zeros(layout),
            neumann_values=np.zeros(
                (layout.dim, num_neumann_faces)
            ),
            cell_source=np.zeros(layout.cell_size),
        )

    def copy(self) -> SolutionState:
        """Return an independent solution snapshot."""
        return SolutionState(
            x=self.x,
            face_displacement=self.face_displacement,
            side_traction=self.side_traction,
            material=self.material,
            neumann_values=self.neumann_values,
            cell_source=self.cell_source,
        )

#@dataclass(eq=False)
#class LoadStepResult:
#    load_factor: float
#    state: SolutionState
#    newton: NewtonResult


#class LoadStepController(_LoadStepSchedule[LoadStepResult]):
#    def __init__(
#        self,
#        assembler: ResidualAssembler,
#        committed: SolutionState,
#        reference_dirichlet: FloatArray,
#        *,
#        reference_neumann: FloatArray | None = None,
#        reference_cell_source: FloatArray | None = None,
#        initial_load_factor: float = 0.0,
#        max_iterations: int = 20,
#        atol: BlockTolerances = (1e-5, 1e-12, 1e-12, 1e-5),
#        rtol: BlockTolerances = (1e-8, 1e-8, 1e-8, 1e-8),
#    ) -> None:
#        # Validate and copy inputs.
#        # Store one assembler and one AnalyticalJacobian.
#        # Initialize accepted state, factor, records and attempts.
#        ...
#
#    def _validate_state(self, state: SolutionState) -> None:
#        ...
#
#    def advance(self, load_factor: float) -> LoadStepResult:
#        ...