"""Cellwise material data and batched J2 updates for the retained solver."""

from collections.abc import Sequence
from dataclasses import fields

import numpy as np

from core.layout import FloatArray, IntArray
from core.state import MaterialHistory, MaterialTrial
from material.J2 import vonMisesModel
from material.J2_batch import (
    BatchReturnMapConvergenceError,
    radial_return_batch,
    tangent_correction_batch,
)
from material.material_state import MaterialPoint
from material.scalar_hardening import HardeningParameters
from utils.newton_utils import LocalUpdateError


class CellMaterial:
    """Per-cell elastic moduli and J2 hardening data, fixed at construction.

    Built once from one MaterialPoint per cell. Every point must use the J2
    (von Mises) model and the same hardening-law functions, which must accept
    array alpha; moduli and hardening parameters may vary by cell. Values are
    read once, like the elastic TPSA operators, so later changes to the
    points are not seen.
    """

    def __init__(self, material_points: Sequence[MaterialPoint]) -> None:
        points = tuple(material_points)
        nc = len(points)
        if nc < 1:
            raise ValueError("Expected one material point per cell.")

        if not all(isinstance(point.model, vonMisesModel) for point in points):
            raise ValueError("Batched material updates support the J2 model only.")

        K_law, H_law = points[0].K_law, points[0].H_law
        if any(point.K_law is not K_law or point.H_law is not H_law for point in points):
            raise ValueError("All material points must share the same hardening laws.")

        self.num_cells = nc
        self.K_law, self.H_law = K_law, H_law
        self.mu = np.array(
            [point.material.isotropic_shear_modulus for point in points], dtype=np.float64,
        )
        self.lam = np.array(
            [point.material.lame_parameter for point in points], dtype=np.float64,
        )
        if not np.all(np.isfinite(self.mu)) or np.any(self.mu <= 0):
            raise ValueError("Shear moduli must be finite and positive.")
        if not np.all(np.isfinite(self.lam)):
            raise ValueError("Lame parameters must be finite.")

        self.parameters = HardeningParameters(**{
            item.name: np.array(
                [getattr(point.parameters, item.name) for point in points], dtype=np.float64,
            )
            for item in fields(HardeningParameters)
        })

        for name, law in (("K_law", K_law), ("H_law", H_law)):
            try:
                outputs = law(np.zeros(nc), self.parameters)
                valid = all(
                    np.broadcast_shapes(np.shape(value), (nc,)) == (nc,)
                    and np.all(np.isfinite(value))
                    for value in outputs
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must accept array alpha.") from error
            if not valid:
                raise ValueError(f"{name} must return finite values per cell.")

    def evaluate(
        self,
        committed: MaterialHistory,
        strain_increment: FloatArray,
        *,
        tol: float = 1.0e-10,
        rtol: float = 1.0e-12,
    ) -> MaterialTrial:
        """Return-map every cell from committed history; nothing is committed."""
        nc = self.num_cells
        for name in ("epsilon", "stress", "plastic_strain", "backstress"):
            if np.shape(getattr(committed, name)) != (3, 3, nc):
                raise ValueError(f"Committed {name} must have shape (3, 3, {nc}).")
        if np.shape(committed.alpha) != (nc,):
            raise ValueError(f"Committed alpha must have shape ({nc},).")

        increment = np.asarray(strain_increment, dtype=np.float64)
        if increment.shape != (3, 3, nc) or not np.all(np.isfinite(increment)):
            raise ValueError(f"Strain increment must be finite with shape (3, 3, {nc}).")

        try:
            result = radial_return_batch(
                committed.stress,
                committed.plastic_strain,
                committed.backstress,
                committed.alpha,
                increment,
                mu=self.mu,
                lam=self.lam,
                parameters=self.parameters,
                K_law=self.K_law,
                H_law=self.H_law,
                tol=tol,
                rtol=rtol,
            )
        except BatchReturnMapConvergenceError as error:
            raise LocalUpdateError(
                f"Return map failed in cell {error.cells[0]} "
                f"({error.cells.size} cell(s) in total)."
            ) from error

        with np.errstate(over="ignore", invalid="ignore"):
            correction = result.elastic_predictor - result.stress
            total = committed.epsilon + increment

        if not np.all(np.isfinite(correction)):
            raise ValueError("Stress corrections must remain finite.")
        if not np.all(np.isfinite(total)):
            raise ValueError("Total strain must remain finite.")

        history = MaterialHistory(
            committed.dim,
            total,
            result.stress,
            result.plastic_strain,
            result.backstress,
            result.alpha,
        )
        return MaterialTrial(history, correction, result)

    def tangent_correction(
        self, trial: MaterialTrial,
    ) -> tuple[IntArray, FloatArray, FloatArray]:
        """Plastic cells and coefficients of C_e - C_alg = c_dev P_dev + c_nn n(x)n."""
        return tangent_correction_batch(
            trial.return_map,
            mu=self.mu,
            parameters=self.parameters,
            K_law=self.K_law,
            H_law=self.H_law,
        )
