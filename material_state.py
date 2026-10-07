"""Committed and trial history for a single material point."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import tensor
from J2 import _materialProperty, vonMisesModel
from scalar_hardening import HardeningParameters, ScalarHardeningLaw


@dataclass
class MaterialState:
    """Stress and history variables used by the J2 return map.

    Defaults describe an undeformed point with full 3D tensors. ``alpha``
    is the accumulated equivalent plastic strain.

    Use ``trial = committed.copy()`` for a trial update. Replace the
    committed state only after the global step has converged.
    """

    stress: tensor.stress = field(
        default_factory=lambda: tensor.stress.zeros((3, 3))
    )
    plastic_strain: tensor.strain = field(
        default_factory=lambda: tensor.strain.zeros((3, 3))
    )
    backstress: tensor.stress = field(
        default_factory=lambda: tensor.stress.zeros((3, 3))
    )
    alpha: float = 0.0

    def copy(self) -> MaterialState:
        """Return an independent state, including copies of all tensors."""
        return deepcopy(self)


@dataclass
class MaterialPoint:
    """Manage one 3D point with a supplied model and hardening laws.

    Commit only after global equilibrium converges.
    """

    material: _materialProperty
    parameters: HardeningParameters
    model: vonMisesModel
    K_law: ScalarHardeningLaw
    H_law: ScalarHardeningLaw
    committed: MaterialState = field(default_factory=MaterialState)
    trial: MaterialState | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.committed = self.committed.copy()

    def update(
        self, strain_increment: tensor.strain, tol: float = 1.0e-10,
        *, rtol: float = 1.0e-12,
    ) -> tuple[MaterialState, None]:
        """Return trial history and a reserved ``None`` tangent slot.

        ``strain_increment`` is measured from the last converged load step,
        not from the previous global iteration. Every call starts from the
        same committed history until ``commit()`` is called. ``tol`` and
        ``rtol`` control absolute and relative local residual convergence.
        Analytical coupling uses ``model.consistent_tangent`` on demand, reusing
        the returned history instead of computing a tangent at every residual call.
        """
        self.trial = None
        if strain_increment.shape != (3, 3) or not strain_increment.is_symmetric:
            raise ValueError("Expected a symmetric 3D strain increment.")

        # Copy also isolates history returned unchanged by an elastic step.
        state_n = self.committed.copy()
        stress_increment = self.material.elastic_tensor.double_contract(strain_increment)
        sigma_trial = state_n.stress + tensor.stress(
            stress_increment.to_numpy(copy=False)
        )
        sigma, plastic_strain, backstress, alpha, _ = self.model.radial_return_map(
            sigma_trial=sigma_trial,
            plastic_strain_n=state_n.plastic_strain,
            backstress_n=state_n.backstress,
            alpha_n=state_n.alpha,
            material=self.material,
            K_law=self.K_law,
            H_law=self.H_law,
            parameters=self.parameters,
            tol=tol,
            rtol=rtol,
        )
        # The coupling Jacobian evaluates model.consistent_tangent on demand,
        # reusing this trial history without repeating the return map.
        tangent = None
        self.trial = MaterialState(sigma, plastic_strain, backstress, alpha)
        return self.trial, tangent

    def commit(self) -> None:
        """Accept the latest successful trial after global convergence."""
        if self.trial is None:
            raise RuntimeError("No successful trial update to commit.")
        self.committed = self.trial.copy()
        self.trial = None

    def rollback(self) -> None:
        """Discard a trial after a rejected step, preserving committed history."""
        self.trial = None
