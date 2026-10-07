"""
Define a plastic material, which should include 

1. yield function
2. plastic potential (if not associative)
3. return map/ implicit integrator

hardening law is obtained
(Temperorarily, data type will not match exactly what used in 
porepy. It will be modified later if integrated with
main porepy code. Expect tensor object.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Callable
from abc import ABC

import numpy as np
import porepy as pp

from scalar_hardening import HardeningParameters, ScalarHardeningLaw
from tensor import Tensor, stress, strain

@dataclass
class ReturnStates:
    plastic_stress : stress
    is_plastic     : bool
    iteratoins     : int
    plastic_multiplier : float

class _materialProperty:
    """Helper class to store material properties."""

    young_modulus: float
    poisson_ratio: float

    @property
    def isotropic_shear_modulus(self) -> float:
        """Shear modulus G for isotropic material."""
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K."""
        return self.young_modulus / (
            3.0 * (1.0 - 2.0 * self.poisson_ratio)
        )

    @property
    def elastic_tensor(self) -> Tensor:
        """Three-dimensional isotropic stiffness acting on symmetric strain."""
        identity = np.eye(3)
        return Tensor(
            self.lame_parameter * np.einsum("ij,kl->ijkl", identity, identity)
            + self.isotropic_shear_modulus * (
                np.einsum("ik,jl->ijkl", identity, identity)
                + np.einsum("il,jk->ijkl", identity, identity)
            )
        )

    def elastic_tensor_tpsa(self, num_cells: int = 1) -> pp.FourthOrderTensor:
        """Return PorePy stiffness with uniform elastic moduli over the cells.

        Use as TPSA's ``fourth_order_tensor`` parameter. Its ``values`` array
        has shape ``(9, 9, num_cells)``.
        """
        return pp.FourthOrderTensor(
            mu=np.full(num_cells, self.isotropic_shear_modulus),
            lmbda=np.full(num_cells, self.lame_parameter),
        )

    def compare_elastic_tensors(
        self,
        num_cells: int = 1,
        *,
        rtol: float = 1.0e-10,
        atol: float = 1.0e-12,
    ) -> tuple[bool, float]:
        """Return (matches, maximum absolute error) across all cells.

        Compare every component using ``abs(tpsa - reference) <=
        atol + rtol * abs(reference)``. Flatten each tensor index pair in
        column-major order to match PorePy's 9-by-9 representation.
        """
        if num_cells < 1:
            raise ValueError("num_cells must be positive.")

        reference = self.elastic_tensor.to_numpy(copy=False).reshape(9, 9, order="F")
        reference = reference[:, :, np.newaxis]
        tpsa = self.elastic_tensor_tpsa(num_cells).values
        matches = np.allclose(tpsa, reference, rtol=rtol, atol=atol)
        max_error = np.max(np.abs(tpsa - reference))
        return bool(matches), float(max_error)

    @property
    def lame_parameter(self) -> float:
        """First Lamé parameter lambda."""
        return (
            self.young_modulus
            * self.poisson_ratio
            / (
                (1.0 + self.poisson_ratio)
                * (1.0 - 2.0 * self.poisson_ratio)
            )
        )

class abstractMaterialEvaluate(ABC):
    """Class of the common parts of the yield functions."""

    def __init__(self, name:str) -> None:
        """Assign name tag to the material class."""
        self.name = name;

    #@abstractmethod
    #def set_current_sigma(self, sigma:np.ndarray):
    #    pass

class vonMisesModel(abstractMaterialEvaluate):
    """
    Evaluate of a given material point with von mises yielding criteria.
    Implemented at each material point.
    """

    def evaluate_equivalent_stress(self, 
                                   deviatoric_stress:tensor.stress) -> float:
        """Compute equivalent von mises stress."""
        #return np.sqrt(3.0/2.0)*np.linalg.norm(stress.deviatoric)
        return np.sqrt(1.5*
                       np.einsum("ij,ij->",
                                 deviatoric_stress.to_numpy(),
                                 deviatoric_stress.to_numpy()))

    def consistency_parameter(self,
                              xi_trial_norm: float,
                              alpha_n      : float,
                              mu           : float,
                              parameters   : HardeningParameters,
                              K_law        : ScalarHardeningLaw,
                              H_law        : ScalarHardeningLaw,
                              tolerance    : float = 1.0e-10,
                              max_iteration: int = 20,
                              *,
                              relative_tolerance: float = 1.0e-12,
                              ) -> float:
        """
        Determine consistency parameter with a local Newton iteration.

        Converged when |R| <= tolerance + relative_tolerance * stress_scale,
        with stress_scale = max(|xi_trial_norm|, |sqrt(2/3) * K(alpha_n)|).
        ``tolerance`` has stress units; ``relative_tolerance`` is dimensionless.

        xi_trial_norm: norm of the shifted trial deviatoric stress
        alpha_n: accumulated equivalent plastic strain
        mu: shear modulus
        """
        if (
            not np.isfinite(tolerance) or tolerance < 0.0
            or not np.isfinite(relative_tolerance) or relative_tolerance < 0.0
        ):
            raise ValueError("Convergence tolerances must be finite and non-negative.")

        dgamma    = 0.0              # consistency parameter
        c         = np.sqrt(2.0/3.0) # const parameter

        # Evaluate yield criterion
        K_n, _ = K_law(alpha_n, parameters)

        if xi_trial_norm - c*K_n <= 0.0:
            return dgamma

        H_n, _ = H_law(alpha_n, parameters)
        stress_scale = max(abs(xi_trial_norm), abs(c * K_n))
        residual_tolerance = tolerance + relative_tolerance * stress_scale

        for _ in range(max_iteration):
            alpha = alpha_n + c*dgamma

            # Evaluate hardening laws
            K, K_prime = K_law(alpha, parameters)
            H, H_prime = H_law(alpha, parameters)

            residual = (
                xi_trial_norm
                - 2.0 * mu * dgamma
                - c * (K + H - H_n)
                )

            if np.isfinite(residual) and abs(residual) <= residual_tolerance:
                return dgamma

            derivative = (
                -2.0 * mu
                - (2.0 / 3.0) * (K_prime + H_prime)
            )

            dgamma = max(
                0.0,
                dgamma - residual / derivative,
            )

        # Check the result of the final Newton update
        # For the case when iteration reaches the limit, but not yet converged
        alpha = alpha_n + c*dgamma
        K, _ = K_law(alpha, parameters)
        H, _ = H_law(alpha, parameters)

        residual = (
            xi_trial_norm
            - 2.0 * mu * dgamma
            - c * (K + H - H_n)
            )

        if not np.isfinite(residual) or abs(residual) > residual_tolerance:
            raise RuntimeError(
                f"Local Newton iteration did not converge: R={residual:.3e}, "
                f"tolerance={residual_tolerance:.3e}"
            )

        return dgamma

    def radial_return_map(
            self,
            sigma_trial: stress,
            plastic_strain_n: strain,
            backstress_n: stress,
            alpha_n: float,
            material: _materialProperty,
            K_law: ScalarHardeningLaw,
            H_law: ScalarHardeningLaw,
            parameters: HardeningParameters,
            tol: float = 1.0e-10,
            *,
            rtol: float = 1.0e-12,
    ) -> tuple[stress, strain, stress, float, float]: 
        """
        Perform return map integration if yield function is
        evaluated positive. The return map evaluation is point-wise.
        ``tol`` and ``rtol`` are the absolute and relative residual tolerances.
        """

        # Compute shear modulus
        mu = material.isotropic_shear_modulus

        s_trial: stress = sigma_trial.deviatoric()
        xi_trial: stress = s_trial - backstress_n.deviatoric()

        xi_trial_norm = np.sqrt(xi_trial.inner(xi_trial))

        dgamma = self.consistency_parameter(
            xi_trial_norm=xi_trial_norm,
            alpha_n=alpha_n,
            mu=mu,
            parameters=parameters,
            K_law=K_law,
            H_law=H_law,
            tolerance=tol,
            relative_tolerance=rtol,
        )

        if dgamma == 0.0:
            return (
                sigma_trial,
                plastic_strain_n,
                backstress_n,
                alpha_n,
                dgamma,
            )

        direction = xi_trial / xi_trial_norm
        c = np.sqrt(2.0 / 3.0)
        alpha = alpha_n + c * dgamma

        H_n, _ = H_law(alpha_n, parameters)
        H, _ = H_law(alpha, parameters)

        sigma = sigma_trial - (2.0 * mu * dgamma) * direction
        plastic_strain = (
            plastic_strain_n
            + dgamma * strain(direction.to_numpy(copy=False))
        )
        backstress = backstress_n + c * (H - H_n) * direction

        return sigma, plastic_strain, backstress, alpha, dgamma

    def consistent_tangent(
        self,
        sigma_trial: stress,
        backstress_n: stress,
        alpha: float,
        dgamma: float,
        material: _materialProperty,
        K_law: ScalarHardeningLaw,
        H_law: ScalarHardeningLaw,
        parameters: HardeningParameters,
    ) -> Tensor:
        """Full 3D derivative of radial_return_map on its selected smooth branch.

        sigma_trial, updated alpha and dgamma must belong to the SAME return map,
        with fixed old history and isotropic elastic moduli. Our multiplier uses
        delta_epsilon_p = dgamma*n, ||n||=1, delta_alpha=sqrt(2/3)*dgamma.
        Thus D = 2*mu + (2/3)*(K_prime(alpha)+H_prime(alpha)) and
            C_alg = C_e - 4*mu**2 * [dgamma/q*P_dev
                                    + (1/D-dgamma/q)*(n outer n)].
        Here q is ||dev(sigma_trial)-dev(backstress_n)|| (not von Mises stress).
        The two correction terms include flow-direction and multiplier changes.
        Nonlinear hardening derivatives are evaluated at the updated alpha.

        The elastic branch returns C_e exactly, including at zero deviatoric
        stress. At a yield switch the classical derivative is not unique; dgamma
        selects the branch actually returned by the local solver. Components are
        unscaled tensor entries, not engineering shear or Mandel coordinates.
        """
        if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(dgamma) or dgamma < 0:
            raise ValueError("Tangent alpha and dgamma must be finite and nonnegative.")
        for value in (sigma_trial.to_numpy(copy=False), backstress_n.to_numpy(copy=False)):
            if (value.shape != (3, 3) or not np.all(np.isfinite(value))
                    or not np.allclose(value, value.T, rtol=1e-10, atol=1e-14)):
                raise ValueError("Tangent requires finite symmetric 3D stresses.")
        mu, bulk = material.isotropic_shear_modulus, material.bulk_modulus
        if not np.isfinite(mu) or not np.isfinite(bulk) or mu <= 0 or bulk <= 0:
            raise ValueError("Tangent requires finite positive shear and bulk moduli.")
        elastic = material.elastic_tensor
        if dgamma == 0.0:
            return elastic
        shifted = (sigma_trial.deviatoric() - backstress_n.deviatoric()).to_numpy()
        q = float(np.linalg.norm(shifted))
        if not np.isfinite(q) or q <= 0:
            raise ValueError("Plastic tangent requires a nonzero shifted trial stress.")
        _, K_prime = K_law(alpha, parameters)
        _, H_prime = H_law(alpha, parameters)
        denominator = 2 * mu + (2 / 3) * (K_prime + H_prime)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Plastic tangent requires a finite positive consistency denominator.")
        identity = np.eye(3)
        deviatoric = 0.5 * (
            np.einsum("ik,jl->ijkl", identity, identity)
            + np.einsum("il,jk->ijkl", identity, identity)
        ) - np.einsum("ij,kl->ijkl", identity, identity) / 3
        direction = shifted / q
        correction = 4 * mu**2 * (
            (dgamma / q) * deviatoric
            + (1 / denominator - dgamma / q)
            * np.einsum("ij,kl->ijkl", direction, direction)
        )
        tangent = elastic.to_numpy() - correction
        if not np.all(np.isfinite(tangent)):
            raise ValueError("Algorithmic tangent must be finite.")
        return Tensor(tangent)

# Save for later
#class mohrCoulombEvaluate(abstractMaterialEvaluate):
#    """Evaluate of a given material point with mohr coulomb yielding criteria."""
    
#class drukerPragerEvaluate(abstractMaterialEvaluate):
#    """Evaluate of a given material point with mohr coulomb yielding criteria."""