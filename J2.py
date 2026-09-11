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

from scalar_hardening import HardeningParameters, ScalarHardeningLaw
from tensor import stress, strain

@dataclass
class ReturnStates:
    plastic_stress : tensor.stress
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
                              ) -> float:
        """
        Determine consistency parameter with a local Newton iteration.
        This newton iteration is simple with constant derivative.

        xi_trial_norm: norm of the deviatoric stress
        alpha_n: equivalent plastic stress
        mu: lame parameter
        """

        dgamma    = 0.0              # consistency parameter
        c         = np.sqrt(2.0/3.0) # const parameter

        # Evaluate yield criterion
        K_n, _ = K_law(alpha_n, parameters)

        if xi_trial_norm - c*K_n <= 0.0:
            return dgamma

        H_n, _ = H_law(alpha_n, parameters)

        # Initialization of the newton iteration parameters
        iteration = 0                # current iteration count
        residual  = 10               # residual

        while (iteration < max_iteration and abs(residual) > tolerance):
            iteration = iteration + 1
            alpha     = alpha_n + c*dgamma

            # Evaluate hardening laws
            K, K_prime = K_law(alpha, parameters)
            H, H_prime = H_law(alpha, parameters)

            residual = (
                xi_trial_norm
                - 2.0 * mu * dgamma
                - c * (K + H - H_n)
                )

            if abs(residual) <= tolerance:
                return dgamma

            derivative = (
                -2.0 * mu
                - (2.0 / 3.0) * (K_prime + H_prime)
            )

            dgamma = max(
                0.0,
                dgamma - residual / derivative,
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
    ) -> tuple[stress, strain, stress, float, float]: 
        """
        Perform return map integration if yield function is
        evaluated positive. The return map evaluation is point-wise.
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

# Save for later
#class mohrCoulombEvaluate(abstractMaterialEvaluate):
#    """Evaluate of a given material point with mohr coulomb yielding criteria."""
    
#class drukerPragerEvaluate(abstractMaterialEvaluate):
#    """Evaluate of a given material point with mohr coulomb yielding criteria."""