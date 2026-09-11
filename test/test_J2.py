from math import sqrt

import pytest

from scalar_hardening import (
    HardeningParameters,
    isotropic_hardening_K,
    kinematic_hardening_H,
)

from J2 import vonMisesModel

@pytest.fixture
def model() -> vonMisesModel:
    return vonMisesModel("vonMises")

@pytest.fixture
def parameters() -> HardeningParameters:
    return HardeningParameters(        
        sigma_y=250.0,
        sigma_u=250.0,
        H_bar=1000.0,
        theta=0.4,
        delta=0.0,
    )

def test_elastic_step_returns_zero(model: vonMisesModel, parameters: HardeningParameters):
    dgamma = model.consistency_parameter(
        xi_trial_norm=200.0,
        alpha_n=0.0,
        mu=80.0,
        parameters=parameters,
        K_law=isotropic_hardening_K,
        H_law=kinematic_hardening_H,
    )

    assert dgamma == 0.0

def test_linear_combined_hardening(model: vonMisesModel, parameters: HardeningParameters):
    mu = 80.0
    alpha_n = 0.02
    xi_trial_norm = 300.0
    c = sqrt(2.0 / 3.0)

    K_n, _ = isotropic_hardening_K(alpha_n, parameters)

    expected = (
        xi_trial_norm - c * K_n
    ) / (
        2.0 * mu
        + (2.0 / 3.0) * parameters.H_bar
    )

    dgamma = model.consistency_parameter(
        xi_trial_norm=xi_trial_norm,
        alpha_n=alpha_n,
        mu=mu,
        parameters=parameters,
        K_law=isotropic_hardening_K,
        H_law=kinematic_hardening_H,
    )

    assert dgamma == pytest.approx(expected, rel=1.0e-10) # type: ignore

def test_radial_return_map_plastic_step(model, parameters):
    import numpy as np

    from J2 import _materialProperty
    from tensor import strain, stress

    material = _materialProperty()
    material.young_modulus = 210_000.0
    material.poisson_ratio = 0.3

    sigma_trial = stress(np.diag([300.0, 0.0, 0.0]))
    plastic_strain_n = strain.zeros((3, 3))
    backstress_n = stress.zeros((3, 3))
    alpha_n = 0.0

    sigma, plastic_strain, backstress, alpha, dgamma = (
        model.radial_return_map(
            sigma_trial=sigma_trial,
            plastic_strain_n=plastic_strain_n,
            backstress_n=backstress_n,
            alpha_n=alpha_n,
            material=material,
            K_law=isotropic_hardening_K,
            H_law=kinematic_hardening_H,
            parameters=parameters,
        )
    )

    c = np.sqrt(2.0 / 3.0)
    mu = material.isotropic_shear_modulus
    xi_trial = sigma_trial.deviatoric().to_numpy()
    xi_trial_norm = np.linalg.norm(xi_trial)
    direction = xi_trial / xi_trial_norm

    expected_dgamma = (
        xi_trial_norm - c * parameters.sigma_y
    ) / (
        2.0 * mu + (2.0 / 3.0) * parameters.H_bar
    )

    assert dgamma == pytest.approx(expected_dgamma)
    assert alpha == pytest.approx(c * expected_dgamma)

    np.testing.assert_allclose(
        sigma.to_numpy(),
        sigma_trial.to_numpy()
        - 2.0 * mu * expected_dgamma * direction,
    )
    np.testing.assert_allclose(
        plastic_strain.to_numpy(),
        expected_dgamma * direction,
    )
    np.testing.assert_allclose(
        backstress.to_numpy(),
        (2.0 / 3.0)
        * (1.0 - parameters.theta)
        * parameters.H_bar
        * expected_dgamma
        * direction,
    )

    yield_stress, _ = isotropic_hardening_K(alpha, parameters)
    shifted_stress = sigma.deviatoric() - backstress.deviatoric()
    residual = (
        np.sqrt(shifted_stress.inner(shifted_stress))
        - c * yield_stress
    )

    assert residual == pytest.approx(0.0, abs=1.0e-10)