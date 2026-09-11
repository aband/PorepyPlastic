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

    assert dgamma == pytest.approx(expected, rel=1.0e-10)