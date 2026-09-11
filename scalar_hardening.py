from dataclasses import dataclass
from math import exp
from typing import Callable

@dataclass
class HardeningParameters:
    # Initial yield stress
    sigma_y: float
    # Saturation yield stress
    sigma_u: float
    # Total hardening modulus
    H_bar  : float
    # Mixing parameter of isotropic/kinematic hardening
    theta  : float
    # Saturation parameter
    delta  : float

ScalarHardeningLaw = Callable[
    [float, HardeningParameters],
    tuple[float, float], # value and derivative
]

def isotropic_hardening_K(
    alpha: float,
    p: HardeningParameters,
) -> tuple[float, float]:
    value = (
        p.sigma_y
        + p.theta * p.H_bar * alpha
        + (p.sigma_u - p.sigma_y) * (1.0 - exp(-p.delta * alpha))
    )
    derivative = (
        p.theta * p.H_bar
        + (p.sigma_u - p.sigma_y)
        * p.delta
        * exp(-p.delta * alpha)
    )
    return value, derivative

def kinematic_hardening_H(
    alpha: float,
    p: HardeningParameters,
) -> tuple[float, float]:
    value = (1.0 - p.theta) * p.H_bar * alpha
    derivative = (1.0 - p.theta) * p.H_bar
    return value, derivative