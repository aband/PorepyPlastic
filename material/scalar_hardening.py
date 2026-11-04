from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray

# Laws also accept array alpha (and array parameter fields) elementwise,
# which the batched return map in material.J2_batch relies on.
Scalar: TypeAlias = float | NDArray[np.float64]

@dataclass
class HardeningParameters:
    # Initial yield stress
    sigma_y: Scalar
    # Saturation yield stress
    sigma_u: Scalar
    # Total hardening modulus
    H_bar  : Scalar
    # Mixing parameter of isotropic/kinematic hardening
    theta  : Scalar
    # Saturation parameter
    delta  : Scalar

ScalarHardeningLaw = Callable[
    [Scalar, HardeningParameters],
    tuple[Scalar, Scalar], # value and derivative
]

def isotropic_hardening_K(
    alpha: Scalar,
    p: HardeningParameters,
) -> tuple[Scalar, Scalar]:
    decay = np.exp(-p.delta * alpha)
    value = (
        p.sigma_y
        + p.theta * p.H_bar * alpha
        + (p.sigma_u - p.sigma_y) * (1.0 - decay)
    )
    derivative = (
        p.theta * p.H_bar
        + (p.sigma_u - p.sigma_y)
        * p.delta
        * decay
    )
    return value, derivative

def kinematic_hardening_H(
    alpha: Scalar,
    p: HardeningParameters,
) -> tuple[Scalar, Scalar]:
    value = (1.0 - p.theta) * p.H_bar * alpha
    derivative = (1.0 - p.theta) * p.H_bar
    return value, derivative
