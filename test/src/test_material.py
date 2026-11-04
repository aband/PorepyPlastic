"""CellMaterial construction rules for batched core material updates."""

from math import exp

import numpy as np
import pytest

from core.material import CellMaterial
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import (
    HardeningParameters, isotropic_hardening_K, kinematic_hardening_H,
)


def _points(count: int = 3) -> list[MaterialPoint]:
    points = []
    for cell in range(count):
        material = _materialProperty()
        material.young_modulus, material.poisson_ratio = 200e9 + 1e9 * cell, 0.3
        parameters = HardeningParameters(250e6 + cell, 300e6, 1e9, 0.4, 10.0)
        points.append(MaterialPoint(
            material, parameters, vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H,
        ))
    return points


def test_reads_cellwise_moduli_and_parameters_once() -> None:
    points = _points()
    material = CellMaterial(points)
    np.testing.assert_array_equal(material.mu, [p.material.isotropic_shear_modulus for p in points])
    np.testing.assert_array_equal(material.lam, [p.material.lame_parameter for p in points])
    np.testing.assert_array_equal(material.parameters.sigma_y, [250e6, 250e6 + 1, 250e6 + 2])
    points[0].parameters.sigma_y = 1.0
    assert material.parameters.sigma_y[0] == 250e6


def test_rejects_non_j2_models() -> None:
    class OtherModel:
        pass

    points = _points()
    points[1].model = OtherModel()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="J2 model only"):
        CellMaterial(points)


def test_rejects_mixed_hardening_laws() -> None:
    def other_K(alpha: float, p: HardeningParameters) -> tuple[float, float]:
        return isotropic_hardening_K(alpha, p)

    points = _points()
    points[2].K_law = other_K
    with pytest.raises(ValueError, match="same hardening laws"):
        CellMaterial(points)


def test_rejects_scalar_only_hardening_laws() -> None:
    def scalar_K(alpha: float, p: HardeningParameters) -> tuple[float, float]:
        return p.sigma_y + p.theta * p.H_bar * alpha * exp(-p.delta * alpha), p.theta * p.H_bar

    points = _points()
    for point in points:
        point.K_law = scalar_K
    with pytest.raises(ValueError, match="K_law must accept array alpha"):
        CellMaterial(points)


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="one material point per cell"):
        CellMaterial([])
