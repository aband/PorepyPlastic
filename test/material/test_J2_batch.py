"""Batched J2 return map and tangent correction against the per-cell model."""

from dataclasses import fields

import numpy as np
import pytest

from material.J2 import ReturnMapConvergenceError, _materialProperty, vonMisesModel
from material.J2_batch import (
    BatchReturnMapConvergenceError, radial_return_batch, tangent_correction_batch,
)
from material.material_state import MaterialPoint, MaterialState
from material.scalar_hardening import (
    HardeningParameters, isotropic_hardening_K, kinematic_hardening_H,
)
from material.tensor import deviatoric_projector, strain, stress

NC = 12


def _points() -> list[MaterialPoint]:
    """Cells with different moduli and saturating, mixed hardening."""
    points = []
    for cell in range(NC):
        material = _materialProperty()
        material.young_modulus = 200e9 * (1.0 + 0.05 * cell)
        material.poisson_ratio = 0.25 + 0.01 * (cell % 4)
        parameters = HardeningParameters(
            sigma_y=250e6 * (1.0 + 0.03 * cell), sigma_u=400e6, H_bar=1e9,
            theta=0.2 + 0.05 * (cell % 5), delta=20.0 + 10.0 * (cell % 3),
        )
        points.append(MaterialPoint(
            material, parameters, vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H,
        ))
    return points


def _symmetric(rng: np.random.Generator, scale: float | np.ndarray) -> np.ndarray:
    values = rng.normal(size=(3, 3, NC))
    return scale * 0.5 * (values + values.swapaxes(0, 1))


def _committed(points: list[MaterialPoint], rng: np.random.Generator) -> list[MaterialState]:
    """Plastic histories with nonzero backstress from one per-cell preload."""
    preload = _symmetric(rng, 0.004)
    return [
        point.evaluate_from(MaterialState(), strain(preload[:, :, cell]))[0]
        for cell, point in enumerate(points)
    ]


def _batch_inputs(points: list[MaterialPoint], states: list[MaterialState]) -> dict[str, object]:
    stacked = {
        name: np.stack([getattr(s, name).to_numpy() for s in states], axis=2)
        for name in ("stress", "plastic_strain", "backstress")
    }
    return {
        "stress_n": stacked["stress"],
        "plastic_strain_n": stacked["plastic_strain"],
        "backstress_n": stacked["backstress"],
        "alpha_n": np.array([s.alpha for s in states]),
        "mu": np.array([p.material.isotropic_shear_modulus for p in points]),
        "lam": np.array([p.material.lame_parameter for p in points]),
        "parameters": HardeningParameters(**{
            item.name: np.array([getattr(p.parameters, item.name) for p in points])
            for item in fields(HardeningParameters)
        }),
        "K_law": isotropic_hardening_K,
        "H_law": kinematic_hardening_H,
    }


@pytest.mark.parametrize("history", ["virgin", "plastic"])
def test_batch_matches_per_cell_return_map(history: str) -> None:
    rng = np.random.default_rng(7)
    points = _points()
    states = [MaterialState() for _ in points] if history == "virgin" else _committed(points, rng)
    # Alternate small (elastic) and large (plastic) increments.
    increment = _symmetric(rng, np.where(np.arange(NC) % 2 == 0, 1e-6, 3e-3))
    inputs = _batch_inputs(points, states)
    saved = {k: v.copy() for k, v in inputs.items() if isinstance(v, np.ndarray)}

    result = radial_return_batch(strain_increment=increment, **inputs)  # type: ignore[arg-type]

    plastic = 0
    for cell, (point, state) in enumerate(zip(points, states, strict=True)):
        expected, predictor, data = point.evaluate_from_with_data(state, strain(increment[:, :, cell]))
        scale = max(1.0, float(np.max(np.abs(predictor.to_numpy()))))
        for name in ("stress", "plastic_strain", "backstress"):
            reference = getattr(expected, name).to_numpy()
            tolerance = 1e-12 * (scale if name != "plastic_strain" else 1.0)
            np.testing.assert_allclose(getattr(result, name)[:, :, cell], reference, rtol=0, atol=tolerance)
        np.testing.assert_allclose(result.elastic_predictor[:, :, cell], predictor.to_numpy(), rtol=0, atol=1e-12 * scale)
        assert result.alpha[cell] == pytest.approx(expected.alpha, rel=1e-12, abs=1e-18)
        assert result.dgamma[cell] == pytest.approx(data.dgamma, rel=1e-10, abs=1e-18)
        assert (result.dgamma[cell] > 0.0) == (data.dgamma > 0.0)
        if data.dgamma == 0.0:
            # The elastic branch returns the committed history exactly.
            np.testing.assert_array_equal(result.plastic_strain[:, :, cell], state.plastic_strain.to_numpy())
            np.testing.assert_array_equal(result.backstress[:, :, cell], state.backstress.to_numpy())
            assert result.alpha[cell] == state.alpha
            np.testing.assert_array_equal(result.flow_direction[:, :, cell], 0.0)
        else:
            plastic += 1
            np.testing.assert_allclose(result.flow_direction[:, :, cell], data.flow_direction, atol=1e-14)
    assert 0 < plastic < NC
    for name, value in saved.items():
        np.testing.assert_array_equal(inputs[name], value)


def test_tangent_correction_matches_per_cell_consistent_tangent() -> None:
    rng = np.random.default_rng(8)
    points = _points()
    states = _committed(points, rng)
    increment = _symmetric(rng, np.where(np.arange(NC) % 3 == 0, 1e-7, 2e-3))
    inputs = _batch_inputs(points, states)
    result = radial_return_batch(strain_increment=increment, **inputs)  # type: ignore[arg-type]
    cells, c_dev, c_nn = tangent_correction_batch(
        result, mu=inputs["mu"], parameters=inputs["parameters"],  # type: ignore[arg-type]
        K_law=isotropic_hardening_K, H_law=kinematic_hardening_H,
    )
    np.testing.assert_array_equal(cells, np.flatnonzero(result.dgamma > 0.0))
    assert 0 < cells.size < NC

    projector = deviatoric_projector(3)
    for k, cell in enumerate(cells):
        point, state = points[cell], states[cell]
        returned, predictor, data = point.evaluate_from_with_data(state, strain(increment[:, :, cell]))
        algorithmic = point.model.consistent_tangent(
            predictor, state.backstress, returned.alpha, data.dgamma, point.material,
            point.K_law, point.H_law, point.parameters, return_map_data=data,
        ).to_numpy()
        expected = point.material.elastic_tensor.to_numpy() - algorithmic
        n = result.flow_direction[:, :, cell]
        actual = c_dev[k] * projector + c_nn[k] * np.einsum("ij,kl->ijkl", n, n)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9 * np.max(np.abs(expected)))


def test_elastic_batch_has_no_tangent_correction() -> None:
    points = _points()
    inputs = _batch_inputs(points, [MaterialState() for _ in points])
    result = radial_return_batch(
        strain_increment=_symmetric(np.random.default_rng(9), 1e-7), **inputs,  # type: ignore[arg-type]
    )
    np.testing.assert_array_equal(result.dgamma, 0.0)
    cells, c_dev, c_nn = tangent_correction_batch(
        result, mu=inputs["mu"], parameters=inputs["parameters"],  # type: ignore[arg-type]
        K_law=isotropic_hardening_K, H_law=kinematic_hardening_H,
    )
    assert cells.size == c_dev.size == c_nn.size == 0


def test_local_failure_reports_the_same_cells_as_the_per_cell_iteration() -> None:
    rng = np.random.default_rng(10)
    points = _points()
    states = [MaterialState() for _ in points]
    increment = _symmetric(rng, np.where(np.arange(NC) % 2 == 0, 1e-6, 2e-2))
    inputs = _batch_inputs(points, states)
    with pytest.raises(BatchReturnMapConvergenceError) as caught:
        radial_return_batch(strain_increment=increment, max_iteration=1, **inputs)  # type: ignore[arg-type]
    assert isinstance(caught.value, ReturnMapConvergenceError)

    failing = []
    for cell, point in enumerate(points):
        predictor = point.material.elastic_tensor.double_contract(strain(increment[:, :, cell]))
        shifted = stress(predictor.to_numpy()).deviatoric().to_numpy()
        try:
            point.model.consistency_parameter(
                float(np.linalg.norm(shifted)), 0.0, point.material.isotropic_shear_modulus,
                point.parameters, point.K_law, point.H_law, max_iteration=1,
            )
        except ReturnMapConvergenceError:
            failing.append(cell)
    assert failing
    np.testing.assert_array_equal(caught.value.cells, failing)


def test_invalid_tolerances_are_rejected() -> None:
    points = _points()
    inputs = _batch_inputs(points, [MaterialState() for _ in points])
    with pytest.raises(ValueError, match="tolerances"):
        radial_return_batch(strain_increment=np.zeros((3, 3, NC)), tol=-1.0, **inputs)  # type: ignore[arg-type]
