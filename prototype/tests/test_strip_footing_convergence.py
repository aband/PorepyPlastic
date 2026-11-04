"""Nested meshes and independent convergence-measure checks."""

import numpy as np
import pytest

from prototype.strip_footing import make_case
from prototype.strip_footing_convergence import curve_difference, field_differences


def test_refinement_halves_all_intervals_and_preserves_boundary_conditions():
    coarse = make_case(1, growth=2.0)
    fine = make_case(1, growth=2.0, refinement_level=1)
    assert fine.grid.num_cells == 4*coarse.grid.num_cells
    assert fine.footing_faces.size == 2*coarse.footing_faces.size
    for name in ("x_nodes", "y_nodes"):
        a, b = getattr(coarse, name), getattr(fine, name)
        np.testing.assert_allclose(a, b[::2], rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.diff(b), np.repeat(np.diff(a)/2, 2))
    np.testing.assert_allclose(fine.grid.cell_volumes.sum(), coarse.grid.cell_volumes.sum())
    np.testing.assert_allclose(fine.grid.face_areas[fine.footing_faces].sum(), .5)
    np.testing.assert_allclose(fine.bc_values[1, fine.footing_faces], -.002)
    assert np.all(fine.bc.is_neu[0, fine.footing_faces])


@pytest.mark.parametrize("level", [-1, True, 1.5])
def test_invalid_refinement(level):
    with pytest.raises(ValueError, match="refinement_level"):
        make_case(refinement_level=level)


def test_curve_norm_integrates_known_linear_curves_on_different_knots():
    coarse, fine = np.zeros((3, 14)), np.zeros((5, 14))
    coarse[:, 3], fine[:, 3] = [0, .4, 1], [0, .1, .3, .8, 1]
    coarse[:, 5], fine[:, 5] = 2*coarse[:, 3], fine[:, 3]
    errors = curve_difference(coarse, fine)
    assert errors["relative_curve_L2"] == pytest.approx(1)
    assert errors["final_pressure_over_c_difference"] == pytest.approx(1)


def test_field_norm_uses_cell_volumes_and_parent_cell_order():
    coarse = {"x_nodes": np.array([0., 1., 3.]), "y_nodes": np.array([0., 2.])}
    fine = {"x_nodes": np.array([0., .5, 1., 2., 3.]), "y_nodes": np.array([0., 1., 2.]),
            "cell_volumes": np.array([.5, .5, 1., 1., .5, .5, 1., 1.])}
    for name in ("displacement", "strain", "stress", "alpha"):
        coarse[name] = np.array([2., 5.])
        fine[name] = np.array([2., 2., 5., 5., 2., 2., 5., 5.])
    assert all(v == 0 for v in field_differences(coarse, fine).values())
    fine["stress"] = fine["stress"] + 1
    differences = field_differences(coarse, fine)
    assert differences["stress_L2"] == pytest.approx(np.sqrt(6))
    assert differences["relative_stress_L2"] == pytest.approx(np.sqrt(6/(2*3**2+4*6**2)))
