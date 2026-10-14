"""Elastic TPSA checks using the names and block ordering from test_tpsa.py."""

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from porepy.numerics.fv.tpsa import Tpsa

from alternatives.plane_strain import BoundaryDisplacement, KEYWORD, PlaneStrainTpsa
from alternatives.postprocessing import TpsaPostprocessing


@pytest.mark.parametrize("cells_per_axis", [2, 4])
@pytest.mark.parametrize(
    "gradient,translation",
    [
        (np.diag([1.0e-4, 0.0]), np.zeros(2)),
        (np.array([[0.0, 2.0e-4], [0.0, 0.0]]), np.zeros(2)),
        (np.array([[0.0, -1.0e-4], [1.0e-4, 0.0]]), np.zeros(2)),
        (np.zeros((2, 2)), np.array([1.0e-4, -2.0e-4])),
    ],
    ids=["extension", "simple_shear", "rigid_rotation", "translation"],
)
def test_2d_linear_displacement(
    cells_per_axis: int,
    gradient: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> None:
    """2D counterpart to PorePy's test_3d_linear_displacement."""
    case = PlaneStrainTpsa(
        cells_per_axis=cells_per_axis,
        displacement_gradient=gradient,
        translation=translation,
    )
    x = case.solve()
    g = case.grid

    assert g.dim == 2
    assert g.num_cells == cells_per_axis**2
    assert x.size == Tpsa(KEYWORD).ndof(g) == 4 * g.num_cells
    u = x[:g.dim * g.num_cells].reshape((g.dim, -1), order="F")
    r = x[g.dim * g.num_cells:(g.dim + 1) * g.num_cells]
    p = x[(g.dim + 1) * g.num_cells:]
    mu = case.material.isotropic_shear_modulus
    lmbda = case.material.lame_parameter
    u_ex = gradient @ g.cell_centers[:2] + translation[:, None]
    r_ex = np.full(g.num_cells, mu * (gradient[0, 1] - gradient[1, 0]))
    p_ex = np.full(g.num_cells, lmbda * np.trace(gradient))

    np.testing.assert_allclose(u, u_ex, rtol=1.0e-10, atol=1.0e-14)
    # Absolute stress tolerance accounts for roundoff at SI elastic moduli.
    np.testing.assert_allclose(r, r_ex, rtol=1.0e-10, atol=1.0e-5)
    np.testing.assert_allclose(p, p_ex, rtol=1.0e-10, atol=1.0e-5)
    assert case.reference_rotation_stress() == pytest.approx(r_ex[0])

    generalized_flux = (
        case.face_discretization @ x + case.rhs_matrix @ case.bound_vec
    )
    stress = generalized_flux[:g.dim * g.num_faces]
    sigma_ex = mu * (gradient + gradient.T) + lmbda * np.trace(gradient) * np.eye(2)
    stress_ex = sigma_ex @ g.face_normals[:2]
    np.testing.assert_allclose(
        stress.reshape((g.dim, -1), order="F"), stress_ex,
        rtol=1.0e-10, atol=1.0e-5,
    )
    # Each cell balances shared face forces, including boundary reactions.
    np.testing.assert_allclose(g.divergence(dim=2) @ stress, 0.0, atol=1.0e-5)

    # Check the exact fields in all three block equations, as in test_tpsa.py.
    sol_ex = np.hstack((u_ex.ravel(order="F"), r_ex, p_ex))
    generalized_flux_ex = (
        case.face_discretization @ sol_ex + case.rhs_matrix @ case.bound_vec
    )
    resid_ex = case.div @ generalized_flux_ex - case.accum @ sol_ex
    np.testing.assert_allclose(resid_ex[:2 * g.num_cells], 0.0, atol=1.0e-5)
    np.testing.assert_allclose(resid_ex[2 * g.num_cells:], 0.0, atol=1.0e-14)


def test_plane_strain_material_reference_and_independent_histories() -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.prepare_simulation()
    assert len(case.material_points) == case.grid.num_cells
    epsilon = case.reference_strain()
    np.testing.assert_array_equal(epsilon.to_numpy()[2], 0.0)
    sigma = case.reference_stress()
    assert sigma[2, 2] == pytest.approx(case.material.lame_parameter * 1.0e-4)
    assert sigma[2, 2] > 0.0

    # Independent constitutive check using the known affine strain, not a
    # numerical reconstruction or a material update from the global solver.
    first, second = case.material_points[:2]
    trial, _ = first.update(epsilon)
    np.testing.assert_allclose(trial.stress.to_numpy(), sigma)
    assert trial.alpha == 0.0
    first.commit()
    np.testing.assert_array_equal(second.committed.stress.to_numpy(), np.zeros((3, 3)))
    assert second.trial is None
    for name in ("stress", "plastic_strain", "backstress"):
        assert not np.shares_memory(
            getattr(first.committed, name).to_numpy(copy=False),
            getattr(second.committed, name).to_numpy(copy=False),
        )


@pytest.mark.parametrize("gradient,translation", [
    (np.diag([1e-4, 0.0]), np.zeros(2)),
    (np.array([[1e-4, 2e-4], [-3e-4, 4e-4]]), np.array([5e-5, -2e-5])),
    (np.array([[0.0, -1e-4], [1e-4, 0.0]]), np.zeros(2)),
    (np.zeros((2, 2)), np.array([1e-4, -2e-4])),
], ids=["extension", "mixed", "rigid_rotation", "translation"])
def test_boundary_callback_reproduces_affine_solution(
    gradient: NDArray[np.float64], translation: NDArray[np.float64],
) -> None:
    affine = PlaneStrainTpsa(cells_per_axis=3, displacement_gradient=gradient,
                            translation=translation)
    custom = PlaneStrainTpsa(
        cells_per_axis=3,
        boundary_displacement=lambda centers: gradient @ centers[:2] + translation[:, None],
    )
    expected, actual = affine.solve(), custom.solve()
    np.testing.assert_array_equal(custom.bc_values, affine.bc_values)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(actual[:2 * custom.grid.num_cells],
                               expected[:2 * affine.grid.num_cells], atol=1e-14)
    post = TpsaPostprocessing(custom)
    expected_strain = np.repeat(affine.reference_strain().to_numpy()[:, :, None],
                                custom.grid.num_cells, axis=2)
    np.testing.assert_allclose(post.strain_green_gauss(), expected_strain, atol=1e-14)
    np.testing.assert_allclose(post.u_cell, affine.reference_displacement(custom.grid.cell_centers),
                               atol=1e-14)


def test_nonuniform_boundary_callback_uses_face_centers_and_porepy_ordering() -> None:
    seen: list[NDArray[np.float64]] = []

    def displacement(centers: NDArray[np.float64]) -> NDArray[np.float64]:
        seen.append(centers.copy())
        x, y = centers[:2]
        return 1e-4 * np.vstack((1 + x**2 + 2*y, -2 + y**2 - 3*x))

    case = PlaneStrainTpsa(cells_per_axis=3, boundary_displacement=displacement)
    case.prepare_simulation()
    g = case.grid
    bf = g.get_all_boundary_faces()
    assert len(seen) == 1
    assert seen[0].shape == (3, bf.size)
    np.testing.assert_array_equal(seen[0], g.face_centers[:, bf])
    x, y = g.face_centers[:2, bf]
    expected = np.zeros((2, g.num_faces))
    expected[:, bf] = 1e-4 * np.vstack((1 + x**2 + 2*y, -2 + y**2 - 3*x))
    # Grid faces have length 1/3: displacement must not be multiplied by that area.
    np.testing.assert_array_equal(case.bc_values, expected)
    np.testing.assert_array_equal(case.bound_vec[0::2], expected[0])
    np.testing.assert_array_equal(case.bound_vec[1::2], expected[1])
    assert np.all(case.bc.is_dir[:, bf])
    assert np.shares_memory(case.bc_values, case.bound_vec)
    case.bc_values[1, bf[0]] = 0.125
    assert case.bound_vec[2 * bf[0] + 1] == 0.125


def test_boundary_callback_owns_data_and_refreshes_on_preparation() -> None:
    inputs: list[NDArray[np.float64]] = []
    outputs: list[NDArray[np.float64]] = []
    scale = 1e-4

    def displacement(centers: NDArray[np.float64]) -> NDArray[np.float64]:
        inputs.append(centers.copy())
        centers *= scale  # A callback may edit its private coordinate array.
        values = centers[:2]  # Return a view, rather than fresh storage.
        outputs.append(values)
        return values

    case = PlaneStrainTpsa(cells_per_axis=2, boundary_displacement=displacement)
    case.prepare_simulation()
    bf = case.grid.get_all_boundary_faces()
    np.testing.assert_array_equal(case.grid.face_centers[:, bf], inputs[0])
    saved = case.bc_values.copy()
    np.testing.assert_array_equal(saved[:, bf], scale * inputs[0][:2])
    assert not np.shares_memory(case.bc_values, outputs[0])
    outputs[0][:] = -123
    np.testing.assert_array_equal(case.bc_values, saved)
    previous = case.bound_vec
    scale *= 2
    case.prepare_simulation()
    assert len(inputs) == 2
    np.testing.assert_array_equal(case.bc_values, 2 * saved)
    np.testing.assert_array_equal(previous.reshape((2, -1), order="F"), saved)
    assert not np.shares_memory(previous, case.bound_vec)


@pytest.mark.parametrize("invalid", [
    np.zeros(()), np.zeros(16), np.zeros((3, 8)), np.zeros((8, 2)), np.zeros((2, 12)),
    np.full((2, 8), np.nan), np.full((2, 8), np.inf), np.full((2, 8), 1j),
], ids=["scalar", "flat", "components", "transposed", "all_faces", "nan", "inf", "complex"])
def test_boundary_callback_rejects_invalid_values(invalid: NDArray[Any]) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2, boundary_displacement=lambda centers: invalid)
    with pytest.raises(ValueError, match="[Bb]oundary displacement"):
        case.prepare_simulation()
    assert case.x is None


def test_boundary_displacement_must_be_callable() -> None:
    with pytest.raises(ValueError, match="callable"):
        PlaneStrainTpsa(boundary_displacement=cast(BoundaryDisplacement, np.zeros((2, 8))))


@pytest.mark.parametrize("argument", ["gradient", "translation", "both"])
def test_boundary_callback_rejects_ambiguous_affine_arguments(argument: str) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        PlaneStrainTpsa(
            boundary_displacement=lambda centers: np.zeros((2, centers.shape[1])),
            displacement_gradient=np.zeros((2, 2)) if argument != "translation" else None,
            translation=np.zeros(2) if argument != "gradient" else None,
        )


@pytest.mark.parametrize("method", [
    "reference_displacement", "reference_strain", "reference_stress",
    "reference_total_pressure", "reference_rotation_stress",
])
def test_custom_boundary_data_has_no_affine_reference(method: str) -> None:
    # Even a callback that happens to be affine does not declare an exact solution.
    case = PlaneStrainTpsa(boundary_displacement=lambda centers: 1e-4 * centers[:2])
    args = (np.zeros((3, 2)),) if method == "reference_displacement" else ()
    with pytest.raises(ValueError, match="Affine reference solutions are unavailable"):
        getattr(case, method)(*args)
