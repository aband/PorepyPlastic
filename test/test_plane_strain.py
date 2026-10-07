"""Elastic TPSA checks using the names and block ordering from test_tpsa.py."""

import numpy as np
import pytest
from numpy.typing import NDArray
from porepy.numerics.fv.tpsa import Tpsa

from coupling.plane_strain import KEYWORD, PlaneStrainTpsa


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
