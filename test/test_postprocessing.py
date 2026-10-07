"""Tests for TPSA inputs, face recovery, and Green–Gauss plane strain."""

from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray
from porepy.grids.grid import Grid
from porepy.grids.structured import CartGrid

from porepy.params.tensor import FourthOrderTensor
from porepy.utils.common_constants import PARAMETERS

from coupling.plane_strain import (
    KEYWORD, PlaneStrainTpsa, _assemble_matrices, _discretize_get_matrices, _solve,
)
from coupling.postprocessing import TpsaPostprocessing, _validate_reconstruction_inputs

ReconstructionInputs: TypeAlias = tuple[
    Grid, NDArray[np.float64], NDArray[np.float64]
]


@pytest.fixture
def inputs() -> ReconstructionInputs:
    grid = CartGrid(np.array([2, 3]), physdims=np.array([1.0, 2.0]))
    grid.compute_geometry()
    return grid, np.zeros((2, grid.num_cells)), np.zeros((2, grid.num_faces))


@pytest.mark.parametrize("interior_value", [0.0, np.nan, np.inf])
def test_validate_tpsa_solution_preserves_inputs(interior_value: float) -> None:
    case = PlaneStrainTpsa(
        cells_per_axis=2,
        displacement_gradient=np.array([[1e-4, 2e-4], [-3e-4, 4e-4]]),
    )
    x = case.solve()
    grid = case.grid
    u_cell = x[:2 * grid.num_cells].reshape((2, grid.num_cells), order="F")
    u_boundary = case.bound_vec.reshape((2, grid.num_faces), order="F").copy()
    interior = np.setdiff1d(
        np.arange(grid.num_faces), grid.get_all_boundary_faces()
    )
    assert interior.size > 0
    u_boundary[:, interior] = interior_value
    arrays = (
        x, case.bound_vec, u_boundary, grid.nodes, grid.cell_volumes,
        grid.cell_centers, grid.face_areas, grid.face_centers, grid.face_normals,
    )
    originals = [array.copy() for array in arrays]

    _validate_reconstruction_inputs(grid, u_cell, u_boundary)

    for array, original in zip(arrays, originals):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("dim", [1, 3])
def test_rejects_non_2d_grid(dim: int) -> None:
    grid = CartGrid(np.full(dim, 2, dtype=int))
    grid.compute_geometry()
    with pytest.raises(ValueError, match="2D"):
        _validate_reconstruction_inputs(
            grid, np.zeros((2, grid.num_cells)), np.zeros((2, grid.num_faces))
        )


def test_requires_computed_geometry() -> None:
    grid = CartGrid(np.array([2, 3]))
    with pytest.raises(ValueError, match="compute_geometry"):
        _validate_reconstruction_inputs(
            grid, np.zeros((2, grid.num_cells)), np.zeros((2, grid.num_faces))
        )


@pytest.mark.parametrize("field", [0, 1], ids=["cell", "boundary"])
@pytest.mark.parametrize("layout", ["flat", "transposed", "short"])
def test_rejects_displacement_shape(
    inputs: ReconstructionInputs, field: int, layout: str,
) -> None:
    grid, *values = inputs
    array = values[field]
    if layout == "flat":
        values[field] = array.ravel(order="F")
    elif layout == "transposed":
        values[field] = array.T
    else:
        values[field] = array[:, :-1]
    with pytest.raises(ValueError, match="Expected u_cell"):
        _validate_reconstruction_inputs(grid, values[0], values[1])


@pytest.mark.parametrize("field", [0, 1], ids=["cell", "boundary"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_used_displacement(
    inputs: ReconstructionInputs, field: int, value: float,
) -> None:
    grid, *values = inputs
    index = 0 if field == 0 else grid.get_all_boundary_faces()[0]
    values[field][1, index] = value
    with pytest.raises(ValueError, match="finite"):
        _validate_reconstruction_inputs(grid, values[0], values[1])


@pytest.mark.parametrize("name", ["cell_volumes", "face_areas"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_rejects_nonpositive_measures(
    inputs: ReconstructionInputs, name: str, value: float,
) -> None:
    grid, u_cell, u_boundary = inputs
    getattr(grid, name)[0] = value
    with pytest.raises(ValueError, match="positive"):
        _validate_reconstruction_inputs(grid, u_cell, u_boundary)


@pytest.mark.parametrize(
    "name", ["cell_volumes", "cell_centers", "face_areas", "face_centers", "face_normals"]
)
@pytest.mark.parametrize("defect", ["missing", "wrong_shape", "nonfinite"])
def test_rejects_invalid_geometry(
    inputs: ReconstructionInputs, name: str, defect: str,
) -> None:
    grid, u_cell, u_boundary = inputs
    if defect == "missing":
        delattr(grid, name)
    elif defect == "wrong_shape":
        setattr(grid, name, getattr(grid, name)[:-1])
    else:
        getattr(grid, name).flat[0] = np.nan
    with pytest.raises(ValueError, match=name):
        _validate_reconstruction_inputs(grid, u_cell, u_boundary)


def test_postprocessing_exposes_solved_fields_and_operators() -> None:
    case = PlaneStrainTpsa(
        cells_per_axis=2,
        displacement_gradient=np.array([[1e-4, 2e-4], [-3e-4, 4e-4]]),
        translation=np.array([1e-5, -2e-5]),
    )
    x = case.solve()
    post = TpsaPostprocessing(case)
    grid = post.grid

    assert post.x is case.x is x
    assert post.matrices is case.matrices
    assert post.d is case.d
    assert post.bc is case.d[PARAMETERS][KEYWORD]["bc"]
    assert np.all(post.bc.is_dir[:, post.boundary_faces])
    assert np.shares_memory(post.u_cell, x)
    assert np.shares_memory(post.bc_values, post.bound_vec)
    assert post.u_boundary is post.bc_values
    np.testing.assert_allclose(
        post.u_cell, case.reference_displacement(grid.cell_centers), atol=1e-14,
    )
    np.testing.assert_allclose(post.r, case.reference_rotation_stress(), atol=1e-5)
    np.testing.assert_allclose(post.p, case.reference_total_pressure(), atol=1e-5)
    np.testing.assert_allclose(
        post.u_boundary[:, post.boundary_faces],
        case.reference_displacement(grid.face_centers[:, post.boundary_faces]),
    )
    np.testing.assert_array_equal(
        post.bc_values.ravel(order="F"), case.bound_vec,
    )

    # The stored operators, boundary data, and solution describe the same solve.
    flux = post.face_discretization @ post.x + post.rhs_matrix @ post.bound_vec
    np.testing.assert_allclose(
        post.div @ flux - post.accum @ post.x, 0.0, atol=1e-5,
    )
    stress = (
        post.matrices["stress"] @ post.u_cell.ravel(order="F")
        + post.matrices["stress_rotation"] @ post.r
        + post.matrices["stress_total_pressure"] @ post.p
        + post.matrices["bound_stress"] @ post.bound_vec
    )
    np.testing.assert_allclose(
        stress.reshape((2, grid.num_faces), order="F"),
        case.reference_stress()[:2, :2] @ grid.face_normals[:2],
        rtol=1e-10, atol=1e-5,
    )


def test_postprocessing_requires_a_solved_case() -> None:
    with pytest.raises(RuntimeError, match="case.solve"):
        TpsaPostprocessing(PlaneStrainTpsa())


@pytest.mark.parametrize("prepare", ["set_geometry", "prepare_simulation"])
def test_postprocessing_retains_previous_solve_after_repreparation(
    prepare: str,
) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    x = case.solve()
    saved_x = x.copy()
    previous = TpsaPostprocessing(case)

    case.cells_per_axis = 3
    getattr(case, prepare)()
    assert case.x is None
    with pytest.raises(RuntimeError, match="case.solve"):
        TpsaPostprocessing(case)

    case.solve()
    current = TpsaPostprocessing(case)
    assert current.grid.num_cells == 9
    assert current.u_cell.shape == (2, 9)
    assert previous.grid.num_cells == 4
    assert previous.matrices is not current.matrices
    assert previous.bc is not current.bc
    np.testing.assert_array_equal(previous.x, saved_x)
    np.testing.assert_allclose(
        current.u_cell, case.reference_displacement(current.grid.cell_centers),
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "solution",
    [
        np.zeros(15),
        np.zeros((4, 4)),
        np.r_[np.zeros(8), np.nan, np.zeros(7)],
        np.r_[np.zeros(15), np.inf],
    ],
    ids=["wrong_size", "not_flat", "nonfinite_rotation", "nonfinite_pressure"],
)
def test_postprocessing_rejects_invalid_solution(
    solution: NDArray[np.float64],
) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.solve()
    case.x = solution
    with pytest.raises(ValueError, match="finite TPSA solution"):
        TpsaPostprocessing(case)


def test_postprocessing_rejects_non_dirichlet_boundary_data() -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.solve()
    face = case.grid.get_all_boundary_faces()[0]
    case.bc.is_dir[:, face] = False
    case.bc.is_neu[:, face] = True
    with pytest.raises(ValueError, match="Dirichlet"):
        TpsaPostprocessing(case)


@pytest.mark.parametrize("cells_per_axis", [1, 3])
@pytest.mark.parametrize(
    "gradient, translation",
    [
        (np.zeros((2, 2)), np.zeros(2)),
        (np.zeros((2, 2)), np.array([2e-4, -3e-4])),
        (np.diag([1e-4, 0.0]), np.zeros(2)),
        (np.array([[0.0, 2e-4], [0.0, 0.0]]), np.zeros(2)),
        (np.array([[0.0, -1e-4], [1e-4, 0.0]]), np.zeros(2)),
        (np.array([[1e-4, 2e-4], [-3e-4, 4e-4]]), np.array([1e-5, -2e-5])),
    ],
    ids=["zero", "translation", "extension", "shear", "rotation", "combined"],
)
def test_green_gauss_recovers_affine_fields(
    cells_per_axis: int,
    gradient: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> None:
    case = PlaneStrainTpsa(
        cells_per_axis=cells_per_axis, displacement_gradient=gradient,
        translation=translation,
    )
    case.solve()
    post = TpsaPostprocessing(case)
    assert post.u_face is post.grad_u is post.epsilon is None
    epsilon = post.strain_green_gauss()
    assert post.u_face is not None and post.grad_u is not None
    assert epsilon is post.epsilon
    nc = post.grid.num_cells
    np.testing.assert_allclose(
        post.u_face, case.reference_displacement(post.grid.face_centers), atol=1e-14,
    )
    np.testing.assert_allclose(
        post.grad_u, np.repeat(gradient[:, :, None], nc, axis=2), atol=1e-14,
    )
    expected = np.zeros((3, 3, nc))
    expected[:2, :2] = (0.5 * (gradient + gradient.T))[:, :, None]
    np.testing.assert_allclose(epsilon, expected, atol=1e-14)
    np.testing.assert_array_equal(epsilon[2], 0.0)
    np.testing.assert_array_equal(epsilon[:, 2], 0.0)
    assert all(point.trial is None for point in case.material_points)


def _rediscretize(case: PlaneStrainTpsa) -> None:
    case.matrices = _discretize_get_matrices(case.grid, case.d)
    (
        case.face_discretization, case.rhs_matrix, case.div, case.accum,
    ) = _assemble_matrices(case.matrices, case.grid, case.d)


@pytest.fixture
def heterogeneous_case() -> PlaneStrainTpsa:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.prepare_simulation()
    # Unequal cell widths exercise the distance as well as the material weights.
    for axis, midpoint in enumerate([0.15, 0.3]):
        case.grid.nodes[axis, case.grid.nodes[axis] == 0.5] = midpoint
    case.grid.compute_geometry()
    case.d[PARAMETERS][KEYWORD]["fourth_order_tensor"] = FourthOrderTensor(
        mu=np.array([1.0, 3.0, 2.0, 5.0]), lmbda=np.full(4, 2.0),
    )
    _rediscretize(case)
    # Arbitrary fields test the face formula without relying on equilibrium.
    case.x = np.array([
        0.2, -0.3, 0.7, 0.4, -0.1, 0.6, 0.9, -0.2,
        0.1, -0.2, 0.3, -0.4, 1.0, -2.0, 3.0, 0.5,
    ])
    bf = case.grid.get_all_boundary_faces()
    case.bc_values[:, bf] = 0.1 + case.grid.face_centers[:2, bf] ** 2
    return case


def test_face_recovery_includes_weighting_and_pressure_jump(
    heterogeneous_case: PlaneStrainTpsa,
) -> None:
    post = TpsaPostprocessing(heterogeneous_case)
    grid = post.grid
    mu = heterogeneous_case.d[PARAMETERS][KEYWORD]["fourth_order_tensor"].mu
    expected = post.u_boundary.copy()
    incidence = grid.cell_faces.tocsr()
    largest_correction = 0.0
    for face in np.setdiff1d(np.arange(grid.num_faces), post.boundary_faces):
        start, end = incidence.indptr[face:face + 2]
        cells = incidence.indices[start:end]
        signs = incidence.data[start:end]
        n = grid.face_normals[:2, face] / grid.face_areas[face]
        distances = np.abs(n @ (
            grid.face_centers[:2, face, None] - grid.cell_centers[:2, cells]
        ))
        weights = mu[cells] / distances
        correction = (signs @ post.p[cells]) * n / (2.0 * weights.sum())
        expected[:, face] = post.u_cell[:, cells] @ (weights / weights.sum()) - correction
        largest_correction = max(largest_correction, float(np.linalg.norm(correction)))
    assert largest_correction > 1e-2
    np.testing.assert_allclose(post.reconstruct_face_displacement(), expected, atol=1e-14)


def test_reconstruction_is_invariant_to_face_orientation(
    heterogeneous_case: PlaneStrainTpsa,
) -> None:
    case = heterogeneous_case
    original = TpsaPostprocessing(case)
    epsilon = original.strain_green_gauss()
    grid = case.grid
    orientation = np.ones(grid.num_faces)
    orientation[::2] = -1.0
    grid.face_normals *= orientation
    grid.cell_faces = grid.cell_faces.multiply(orientation[:, None]).tocsc()
    _rediscretize(case)
    flipped = TpsaPostprocessing(case)
    np.testing.assert_allclose(flipped.strain_green_gauss(), epsilon, atol=1e-14)
    assert original.u_face is not None and flipped.u_face is not None
    assert original.grad_u is not None and flipped.grad_u is not None
    np.testing.assert_allclose(flipped.u_face, original.u_face, atol=1e-14)
    np.testing.assert_allclose(flipped.grad_u, original.grad_u, atol=1e-14)


def test_green_gauss_matches_tpsa_mass_and_rotation_equations() -> None:
    case = PlaneStrainTpsa(cells_per_axis=3)
    case.prepare_simulation()
    bf = case.grid.get_all_boundary_faces()
    x, y = case.grid.face_centers[:2, bf]
    case.bc_values[:, bf] = 1e-4 * np.array([x**2 + x*y, y**2 - x*y])
    case.x = _solve(
        case.face_discretization, case.rhs_matrix, case.div, case.accum, case.bound_vec,
    )
    post = TpsaPostprocessing(case)
    gradient = post.gradient_green_gauss()
    assert np.ptp(post.p) > 1.0
    np.testing.assert_allclose(
        gradient[0, 0] + gradient[1, 1], post.p / case.material.lame_parameter,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        gradient[0, 1] - gradient[1, 0], post.r / case.material.isotropic_shear_modulus,
        atol=1e-14,
    )


@pytest.mark.parametrize("interior_value", [np.nan, np.inf, -np.inf])
def test_reconstruction_ignores_unused_boundary_values_without_mutation(
    interior_value: float,
) -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.solve()
    post = TpsaPostprocessing(case)
    expected = post.strain_green_gauss()
    interior = np.setdiff1d(np.arange(post.grid.num_faces), post.boundary_faces)
    post.u_boundary[:, interior] = interior_value
    arrays = (post.x, post.bound_vec, post.grid.face_normals, post.grid.cell_volumes)
    originals = [array.copy() for array in arrays]
    np.testing.assert_allclose(post.strain_green_gauss(), expected, atol=1e-14)
    for array, original in zip(arrays, originals):
        np.testing.assert_array_equal(array, original)


def test_reconstruction_refreshes_after_in_place_updates() -> None:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.solve()
    post = TpsaPostprocessing(case)
    assert np.max(np.abs(post.strain_green_gauss())) > 1e-5
    post.x[:] = 0.0
    translation = np.array([[2e-4], [-3e-4]])
    post.u_cell[:] = translation
    post.u_boundary[:, post.boundary_faces] = translation
    np.testing.assert_allclose(post.strain_green_gauss(), 0.0, atol=1e-14)
    assert post.u_face is not None
    np.testing.assert_allclose(
        post.u_face, np.repeat(translation, post.grid.num_faces, axis=1), atol=1e-14,
    )
    post.reconstruct_face_displacement()
    assert post.grad_u is post.epsilon is None
    post.p[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        post.strain_green_gauss()
    assert post.u_face is post.grad_u is post.epsilon is None


@pytest.mark.parametrize("lengths", [(2.0, 0.75), (0.3, 1.7)])
@pytest.mark.parametrize("angle", [0.0, np.pi / 6], ids=["aligned", "rotated"])
def test_green_gauss_integrates_quadratic_displacement(
    lengths: tuple[float, float], angle: float,
) -> None:
    """Exact face averages recover the analytic volume-average gradient.

    One rectangular cell isolates the surface sum from interior face recovery.
    Two-point Gauss quadrature integrates each quadratic face trace exactly.
    """
    case = PlaneStrainTpsa(cells_per_axis=1)
    case.prepare_simulation()
    grid = case.grid
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)],
    ])
    grid.nodes[:2] = rotation @ (np.array(lengths)[:, None] * grid.nodes[:2])
    grid.compute_geometry()
    _rediscretize(case)

    def displacement(points: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y = points
        return np.array([x**2 + 2*x*y + 3*y**2, 4*x**2 - x*y + 2*y**2])

    # Rotating area-weighted normals gives tangents of length equal to each face.
    offset = np.vstack((grid.face_normals[1], -grid.face_normals[0])) / np.sqrt(12.0)
    case.bc_values[:] = 0.5 * (
        displacement(grid.face_centers[:2] + offset)
        + displacement(grid.face_centers[:2] - offset)
    )
    case.x = _solve(
        case.face_discretization, case.rhs_matrix, case.div, case.accum, case.bound_vec,
    )
    post = TpsaPostprocessing(case)
    epsilon = post.strain_green_gauss()
    assert post.grad_u is not None and post.u_face is not None

    # The analytic gradient is linear, so its cell average equals its center value.
    x, y = grid.cell_centers[:2, 0]
    expected_gradient = np.array([[2*x + 2*y, 2*x + 6*y], [8*x - y, -x + 4*y]])
    expected_strain = np.array([
        [2*x + 2*y, 5*x + 2.5*y, 0.0],
        [5*x + 2.5*y, -x + 4*y, 0.0],
        [0.0, 0.0, 0.0],
    ])
    np.testing.assert_allclose(post.u_face, case.bc_values, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        post.grad_u, expected_gradient[:, :, None], rtol=1e-13, atol=1e-13,
    )
    np.testing.assert_allclose(
        epsilon, expected_strain[:, :, None], rtol=1e-13, atol=1e-13,
    )


def test_green_gauss_interior_faces_cancel_in_domain_integral(
    heterogeneous_case: PlaneStrainTpsa,
) -> None:
    """The volume integral depends only on the prescribed external boundary."""
    post = TpsaPostprocessing(heterogeneous_case)
    epsilon = post.strain_green_gauss()
    assert post.grad_u is not None
    grid, bf = post.grid, post.boundary_faces
    center = np.average(grid.cell_centers[:2], axis=1, weights=grid.cell_volumes)
    normals = grid.face_normals[:2, bf]
    # For this convex rectangle, point normals away from the domain centroid.
    direction = np.sign(np.sum(normals * (grid.face_centers[:2, bf] - center[:, None]), axis=0))
    boundary_integral = post.u_boundary[:, bf] @ (normals * direction).T
    np.testing.assert_allclose(
        np.sum(post.grad_u * grid.cell_volumes, axis=2), boundary_integral,
        rtol=1e-13, atol=1e-13,
    )
    expected_strain_integral = np.zeros((3, 3))
    expected_strain_integral[:2, :2] = 0.5 * (boundary_integral + boundary_integral.T)
    np.testing.assert_allclose(
        np.sum(epsilon * grid.cell_volumes, axis=2), expected_strain_integral,
        rtol=1e-13, atol=1e-13,
    )
