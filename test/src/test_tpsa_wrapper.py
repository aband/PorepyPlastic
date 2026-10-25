"""Elastic TPSA wrapper: dimension handling, boundary loads, and validation."""

from typing import cast

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray

from core.tpsa_wrapper import Coefficient, ElasticTpsa


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def grid(request: pytest.FixtureRequest) -> pp.Grid:
    # Nonuniform spacing exposes missing cell volumes and face-area factors.
    coordinates = [
        np.array([0.0, 0.3, 1.4]),
        np.array([0.0, 0.4, 1.0]),
        np.array([0.0, 0.2, 0.9]),
    ]
    result = pp.TensorGrid(*coordinates[:request.param])
    result.compute_geometry()
    return result


@pytest.fixture
def bc(grid: pp.Grid) -> pp.BoundaryConditionVectorial:
    return pp.BoundaryConditionVectorial(grid, grid.get_all_boundary_faces(), "dir")


@pytest.fixture
def wrapper(grid: pp.Grid, bc: pp.BoundaryConditionVectorial) -> ElasticTpsa:
    return ElasticTpsa(grid, bc, mu=2.0, lam=3.0)


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_matrix_dimensions_and_material_storage(
    grid: pp.Grid, bc: pp.BoundaryConditionVectorial, heterogeneous: bool,
) -> None:
    nc, nf, nd = grid.num_cells, grid.num_faces, grid.dim
    nr = 1 if nd == 2 else 3
    mu: Coefficient = np.linspace(1.5, 2.5, nc) if heterogeneous else 2.0
    lam: Coefficient = np.linspace(2.0, 4.0, nc) if heterogeneous else 3.0
    expected_mu = np.broadcast_to(mu, (nc,)).copy()
    expected_lam = np.broadcast_to(lam, (nc,)).copy()
    model = ElasticTpsa(grid, bc, mu, lam, keyword="elastic_test")
    matrices = model.matrices

    nx, nq = (nd + nr + 1) * nc, (nd + nr + 1) * nf
    shapes = {
        "face_discretization": (nq, nx),
        "rhs_matrix": (nq, nd * nf),
        "div": (nx, nq),
        "accum": (nx, nx),
        "cell_matrix": (nx, nx),
    }
    for name, shape in shapes.items():
        matrix = getattr(matrices, name)
        assert isinstance(matrix, sps.csr_array)
        assert matrix.shape == shape
        assert np.all(np.isfinite(matrix.data))

    params = model.data[pp.PARAMETERS]["elastic_test"]
    assert params["bc"] is bc
    assert matrices.raw is model.data[pp.DISCRETIZATION_MATRICES]["elastic_test"]
    if isinstance(mu, np.ndarray) and isinstance(lam, np.ndarray):
        # The wrapper must own its coefficients, even if inputs are later edited.
        mu[:] = -1.0
        lam[:] = -1.0
    np.testing.assert_array_equal(params["fourth_order_tensor"].mu, expected_mu)
    np.testing.assert_array_equal(params["fourth_order_tensor"].lmbda, expected_lam)
    expected_accumulation = np.concatenate((
        np.zeros(nd * nc),
        np.repeat(grid.cell_volumes / expected_mu, nr),
        grid.cell_volumes / expected_lam,
    ))
    np.testing.assert_allclose(matrices.accum.diagonal(), expected_accumulation)


@pytest.mark.parametrize("mixed_boundary", [False, True], ids=["dirichlet", "mixed"])
@pytest.mark.parametrize("field", ["translation", "rotation", "affine"])
def test_exact_affine_solution_and_face_tractions(
    grid: pp.Grid, mixed_boundary: bool, field: str,
) -> None:
    """Check u/r/p and integrated face forces against continuum elasticity."""
    nd, nc, nf = grid.dim, grid.num_cells, grid.num_faces
    mu, lam = 2.0, 3.0
    gradient = np.array([
        [0.1, 0.2, -0.3], [-0.4, 0.5, 0.6], [0.7, -0.8, 0.9],
    ])[:nd, :nd].copy()
    if field == "translation":
        gradient[:] = 0.0
    elif field == "rotation":
        gradient = 0.5 * (gradient - gradient.T)
    translation = np.arange(1, nd + 1, dtype=float) / 7
    sigma = mu * (gradient + gradient.T) + lam * np.trace(gradient) * np.eye(nd)

    boundary = grid.get_all_boundary_faces()
    dirichlet = (boundary[np.isclose(grid.face_centers[0, boundary], 0.0)]
                 if mixed_boundary else boundary)
    neumann = np.setdiff1d(boundary, dirichlet)
    bc = pp.BoundaryConditionVectorial(grid, dirichlet, "dir")
    model = ElasticTpsa(grid, bc, mu, lam)
    values = np.zeros((nd, nf))
    values[:, dirichlet] = gradient @ grid.face_centers[:nd, dirichlet] + translation[:, None]
    if neumann.size:
        signs = np.asarray(grid.cell_faces[neumann].sum(axis=1)).ravel()
        # PorePy face normals already contain face areas. Orient them outward.
        values[:, neumann] = sigma @ (grid.face_normals[:nd, neumann] * signs)
    packed = values.ravel(order="F")
    rhs = model.boundary_rhs(packed)

    u_exact = gradient @ grid.cell_centers[:nd] + translation[:, None]
    rotation = (np.array([gradient[0, 1] - gradient[1, 0]]) if nd == 2 else np.array([
        gradient[1, 2] - gradient[2, 1],
        gradient[2, 0] - gradient[0, 2],
        gradient[0, 1] - gradient[1, 0],
    ]))
    r_exact = np.repeat((mu * rotation)[:, None], nc, axis=1)
    p_exact = np.full(nc, lam * np.trace(gradient))
    exact = np.concatenate((u_exact.ravel(order="F"), r_exact.ravel(order="F"), p_exact))
    matrices = model.matrices
    np.testing.assert_allclose(matrices.cell_matrix @ exact, rhs, rtol=1e-11, atol=1e-12)

    solution = sps.linalg.spsolve(matrices.cell_matrix, rhs)
    np.testing.assert_allclose(solution, exact, rtol=1e-10, atol=1e-11)
    flux = matrices.face_discretization @ solution + matrices.rhs_matrix @ packed
    traction = flux[:nd * nf].reshape((nd, nf), order="F")
    np.testing.assert_allclose(traction, sigma @ grid.face_normals[:nd], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(grid.divergence(dim=nd) @ traction.ravel(order="F"),
                               0.0, atol=1e-11)


def test_boundary_rhs_reuses_operators_without_mutating_inputs(
    wrapper: ElasticTpsa, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_rebuild(*args: object, **kwargs: object) -> None:
        pytest.fail("Changing boundary values must not rebuild the discretization.")

    monkeypatch.setattr(pp.Tpsa, "discretize", unexpected_rebuild)
    monkeypatch.setattr(wrapper, "_assemble_matrices", unexpected_rebuild)
    matrices = wrapper.matrices
    saved_matrix = matrices.cell_matrix.copy()
    values = np.linspace(-0.5, 0.8, wrapper.grid.dim * wrapper.grid.num_faces)
    saved_values = values.copy()
    zero = wrapper.boundary_rhs(np.zeros_like(values))
    first = wrapper.boundary_rhs(values)
    assert first.shape == (matrices.cell_matrix.shape[0],)
    assert first.dtype == np.float64
    np.testing.assert_array_equal(zero, np.zeros_like(first))
    np.testing.assert_allclose(wrapper.boundary_rhs(-2 * values), -2 * first)
    np.testing.assert_array_equal(values, saved_values)
    assert wrapper.matrices is matrices
    np.testing.assert_array_equal(matrices.cell_matrix.data, saved_matrix.data)
    first[:] = 0.0
    assert np.linalg.norm(wrapper.boundary_rhs(values)) > 0.0


@pytest.mark.parametrize("invalid,match", [
    ("scalar", "shape"), ("short", "shape"), ("long", "shape"),
    ("matrix", "shape"), ("nan", "finite"), ("inf", "finite"), ("complex", "real"),
])
def test_boundary_rhs_rejects_invalid_data(
    wrapper: ElasticTpsa, invalid: str, match: str,
) -> None:
    nd, nf = wrapper.grid.dim, wrapper.grid.num_faces
    size = nd * nf
    values = {
        "scalar": np.array(0.0),
        "short": np.zeros(size - 1),
        "long": np.zeros(size + 1),
        "matrix": np.zeros((nd, nf)),
        "nan": np.full(size, np.nan),
        "inf": np.full(size, np.inf),
        "complex": np.full(size, 1j),
    }[invalid]
    with pytest.raises(ValueError, match=match):
        wrapper.boundary_rhs(cast(NDArray[np.float64], values))


@pytest.mark.parametrize("name", ["mu", "lam"])
@pytest.mark.parametrize("invalid,match", [
    (0.0, "positive"), (-1.0, "positive"), (np.nan, "finite"),
    (np.inf, "finite"), (1j, "real"),
    (np.ones(3), "shape"), (np.ones((2, 2)), "shape"),
])
def test_invalid_material_coefficients(
    grid: pp.Grid, bc: pp.BoundaryConditionVectorial,
    name: str, invalid: object, match: str,
) -> None:
    coefficients: dict[str, Coefficient] = {"mu": 2.0, "lam": 3.0}
    coefficients[name] = cast(Coefficient, invalid)
    with pytest.raises(ValueError, match=f"{name}.*{match}"):
        ElasticTpsa(grid, bc, **coefficients)


@pytest.mark.parametrize("attribute", ["dim", "num_faces"])
def test_incompatible_boundary_conditions(
    grid: pp.Grid, bc: pp.BoundaryConditionVectorial,
    attribute: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bc, attribute, getattr(bc, attribute) + 1)
    with pytest.raises(ValueError, match="Boundary-condition dimensions"):
        ElasticTpsa(grid, bc, mu=2.0, lam=3.0)


def test_unsupported_dimension() -> None:
    grid = pp.CartGrid([2])
    grid.compute_geometry()
    bc = pp.BoundaryConditionVectorial(grid, grid.get_all_boundary_faces(), "dir")
    with pytest.raises(ValueError, match="dimension"):
        ElasticTpsa(grid, bc, mu=2.0, lam=3.0)


def test_empty_grid(
    grid: pp.Grid, bc: pp.BoundaryConditionVectorial, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grid, "num_cells", 0)
    with pytest.raises(ValueError, match="at least one cell"):
        ElasticTpsa(grid, bc, mu=2.0, lam=3.0)
