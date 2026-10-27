"""Batched Mandel conventions shared by strain reconstruction and Jacobians."""

from typing import cast

import numpy as np
from numpy.typing import NDArray
import pytest

from material.tensor import strain, stress
from utils.reconstruction import MANDEL_BASIS, from_mandel, mandel, mandel_basis


def test_default_2d_matches_legacy_order_and_plane_strain_embedding() -> None:
    tensors = np.array([
        [[1.0, 11.0], [4.0, 14.0], [5.0, 15.0]],
        [[4.0, 14.0], [2.0, 12.0], [6.0, 16.0]],
        [[5.0, 15.0], [6.0, 16.0], [3.0, 13.0]],
    ])
    root2 = np.sqrt(2.0)
    expected = np.array([1.0, 2.0, root2 * 4.0, 11.0, 12.0, root2 * 14.0])
    np.testing.assert_array_equal(mandel(tensors), expected)
    np.testing.assert_array_equal(mandel(tensors, dim=2), expected)
    restored = from_mandel(expected)
    np.testing.assert_array_equal(restored, from_mandel(expected, dim=2))
    np.testing.assert_allclose(restored[:2, :2], tensors[:2, :2])
    np.testing.assert_array_equal(restored[2], 0.0)
    np.testing.assert_array_equal(restored[:, 2], 0.0)
    # The in-plane basis used by the existing Jacobians must retain its meaning.
    basis_projection = np.einsum("aij,ijc->ac", MANDEL_BASIS, tensors[:2, :2])
    np.testing.assert_allclose(basis_projection.ravel(order="F"), expected)
    np.testing.assert_array_equal(mandel(tensors[:2, :2]), expected)


def test_3d_order_and_shear_scaling() -> None:
    tensors = np.array([
        [[1.0, 11.0], [6.0, 16.0], [5.0, 15.0]],
        [[6.0, 16.0], [2.0, 12.0], [4.0, 14.0]],
        [[5.0, 15.0], [4.0, 14.0], [3.0, 13.0]],
    ])
    root2 = np.sqrt(2.0)
    expected = np.array([
        1.0, 2.0, 3.0, root2 * 4.0, root2 * 5.0, root2 * 6.0,
        11.0, 12.0, 13.0, root2 * 14.0, root2 * 15.0, root2 * 16.0,
    ])
    np.testing.assert_array_equal(mandel(tensors, dim=3), expected)
    np.testing.assert_allclose(from_mandel(expected, dim=3), tensors)


@pytest.mark.parametrize("tensor_type", [strain, stress])
def test_3d_matches_material_tensor_conversions(
    tensor_type: type[strain] | type[stress],
) -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(size=24)
    tensors = from_mandel(values, dim=3)
    for cell, vector in enumerate(values.reshape((-1, 6))):
        material_tensor = tensor_type.from_mandel(vector)
        np.testing.assert_allclose(tensors[:, :, cell], material_tensor.to_numpy())
        material_tensor.create_mandel_representation()
        actual = mandel(tensors[:, :, cell:cell + 1], dim=3)
        np.testing.assert_allclose(actual, material_tensor.mandel_vec)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("num_cells", [0, 1, 5])
def test_round_trip_and_owned_storage(dim: int, num_cells: int) -> None:
    rng = np.random.default_rng(29)
    raw = rng.normal(size=(3, 3, num_cells))
    tensors = 0.5 * (raw + raw.swapaxes(0, 1))
    saved = tensors.copy()
    encoded = mandel(tensors, dim=dim)
    decoded = from_mandel(encoded, dim=dim)
    assert encoded.shape == ((3 if dim == 2 else 6) * num_cells,)
    assert decoded.shape == (3, 3, num_cells)
    assert encoded.dtype == decoded.dtype == np.float64
    np.testing.assert_allclose(decoded[:dim, :dim], tensors[:dim, :dim])
    np.testing.assert_array_equal(decoded, decoded.swapaxes(0, 1))
    np.testing.assert_allclose(mandel(decoded, dim=dim), encoded)
    if dim == 2:
        np.testing.assert_array_equal(decoded[2], 0.0)
        np.testing.assert_array_equal(decoded[:, 2], 0.0)
    assert not np.shares_memory(encoded, tensors)
    assert not np.shares_memory(decoded, encoded)
    saved_encoded = encoded.copy()
    decoded.fill(-123.0)
    np.testing.assert_array_equal(encoded, saved_encoded)
    encoded.fill(456.0)
    np.testing.assert_array_equal(tensors, saved)


@pytest.mark.parametrize("dim", [2, 3])
def test_mandel_preserves_tensor_inner_product(dim: int) -> None:
    rng = np.random.default_rng(31)
    left, right = rng.normal(size=(2, 3, 3, 4))
    left = 0.5 * (left + left.swapaxes(0, 1))
    right = 0.5 * (right + right.swapaxes(0, 1))
    m = 3 if dim == 2 else 6
    actual = np.sum(
        mandel(left, dim=dim).reshape((-1, m))
        * mandel(right, dim=dim).reshape((-1, m)), axis=1,
    )
    expected = np.einsum("ijc,ijc->c", left[:dim, :dim], right[:dim, :dim])
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("dim", [2, 3])
def test_noncontiguous_inputs(dim: int) -> None:
    rng = np.random.default_rng(41)
    raw = rng.normal(size=(3, 3, 8))
    tensors = (raw + raw.swapaxes(0, 1))[:, :, ::2]
    assert not tensors.flags.c_contiguous
    actual = mandel(tensors, dim=dim)
    np.testing.assert_array_equal(actual, mandel(tensors.copy(), dim=dim))
    storage = np.empty(2 * actual.size)
    storage[::2] = actual
    np.testing.assert_array_equal(
        from_mandel(storage[::2], dim=dim), from_mandel(actual, dim=dim),
    )


@pytest.mark.parametrize("dim", [-1, 0, 1, 4])
def test_invalid_dimension(dim: int) -> None:
    with pytest.raises(ValueError, match="dimension must be 2 or 3"):
        mandel(np.zeros((3, 3, 1)), dim=dim)
    with pytest.raises(ValueError, match="dimension must be 2 or 3"):
        from_mandel(np.zeros(6), dim=dim)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("shape", [(9,), (3, 3), (3, 2, 4), (4, 4, 2)])
def test_invalid_tensor_shape(dim: int, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="Expected tensors with shape"):
        mandel(np.zeros(shape), dim=dim)


def test_3d_requires_three_tensor_dimensions() -> None:
    with pytest.raises(ValueError, match="Expected tensors with shape"):
        mandel(np.zeros((2, 2, 4)), dim=3)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("invalid", ["scalar", "column", "short", "long"])
def test_invalid_mandel_vector_shape(dim: int, invalid: str) -> None:
    m = 3 if dim == 2 else 6
    values = {
        "scalar": np.array(0.0),
        "column": np.zeros((m, 1)),
        "short": np.zeros(m - 1),
        "long": np.zeros(m + 1),
    }[invalid]
    with pytest.raises(ValueError, match="Expected a flat Mandel vector"):
        from_mandel(values, dim=dim)


@pytest.mark.parametrize("dim", [2, 3])
def test_complex_values_are_not_silently_discarded(dim: int) -> None:
    with pytest.raises(ValueError, match="tensors must be real"):
        mandel(cast(NDArray[np.float64], np.full((3, 3, 1), 1j)), dim=dim)
    with pytest.raises(ValueError, match="values must be real"):
        from_mandel(cast(NDArray[np.float64], np.full(6, 1j)), dim=dim)


@pytest.mark.parametrize("dim", [2, 3])
def test_mandel_basis_is_orthonormal_and_matches_conversions(dim: int) -> None:
    basis = mandel_basis(dim)
    m = 3 if dim == 2 else 6
    assert basis.shape == (m, dim, dim)
    np.testing.assert_array_equal(basis, basis.swapaxes(1, 2))
    np.testing.assert_allclose(np.einsum("aij,bij->ab", basis, basis), np.eye(m), atol=1e-15)
    rng = np.random.default_rng(51)
    tensors = rng.normal(size=(3, 3, 4))
    tensors = 0.5 * (tensors + tensors.swapaxes(0, 1))
    projected = np.einsum("aij,ijc->ac", basis, tensors[:dim, :dim])
    np.testing.assert_allclose(projected.ravel(order="F"), mandel(tensors, dim=dim))
    restored = np.einsum("ac,aij->ijc", projected, basis)
    np.testing.assert_allclose(restored, tensors[:dim, :dim])


def test_default_mandel_basis_preserves_legacy_constant_and_owns_storage() -> None:
    expected = np.zeros((3, 2, 2))
    expected[0, 0, 0] = expected[1, 1, 1] = 1.0
    expected[2, 0, 1] = expected[2, 1, 0] = 1.0 / np.sqrt(2.0)
    np.testing.assert_array_equal(MANDEL_BASIS, expected)
    basis = mandel_basis()
    np.testing.assert_array_equal(basis, expected)
    assert not np.shares_memory(basis, MANDEL_BASIS)
    basis.fill(0.0)
    np.testing.assert_array_equal(mandel_basis(), expected)
    np.testing.assert_array_equal(MANDEL_BASIS, expected)


@pytest.mark.parametrize("dim", [0, 1, 4])
def test_mandel_basis_rejects_unsupported_dimensions(dim: int) -> None:
    with pytest.raises(ValueError, match="dimension must be 2 or 3"):
        mandel_basis(dim)
