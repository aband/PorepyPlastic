"""Tensor regression tests.

Run from the repository root:
    python -m pytest test/test_tensor.py -q

These tests expect matrix multiplication, outer products, and double
contractions to return a general Tensor. Addition, subtraction, scalar
arithmetic, and symmetric componentwise products preserve the concrete type.
"""

from typing import Self, get_type_hints

import numpy as np
import pytest

from tensor import Tensor, strain, stress, symmetricSecondOrderTensor


@pytest.mark.parametrize("tensor_type", [stress, strain])
def test_zeros_and_identity_preserve_concrete_type(tensor_type):
    zero = tensor_type.zeros((3, 3))
    identity = tensor_type.identity(3)

    assert type(zero) is tensor_type
    assert type(identity) is tensor_type
    np.testing.assert_array_equal(zero.to_numpy(), np.zeros((3, 3)))
    np.testing.assert_array_equal(identity.to_numpy(), np.eye(3))


def test_tensor_properties():
    value = Tensor([[1.0, 2.0], [2.0, 3.0]])

    assert value.tensor_order == 2
    assert value.rank == 2
    assert value.shape == (2, 2)
    assert value.is_identity_dimension
    assert value.is_symmetric


def test_rectangular_tensor_is_not_equal_dimensioned():
    value = Tensor(np.zeros((2, 3)))

    assert not value.is_identity_dimension
    with pytest.raises(ValueError, match="identical dimensions"):
        _ = value.is_symmetric


def test_nonsymmetric_tensor_is_detected():
    value = Tensor([[1.0, 2.0], [0.0, 1.0]])

    assert not value.is_symmetric


def test_constructor_copies_input_by_default():
    data = np.eye(2)
    value = Tensor(data)

    data[0, 0] = 10.0

    assert value.to_numpy()[0, 0] == pytest.approx(1.0)


def test_to_numpy_copy_is_independent():
    value = Tensor(np.eye(2))
    result = value.to_numpy()

    result[0, 0] = 10.0

    assert value.to_numpy()[0, 0] == pytest.approx(1.0)


def test_to_numpy_view_is_read_only():
    value = Tensor(np.eye(2))
    result = value.to_numpy(copy=False)

    with pytest.raises(ValueError):
        result[0, 0] = 10.0


@pytest.mark.parametrize("tensor_type", [stress, strain])
def test_arithmetic_preserves_concrete_type_and_values(tensor_type):
    left_data = np.array([[2.0, 1.0], [1.0, 4.0]])
    right_data = 0.5 * left_data
    left = tensor_type(left_data)
    right = tensor_type(right_data)

    results = [
        (left + right, left_data + right_data),
        (left - right, left_data - right_data),
        (-left, -left_data),
        (left * 2.0, left_data * 2.0),
        (2.0 * left, left_data * 2.0),
        (left / 2.0, left_data / 2.0),
        (left.hadamard(right), left_data * right_data),
    ]

    for result, expected in results:
        assert type(result) is tensor_type
        np.testing.assert_allclose(result.to_numpy(), expected)


@pytest.mark.parametrize("tensor_type", [Tensor, stress, strain])
def test_matrix_product_returns_general_tensor(tensor_type):
    # Symmetric matrices need not have a symmetric matrix product.
    left = tensor_type([[2.0, 1.0], [1.0, 4.0]])
    right = tensor_type([[3.0, 0.0], [0.0, 1.0]])

    result = left @ right

    assert type(result) is Tensor
    np.testing.assert_allclose(result.to_numpy(), [[6.0, 1.0], [3.0, 4.0]])
    assert not result.is_symmetric


def test_stress_subtraction_returns_new_stress():
    left = stress(np.diag([3.0, 4.0, 5.0]))
    right = stress(np.diag([1.0, 1.5, 2.0]))

    result = left - right

    assert type(result) is stress
    assert result is not left
    assert result is not right
    np.testing.assert_allclose(result.to_numpy(), np.diag([2.0, 2.5, 3.0]))


def test_subtraction_return_annotation_is_self():
    assert get_type_hints(Tensor.__sub__)["return"] is Self


@pytest.mark.parametrize("operation", ["add", "subtract", "hadamard"])
def test_binary_operations_reject_different_shapes(operation):
    left = stress(np.eye(2))
    right = stress(np.eye(3))

    with pytest.raises(ValueError, match="Tensor shapes must agree"):
        if operation == "add":
            _ = left + right
        elif operation == "subtract":
            _ = left - right
        else:
            _ = left.hadamard(right)


def test_invalid_scalar_multiplication_raises_type_error():
    value = stress(np.eye(2))

    with pytest.raises(TypeError):
        _ = value * "invalid"


def test_division_by_zero_raises():
    value = stress(np.eye(2))

    with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
        _ = value / 0.0


def test_inner_product():
    left_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    right_data = np.array([[5.0, 6.0], [7.0, 8.0]])
    left = Tensor(left_data)
    right = Tensor(right_data)

    result = left.inner(right)

    assert result == pytest.approx(np.sum(left_data * right_data))


def test_outer_product_for_general_tensors():
    left_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    right_data = np.array([[5.0, 6.0], [7.0, 8.0]])
    left = Tensor(left_data)
    right = Tensor(right_data)

    result = left.outer(right)

    assert type(result) is Tensor
    assert result.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(result.to_numpy(), np.tensordot(left_data, right_data, axes=0))


@pytest.mark.parametrize("tensor_type", [stress, strain])
def test_outer_product_of_second_order_tensors_returns_rank_four(tensor_type):
    left_data = np.array([[2.0, 1.0], [1.0, 4.0]])
    right_data = np.array([[3.0, 0.0], [0.0, 1.0]])
    left = tensor_type(left_data)
    right = tensor_type(right_data)

    result = left.outer(right)

    assert type(result) is Tensor
    assert result.rank == 4
    assert result.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(
        result.to_numpy(), np.einsum("ij,kl->ijkl", left_data, right_data)
    )


def test_double_contraction():
    fourth_order_data = np.arange(16.0).reshape(2, 2, 2, 2)
    second_order_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    fourth_order = Tensor(fourth_order_data)
    second_order = Tensor(second_order_data)

    result = fourth_order.double_contract(second_order)
    expected = np.tensordot(
        fourth_order_data,
        second_order_data,
        axes=((-2, -1), (0, 1)),
    )

    assert type(result) is Tensor
    np.testing.assert_allclose(result.to_numpy(), expected)


@pytest.mark.parametrize("tensor_type", [stress, strain])
def test_double_contraction_of_second_order_tensors_returns_rank_zero(tensor_type):
    left = tensor_type([[2.0, 1.0], [1.0, 4.0]])
    right = tensor_type([[3.0, 2.0], [2.0, 1.0]])

    result = left.double_contract(right)

    assert type(result) is Tensor
    assert result.rank == 0
    assert result.shape == ()
    # Full contraction includes both off-diagonal entries: 6 + 2 + 2 + 4.
    assert result.to_numpy().item() == pytest.approx(14.0)


def test_double_contraction_requires_rank_two_or_greater():
    vector = Tensor([1.0, 2.0])
    matrix = Tensor(np.eye(2))

    with pytest.raises(ValueError, match="rank two or greater"):
        vector.double_contract(matrix)


def test_double_contraction_checks_contracted_dimensions():
    left = Tensor(np.zeros((2, 3, 4)))
    right = Tensor(np.zeros((2, 2)))

    with pytest.raises(ValueError, match="Contracted dimensions do not agree"):
        left.double_contract(right)


def test_second_order_tensor_rejects_invalid_order_and_shape():
    with pytest.raises(ValueError, match="Order does not equal 2"):
        symmetricSecondOrderTensor([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Not identical dimensions"):
        symmetricSecondOrderTensor(np.zeros((2, 3)))


@pytest.mark.parametrize("tensor_type", [stress, strain])
def test_from_mandel_preserves_concrete_type(tensor_type):
    vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    expected = np.array(
        [
            [1.0, 6.0 * inverse_sqrt_two, 5.0 * inverse_sqrt_two],
            [6.0 * inverse_sqrt_two, 2.0, 4.0 * inverse_sqrt_two],
            [5.0 * inverse_sqrt_two, 4.0 * inverse_sqrt_two, 3.0],
        ]
    )

    result = tensor_type.from_mandel(vector)

    assert type(result) is tensor_type
    np.testing.assert_allclose(result.to_numpy(), expected)


def test_from_mandel_rejects_invalid_shape():
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        stress.from_mandel(np.zeros(3))


def test_mandel_round_trip():
    vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    value = stress.from_mandel(vector)

    value.create_mandel_representation()

    np.testing.assert_allclose(value.mandel_vec, vector)


@pytest.mark.parametrize("tensor_type", [stress, strain])
def test_spherical_and_deviatoric_parts(tensor_type):
    value = tensor_type(np.diag([3.0, 6.0, 9.0]))

    spherical = value.spherical()
    deviatoric = value.deviatoric()

    assert type(spherical) is tensor_type
    assert type(deviatoric) is tensor_type
    assert value.trace == pytest.approx(18.0)
    assert value.mean == pytest.approx(6.0)
    np.testing.assert_allclose(spherical.to_numpy(), 6.0 * np.eye(3))
    np.testing.assert_allclose(deviatoric.to_numpy(), np.diag([-3.0, 0.0, 3.0]))
    assert deviatoric.trace == pytest.approx(0.0)


def test_hydrostatic_part_preserves_stress_type():
    value = stress(np.diag([3.0, 6.0, 9.0]))

    result = value.hydrostatic()

    assert type(result) is stress
    np.testing.assert_allclose(result.to_numpy(), 6.0 * np.eye(3))
