"""
This file contains a wrapper aroung np.ndarray representing 
symmetricSecondOrderTensor and strain.

This tensor wrapper is used for
1. Material point evaluation.
2. Post-processing

Data convertion mainly use mandel representation.
"""

from __future__ import annotations

from abc import ABC
from typing import Self

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

import itertools

class Tensor(ABC):
    """Class of the common parts of the arbitrary rank of Tensor."""

    __slots__ = ("_data",)

    def __init__(self, 
                 data:ArrayLike, 
                 *,
                 copy:bool = True) -> None:
        self._data = np.array(
            data,
            dtype = np.float64,
            copy  = copy
        )

    @property
    def tensor_order(self) -> int:
        return self._data.ndim

    @property
    def is_identity_dimension(self)->bool:
        if len(set(self._data.shape)) > 1:
            return False
        else:
            return True

    @property
    def is_symmetric(self) -> bool:

        # Tensor should have identical dimensions to be symmetric
        if not self.is_identity_dimension:
            raise ValueError("Tensor does not have identical dimensions.")

        for perm in itertools.permutations(range(self._data.ndim)):
            if not np.allclose(self._data, np.transpose(self._data, perm), atol=10e-14):
                return False
        return True

    @classmethod
    def zeros(cls,shape:tuple[int,...],) -> Self:
        return cls(np.zeros(shape,dtype=np.float64),copy=False)

    @classmethod
    def identity(cls, dimension: int) -> Self:
        return cls(np.eye(dimension,dtype=np.float64),copy=False)

    @property
    def shape(self)->tuple[int,...]:
        return self._data.shape

    @property
    def rank(self)->int:
        return self._data.ndim

    def _same_shape(self, other:Tensor) -> None:
        if self.shape != other.shape:
            raise ValueError(
                "Tensor shapes must agree: "
                f"{self.shape} != {other.shape}."
            )

    def __add__(self, other:Tensor) -> Self:
        self._same_shape(other)
        return type(self)(
            self._data + other._data,
            copy=False
        )

    def __sub__(self, other:Tensor) -> Self:
        self._same_shape(other)
        return type(self)(
            self._data - other._data,
            copy=False
        )

    def __neg__(self) -> Self:
        return type(self)(
            -self._data,
            copy=False,
        )

    def __mul__(
        self,
        scalar: object,
    ) -> Self:
        if not isinstance(
            scalar,
            (Real, np.integer, np.floating),
        ):
            return NotImplemented

        return type(self)(
            self._data * float(scalar),
            copy=False,
        )

    def __rmul__(
        self,
        scalar: object,
    ) -> Self:
        return self.__mul__(scalar)

    def __truediv__(
        self,
        scalar: object,
    ) -> Tensor:
        if not isinstance(
            scalar,
            (Real, np.integer, np.floating),
        ):
            return NotImplemented

        scalar_value = float(scalar)

        if scalar_value == 0.0:
            raise ZeroDivisionError(
                "Cannot divide by zero."
            )

        return type(self)(
            self._data / scalar_value,
            copy=False,
        )

    def __matmul__(
        self,
        other: Tensor,
    ) -> Self:
        return type(self)(
            np.matmul(self._data, other._data),
            copy=False,
            )

    def hadamard(
        self,
        other: Tensor,
    ) -> Self:
        """Componentwise multiplication."""
        self._same_shape(other)

        return type(self)(
            self._data * other._data,
            copy=False,
        )

    def inner(
        self,
        other: Tensor,
    ) -> float:
        """Full contraction of two equal-order tensors."""
        self._same_shape(other)

        return float(
            np.tensordot(
                self._data,
                other._data,
                axes=self.rank,
            )
        )

    def outer(
        self,
        other: Tensor,
    ) -> Tensor:
        """Tensor product with no contracted indices."""
        return type(self)(
            np.tensordot(
                self._data,
                other._data,
                axes=0,
            ),
            copy=False,
        )

    def double_contract(
        self,
        other: Tensor,
    ) -> Tensor:
        """Contract the final two axes of self with
        the first two axes of other.
        """
        if self.rank < 2 or other.rank < 2:
            raise ValueError(
                "Double contraction requires tensors "
                "of rank two or greater."
            )

        if self.shape[-2:] != other.shape[:2]:
            raise ValueError(
                "Contracted dimensions do not agree: "
                f"{self.shape[-2:]} != {other.shape[:2]}."
            )

        return type(self)(
            np.tensordot(
                self._data,
                other._data,
                axes=(
                    (-2, -1),
                    (0, 1),
                    ),
                    ),
                copy=False
            )

    def to_numpy(
        self,
        *,
        copy: bool = True,
    ) -> np.ndarray:
        if copy:
            return self._data.copy()

        view = self._data.view()
        view.flags.writeable = False
        return view

    @property
    def print_as_matrix(self) ->None:
        #project the target tensor into matrix form
        pass

class symmetricSecondOrderTensor(Tensor):
    """
    Must be a second order tensor.
    With identical dimensions.
    """

    def __init__(self, 
                 value: ArrayLike, 
                 *,
                 copy : bool = True) -> None:
        super().__init__(value,copy = copy)
        if self.tensor_order != 2:
            raise ValueError("Not a symmetricSecondOrderTensor tensor. Order does not equal 2.")
        if not self.is_identity_dimension:
            raise ValueError("Not a symmetricSecondOrderTensor tensor. Not identical dimensions.")

        # Get dimension (Useful when storing all symmetricSecondOrderTensor tensor for all cells)
        #self.dimension = self._data.shape[0]
        # Hydrostatic symmetricSecondOrderTensor
        #self.hydrostatic = 1.0/3.0 * self._data.trace()
        # Deviatoric symmetricSecondOrderTensor
        #self.deviatoric = self._data - self.hydrostatic*np.eye(self.dimension)

    @classmethod
    def from_mandel(
        cls,
        vector: ArrayLike,
    ) -> Self:
        """
        Create Tensor object from Mandel representation.
        """
        vector_array = np.asarray(
            vector,
            dtype=np.float64,
        )

        if vector_array.shape != (6,):
            raise ValueError(
                "Expected a Mandel vector with shape (6,)."
            )

        inverse_sqrt_two = 1.0 / np.sqrt(2.0)

        return cls([
            [
                vector_array[0],
                vector_array[5] * inverse_sqrt_two,
                vector_array[4] * inverse_sqrt_two,
            ],
            [
                vector_array[5] * inverse_sqrt_two,
                vector_array[1],
                vector_array[3] * inverse_sqrt_two,
            ],
            [
                vector_array[4] * inverse_sqrt_two,
                vector_array[3] * inverse_sqrt_two,
                vector_array[2],
            ],
        ])

    def create_mandel_representation(self) -> None:
        if not self.is_symmetric:
            raise ValueError("Not a symmetrical tensor, no mandel representation.")

        self.mandel_vec = np.array([self._data[0,0],
                                    self._data[1,1],
                                    self._data[2,2],
                                    np.sqrt(2.0)*self._data[1,2],
                                    np.sqrt(2.0)*self._data[0,2],
                                    np.sqrt(2.0)*self._data[0,1],
                                    ],dtype=np.float64)

    @property
    def dimension(self)->int:
        return self._data.shape[0] 

    @property
    def trace(self)->float:
        return float(np.trace(self._data))

    @property
    def mean(self)->float:
        return self.trace/self.dimension

    def spherical(self)->Self:
        spherical_data = (
            self.mean *
            np.eye(
                self.dimension,
                dtype=np.float64
                )
        )
        return type(self)(
            spherical_data,
            copy=False,
        )

    def deviatoric(self)->Self:
        """
        Return the deviatoric part of the target tensor.
        """
        deviatoric_data = (
            self._data
            - self.spherical().to_numpy(copy=False)
        )

        return type(self)(
            deviatoric_data,
            copy=False,
        )

class strain(symmetricSecondOrderTensor):
    __slots__ = ()

class stress(symmetricSecondOrderTensor):
    __slots__ = ()

    def hydrostatic(self) -> Self:
        return self.spherical()