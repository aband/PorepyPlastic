"""
Adapt TPSA solver from porepy to my solver.

Take in FourthOrderTensor as stiffness matrix.
Take in displacement field and constitutive relationship matrix.
Receive fouth order C matrix generated from TPSA solver.

Perform least square approximation for the gradient of displacements.
"""

import porepy as pp
from porepy.grids.grid import Grid
from porepy.numerics.discretization import Discretization
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data
from porepy.params.tensor import FourthOrderTensor

#C = pp.FourthOrderTensor(mu, lmbda)
