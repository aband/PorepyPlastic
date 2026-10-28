"""TPSA with retained face displacements and nested constitutive updates.

Supports 2D plane strain and spatial 3D on orthogonal grids, including Cartesian subgrids,
with full-vector Dirichlet or Neumann conditions on boundary faces. Rotation stress
has one component in 2D and three in 3D; cell-major Mandel strains have three
and six components respectively. Analytical and local finite-difference
Jacobians, Newton backtracking and fixed/adaptive load schedules support both.

For 3D state creation use prototype.tpsa_state.TpsaState or RetainedState.zeros.
The shared utils.state.TpsaState and existing examples/visualization remain 2D.
A prepared case for prototype.runner requires grid, bc, material_points and
bc_values (dim, num_faces); elastic coefficients come from each material point.

Run the standalone 3D example with python -B -m prototype.example_3d.
Run the paper cantilever with python -B -m prototype.cantilever_example --study.
Run Cook's membrane with python -B -m prototype.cooks_membrane_example.
Run local tests with python -B -m pytest prototype/tests.
"""
