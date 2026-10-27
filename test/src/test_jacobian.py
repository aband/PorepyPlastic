"""Fixed Jacobian maps, TPSA reduction, and material-correction tangents.

Use exact linear identities, surface sums, and full constitutive tensor actions;
no finite-difference approximation is used in these tests.
"""

from copy import deepcopy
from typing import cast

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

from core.jacobian import AnalyticalJacobian
from core.layout import FloatArray
from core.residual import LoadIncrement, ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint, MaterialState
from material.scalar_hardening import (
    HardeningParameters, isotropic_hardening_K, kinematic_hardening_H,
)
from material.tangent import analytical_material_tangent
from material.tensor import strain
from utils.material_evaluation import MaterialTrial, evaluate_material_at_strain
from utils.reconstruction import mandel


@pytest.fixture(params=[(2, 0.0), (2, 0.37), (3, 0.0), (3, 0.37)],
                ids=["2d", "2d-rotated", "3d", "3d-rotated"])
def grid(request: pytest.FixtureRequest) -> pp.Grid:
    dim, angle = request.param
    result = pp.TensorGrid(
        np.array([0.0, 0.3, 1.4]), np.array([0.0, 0.4, 1.0]),
        np.array([0.0, 0.2, 0.9]) if dim == 3 else None,
    )
    c, s = np.cos(angle), np.sin(angle)
    transform = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if dim == 3:
        transform = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ transform
    result.nodes[:] = transform @ result.nodes
    result.compute_geometry()
    return result


def _assembler(grid: pp.Grid, boundary_type: str = "mixed") -> ResidualAssembler:
    boundary = grid.get_all_boundary_faces()
    prescribed = {
        "dirichlet": boundary,
        "neumann": boundary[:0],
        "mixed": boundary[::2][::-1],
    }[boundary_type]
    bc = pp.BoundaryConditionVectorial(grid, prescribed, "dir")
    points = []
    for cell in range(grid.num_cells):
        material = _materialProperty()
        material.young_modulus = 9.0 + 3.0 * cell
        material.poisson_ratio = 0.25
        points.append(MaterialPoint(
            material, HardeningParameters(0.05, 0.05, 0.2, 0.4, 0.0),
            vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H,
        ))
    return ResidualAssembler(grid, bc, points)


@pytest.fixture(params=["dirichlet", "neumann", "mixed"])
def jacobian(grid: pp.Grid, request: pytest.FixtureRequest) -> AnalyticalJacobian:
    return AnalyticalJacobian(_assembler(grid, request.param))


def _loads(assembler: ResidualAssembler) -> LoadIncrement:
    layout = assembler.layout
    rng = np.random.default_rng(46)
    return LoadIncrement(
        rng.normal(size=(layout.dim, layout.dirichlet_faces.size)),
        rng.normal(size=(layout.dim, assembler.neumann_faces.size)),
        rng.normal(size=layout.cell_size),
    )


def test_auxiliary_matrix_matches_independent_surface_sums(jacobian: AnalyticalJacobian) -> None:
    assembler, layout = jacobian.assembler, jacobian.layout
    grid = jacobian.operators.grid
    nd, nr, nc = layout.dim, layout.rotation_dim, layout.num_cells
    z = np.random.default_rng(47).normal(size=layout.size)
    z.setflags(write=False)
    faces = np.zeros((nd, grid.num_faces))
    faces[:, layout.free_faces] = z[layout.face_slice].reshape((nd, -1), order="F")
    rotation = np.zeros((nr, nc))
    pressure = np.zeros(nc)
    incidence = grid.cell_faces.tocoo()
    for face, cell, sign in zip(incidence.row, incidence.col, incidence.data, strict=True):
        # Grid normals already contain area. No cached side maps are used here.
        normal = sign * grid.face_normals[:nd, face]
        displacement = faces[:, face]
        if nd == 2:
            rotation[0, cell] += displacement[0] * normal[1] - displacement[1] * normal[0]
        else:
            rotation[:, cell] += np.cross(displacement, normal)
        pressure[cell] += displacement @ normal
    mu = np.array([point.material.isotropic_shear_modulus for point in assembler.material_points])
    lam = np.array([point.material.lame_parameter for point in assembler.material_points])
    expected = np.zeros(layout.size)
    expected[layout.r_slice] = rotation.ravel(order="F") - np.repeat(grid.cell_volumes / mu, nr) * z[layout.r_slice]
    expected[layout.p_slice] = pressure - grid.cell_volumes / lam * z[layout.p_slice]
    np.testing.assert_allclose(jacobian.H @ z, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(jacobian.H.diagonal()[layout.r_slice], -np.repeat(grid.cell_volumes / mu, nr))
    np.testing.assert_allclose(jacobian.H.diagonal()[layout.p_slice], -grid.cell_volumes / lam)
    assert jacobian.H[layout.u_slice, :].nnz == 0
    assert jacobian.H[layout.face_slice, :].nnz == 0
    assert jacobian.H[:, layout.u_slice].nnz == 0


def test_fixed_maps_reproduce_elastic_fields_and_residual(jacobian: AnalyticalJacobian) -> None:
    assembler, layout, operators = jacobian.assembler, jacobian.layout, jacobian.operators
    nd, nc, ns = layout.dim, layout.num_cells, operators.num_sides
    m = 3 if nd == 2 else 6
    shapes = {
        "B_z": (m * nc, layout.size), "T": (nd * ns, layout.size),
        "S": (layout.size, nd * ns), "H": (layout.size, layout.size),
        "L": (layout.size, layout.size),
        "P_side": (nd * ns, m * nc), "P": (layout.size, m * nc),
    }
    for name, shape in shapes.items():
        matrix = getattr(jacobian, name)
        assert isinstance(matrix, sps.csr_array)
        assert matrix.shape == shape
        assert np.all(np.isfinite(matrix.data))
    assert jacobian.B_z[:, layout.cell_slice].nnz == 0
    z = np.random.default_rng(48).normal(size=layout.size)
    loads = _loads(assembler)
    origin = assembler.evaluate_elastic(np.zeros(layout.size), loads)
    actual = assembler.evaluate_elastic(z, loads)
    np.testing.assert_allclose(
        jacobian.T @ z + origin.traction_increment.ravel(order="F"),
        actual.traction_increment.ravel(order="F"), rtol=1e-12, atol=1e-12,
    )
    # Fixed boundary and source terms are affine offsets, not Jacobian entries.
    for block in (layout.u_slice, layout.r_slice, layout.p_slice, layout.face_slice):
        expected = actual.residual[block]
        np.testing.assert_allclose((jacobian.L @ z + origin.residual)[block], expected, rtol=1e-12, atol=1e-12)
    epsilon = operators.green_gauss_reconstruction(actual.face_increment)
    epsilon0 = operators.green_gauss_reconstruction(origin.face_increment)
    np.testing.assert_allclose(jacobian.B_z @ z, mandel(epsilon - epsilon0, dim=nd), atol=1e-12)


def test_balance_matrix_sums_outward_tractions(jacobian: AnalyticalJacobian) -> None:
    layout, operators = jacobian.layout, jacobian.operators
    nd = layout.dim
    traction = np.random.default_rng(49).normal(size=(nd, operators.num_sides))
    expected = np.zeros(layout.size)
    for side, (face, cell) in enumerate(zip(operators.side_faces, operators.side_cells, strict=True)):
        expected[nd * cell:nd * (cell + 1)] += traction[:, side]
        for component in range(nd):
            row = layout.face_dofs[component, face]
            if row >= 0:
                expected[row] += traction[component, side]
    np.testing.assert_allclose(jacobian.S @ traction.ravel(order="F"), expected, atol=1e-13)
    np.testing.assert_array_equal(jacobian.S.data, 1.0)
    assert jacobian.S[layout.r_slice, :].nnz == 0
    assert jacobian.S[layout.p_slice, :].nnz == 0
    free_side = np.isin(operators.side_faces, layout.free_faces)
    counts = 1 + np.repeat(free_side, nd)
    np.testing.assert_array_equal(np.asarray(jacobian.S.sum(axis=0)).ravel(), counts)


def test_eliminating_free_faces_recovers_tpsa_matrix_and_loads(jacobian: AnalyticalJacobian) -> None:
    assembler, layout = jacobian.assembler, jacobian.layout
    grid = jacobian.operators.grid
    nx = layout.cell_size
    L = jacobian.L
    diagonal = L[nx:, nx:].diagonal()
    assert np.all(diagonal > 0.0)
    np.testing.assert_allclose(L[nx:, nx:].toarray(), np.diag(diagonal), atol=1e-14)
    inverse = sps.diags_array(1.0 / diagonal, format="csr")
    reduced = L[:nx, :nx] - L[:nx, nx:] @ inverse @ L[nx:, :nx]
    np.testing.assert_allclose(
        reduced.toarray(), assembler.elastic.matrices.cell_matrix.toarray(), rtol=1e-12, atol=1e-12,
    )

    loads = _loads(assembler)
    origin = assembler.evaluate_elastic(np.zeros(layout.size), loads).residual
    reduced_offset = origin[:nx] - L[:nx, nx:] @ (inverse @ origin[nx:])
    boundary = np.zeros((layout.dim, grid.num_faces))
    boundary[:, layout.dirichlet_faces] = loads.dirichlet_increment
    boundary[:, assembler.neumann_faces] = loads.neumann_increment
    expected = -assembler.elastic.boundary_rhs(boundary.ravel(order="F")) - loads.cell_source_increment
    np.testing.assert_allclose(reduced_offset, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3], ids=["2d", "3d"])
@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann", "mixed"])
def test_single_cell_boundary_partitions(dim: int, boundary_type: str) -> None:
    grid = pp.CartGrid(np.ones(dim, dtype=int))
    grid.compute_geometry()
    assembler = _assembler(grid, boundary_type)
    jacobian = AnalyticalJacobian(assembler)
    layout = assembler.layout
    z = np.random.default_rng(50).normal(size=layout.size)
    loads = _loads(assembler)
    origin = assembler.evaluate_elastic(np.zeros(layout.size), loads)
    actual = assembler.evaluate_elastic(z, loads)
    np.testing.assert_allclose(jacobian.L @ z + origin.residual, actual.residual, atol=1e-12)
    assert jacobian.H.shape == (layout.size, layout.size)
    if boundary_type == "dirichlet":
        assert layout.free_faces.size == 0
        assert jacobian.B_z.nnz == 0
        np.testing.assert_array_equal(jacobian.H.toarray(), -assembler.accumulation.toarray())
    elif boundary_type == "neumann":
        assert loads.dirichlet_increment.shape == (dim, 0)


def test_reversing_face_orientation_preserves_fixed_matrices(jacobian: AnalyticalJacobian) -> None:
    assembler = jacobian.assembler
    grid = jacobian.operators.grid
    signs = np.ones(grid.num_faces)
    signs[::2] = -1.0
    grid.face_normals *= signs
    grid.cell_faces = sps.csc_matrix(grid.cell_faces.multiply(signs[:, None]))
    other = AnalyticalJacobian(ResidualAssembler(grid, assembler.elastic.bc, assembler.material_points))
    for name in ("B_z", "T", "S", "H", "L", "P_side", "P"):
        np.testing.assert_allclose(getattr(other, name).toarray(), getattr(jacobian, name).toarray(), atol=1e-12)


def test_construction_reuses_cached_operators_and_preserves_them(
    jacobian: AnalyticalJacobian, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembler, operators = jacobian.assembler, jacobian.operators
    saved_B, saved_accum = operators.B_free.copy(), assembler.accumulation.copy()
    operators.B_free.data.setflags(write=False)
    assembler.accumulation.data.setflags(write=False)

    def unexpected_rebuild(*args: object, **kwargs: object) -> None:
        pytest.fail("Fixed Jacobian assembly must reuse operators without local material updates.")

    for name in ("_build_strain_matrix", "_build_side_geometry", "_build_traction_coefficients"):
        monkeypatch.setattr(operators, name, unexpected_rebuild)
    monkeypatch.setattr(assembler.elastic, "_assemble_matrices", unexpected_rebuild)
    monkeypatch.setattr(MaterialPoint, "update", unexpected_rebuild)
    rebuilt = AnalyticalJacobian(assembler)
    np.testing.assert_array_equal(operators.B_free.toarray(), saved_B.toarray())
    np.testing.assert_array_equal(assembler.accumulation.toarray(), saved_accum.toarray())
    for name in ("B_z", "T", "S", "H", "L", "P_side", "P"):
        np.testing.assert_array_equal(getattr(rebuilt, name).toarray(), getattr(jacobian, name).toarray())


@pytest.mark.parametrize("missing", ["side_stiffness", "rotation_transfer"])
def test_missing_traction_coefficients_are_rejected(grid: pp.Grid, missing: str) -> None:
    assembler = _assembler(grid)
    setattr(assembler.operators, missing, None)
    with pytest.raises(ValueError, match="Traction coefficients"):
        AnalyticalJacobian(assembler)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_auxiliary_matrix_is_rejected(grid: pp.Grid, value: float) -> None:
    assembler = _assembler(grid)
    assembler.accumulation = assembler.accumulation.copy()
    assembler.accumulation.data[0] = value
    with pytest.raises(ValueError, match="Auxiliary matrix.*finite"):
        AnalyticalJacobian(assembler)


def _material_trial(
    jacobian: AnalyticalJacobian, regime: str,
) -> tuple[MaterialHistory, MaterialTrial]:
    layout, points = jacobian.layout, jacobian.assembler.material_points
    nd, nc = layout.dim, layout.num_cells
    old = MaterialHistory.zeros(layout)
    scale = 1.0 + 0.1 * np.arange(nc)
    if regime in ("reloading", "unloading"):
        preload = np.diag([0.02, -0.005, -0.003 if nd == 3 else 0.0])[:, :, None] * scale
        previous = evaluate_material_at_strain(layout, points, old, preload, dim=nd)
        old = MaterialHistory(nd, previous.epsilon, previous.material_states)

    direction = np.array([[1.0, -0.2, 0.25], [-0.2, 0.1, 0.15], [0.25, 0.15, -0.1]])
    if nd == 2:
        direction[2, :] = direction[:, 2] = 0.0
    increment = 0.02 * direction[:, :, None] * scale
    if regime == "elastic":
        increment *= 0.001
    elif regime == "mixed":
        increment[:, :, ::2] *= 0.001
    elif regime == "unloading":
        increment = -0.01 * old.epsilon
    trial = evaluate_material_at_strain(layout, points, old, old.epsilon + increment, dim=nd)
    return old, trial


def _hooke(point: MaterialPoint, increment: FloatArray) -> FloatArray:
    material = point.material
    return np.asarray(
        material.lame_parameter * np.trace(increment) * np.eye(3)
        + 2 * material.isotropic_shear_modulus * increment,
        dtype=np.float64,
    )


def _assert_material_state(actual: MaterialState, expected: MaterialState) -> None:
    assert actual.alpha == expected.alpha
    for name in ("stress", "plastic_strain", "backstress"):
        np.testing.assert_array_equal(getattr(actual, name).to_numpy(), getattr(expected, name).to_numpy())


def test_stress_correction_maps_match_independent_surface_sums(jacobian: AnalyticalJacobian) -> None:
    layout, operators = jacobian.layout, jacobian.operators
    grid, nd, nc = operators.grid, layout.dim, layout.num_cells
    correction = np.random.default_rng(51).normal(size=(3, 3, nc))
    correction = 0.5 * (correction + correction.swapaxes(0, 1))
    side_values = np.empty((nd, operators.num_sides))
    momentum = np.zeros((nd, nc))
    faces = np.zeros((nd, grid.num_faces))
    for side, (face, cell) in enumerate(zip(operators.side_faces, operators.side_cells, strict=True)):
        # PorePy normals contain area; incidence supplies the outward orientation.
        normal = float(cast(float, grid.cell_faces[face, cell])) * grid.face_normals[:nd, face]
        side_values[:, side] = -correction[:nd, :nd, cell] @ normal
        momentum[:, cell] += side_values[:, side]
        faces[:, face] += side_values[:, side]
    expected = np.zeros(layout.size)
    expected[layout.u_slice] = momentum.ravel(order="F")
    expected[layout.face_slice] = faces[:, layout.free_faces].ravel(order="F")
    q = mandel(correction, dim=nd)
    np.testing.assert_allclose(jacobian.P_side @ q, side_values.ravel(order="F"), atol=1e-13)
    np.testing.assert_allclose(jacobian.P @ q, expected, atol=1e-13)
    assert jacobian.P[layout.r_slice, :].nnz == 0
    assert jacobian.P[layout.p_slice, :].nnz == 0

    # A uniform stress has zero cell divergence and no interior traction jump.
    uniform = mandel(np.repeat(correction[:, :, :1], nc, axis=2), dim=nd)
    balanced = jacobian.P @ uniform
    np.testing.assert_allclose(balanced[layout.u_slice], 0.0, atol=1e-13)
    interior = np.setdiff1d(np.arange(grid.num_faces), grid.get_all_boundary_faces())
    np.testing.assert_allclose(balanced[layout.face_dofs[:, interior]], 0.0, atol=1e-13)


def test_stress_maps_reproduce_nonlinear_residual_correction(jacobian: AnalyticalJacobian) -> None:
    assembler, layout = jacobian.assembler, jacobian.layout
    old, _ = _material_trial(jacobian, "reloading")
    z = 0.02 * np.random.default_rng(52).normal(size=layout.size)
    loads = _loads(assembler)
    loads.dirichlet_increment[:] *= 0.02
    elastic = assembler.evaluate_elastic(z, loads)
    actual = assembler.evaluate(z, loads, old)
    assert actual.material_trial is not None
    q = mandel(actual.material_trial.stress_correction, dim=layout.dim)
    assert np.linalg.norm(q) > 1e-3
    np.testing.assert_allclose(
        actual.traction_increment.ravel(order="F"),
        elastic.traction_increment.ravel(order="F") + jacobian.P_side @ q, atol=1e-12,
    )
    np.testing.assert_allclose(actual.residual, elastic.residual + jacobian.P @ q, atol=1e-12)


@pytest.mark.parametrize("regime", ["elastic", "plastic", "mixed", "reloading", "unloading"])
def test_material_blocks_match_full_tensor_actions(grid: pp.Grid, regime: str) -> None:
    jacobian = AnalyticalJacobian(_assembler(grid))
    layout = jacobian.layout
    nd, nc = layout.dim, layout.num_cells
    m = 3 if nd == 2 else 6
    old, trial = _material_trial(jacobian, regime)
    active = np.array([new.alpha > previous.alpha for new, previous in zip(
        trial.material_states, old.material_states, strict=True,
    )])
    expected_active = (
        np.zeros(nc, dtype=bool) if regime in ("elastic", "unloading")
        else np.arange(nc) % 2 == 1 if regime == "mixed" else np.ones(nc, dtype=bool)
    )
    np.testing.assert_array_equal(active, expected_active)
    if regime in ("reloading", "unloading"):
        assert all(state.alpha > 0 for state in old.material_states)

    matrix = jacobian._build_material_correction_tangent(old, trial)
    assert isinstance(matrix, sps.csr_array)
    assert matrix.shape == (m * nc, m * nc)
    assert np.all(np.isfinite(matrix.data))
    entries = matrix.tocoo()
    np.testing.assert_array_equal(entries.row // m, entries.col // m)
    np.testing.assert_allclose(matrix.toarray(), matrix.T.toarray(), atol=1e-12)

    for cell, point in enumerate(jacobian.assembler.material_points):
        block = matrix[m * cell:m * (cell + 1), m * cell:m * (cell + 1)]
        if active[cell]:
            assert np.linalg.norm(block.data) > 1e-3
        else:
            assert block.nnz == 0
        # The full 3D local tangent is independently tested in test/material.
        full = analytical_material_tangent(
            point, old.material_states[cell], trial.epsilon[:, :, cell] - old.epsilon[:, :, cell],
            dim=3, returned=trial.material_states[cell],
        )
        for i in range(nd):
            for j in range(i, nd):
                variation = np.zeros((3, 3, nc))
                variation[i, j, cell] = variation[j, i, cell] = 1.0 if i == j else 1 / np.sqrt(2)
                elastic = _hooke(point, variation[:, :, cell])
                expected = np.zeros_like(variation)
                expected[:, :, cell] = elastic - np.einsum("ijkl,kl->ij", full, variation[:, :, cell])
                np.testing.assert_allclose(
                    matrix @ mandel(variation, dim=nd), mandel(expected, dim=nd), atol=2e-12,
                )
                # Check the cached elastic projection using Hooke's law directly.
                np.testing.assert_allclose(
                    jacobian._elastic_mandel[cell] @ mandel(variation[:, :, cell:cell + 1], dim=nd),
                    mandel(elastic[:, :, None], dim=nd), atol=2e-12,
                )
    if not np.any(active):
        assert matrix.nnz == 0
        np.testing.assert_array_equal(
            (jacobian.L + jacobian.P @ matrix @ jacobian.B_z).toarray(), jacobian.L.toarray(),
        )


def test_material_tangent_reuses_trial_and_preserves_history_and_cache(
    grid: pp.Grid, monkeypatch: pytest.MonkeyPatch,
) -> None:
    jacobian = AnalyticalJacobian(_assembler(grid))
    old, trial = _material_trial(jacobian, "reloading")
    expected = jacobian._build_material_correction_tangent(old, trial)
    unloading = evaluate_material_at_strain(
        jacobian.layout, jacobian.assembler.material_points, old, 0.99 * old.epsilon, dim=grid.dim,
    )
    old_before = old.copy()
    trial_before = [state.copy() for state in trial.material_states]
    epsilon_before, correction_before = trial.epsilon.copy(), trial.stress_correction.copy()
    basis_before = jacobian._mandel_basis.copy()
    elastic_before = tuple(block.copy() for block in jacobian._elastic_mandel)
    for array in (old.epsilon, trial.epsilon, trial.stress_correction, jacobian._mandel_basis,
                  *jacobian._elastic_mandel):
        array.setflags(write=False)

    # Material-point slots deliberately differ from the supplied global history.
    points = jacobian.assembler.material_points
    for cell, point in enumerate(points):
        point.update(strain(0.25 * old.epsilon[:, :, cell]))
        point.commit()
        point.update(strain(0.03 * old.epsilon[:, :, cell]))
    point_snapshots = []
    for point in points:
        assert point.trial is not None
        point_snapshots.append((point.committed, point.trial, point.committed.copy(), point.trial.copy()))

    def unexpected_update(*args: object, **kwargs: object) -> None:
        pytest.fail("Material tangent assembly must reuse the matching returned state.")

    monkeypatch.setattr(MaterialPoint, "update", unexpected_update)
    actual = jacobian._build_material_correction_tangent(old, trial)
    assert jacobian._build_material_correction_tangent(old, unloading).nnz == 0
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    np.testing.assert_array_equal(jacobian._mandel_basis, basis_before)
    for cached, saved in zip(jacobian._elastic_mandel, elastic_before, strict=True):
        np.testing.assert_array_equal(cached, saved)
    np.testing.assert_array_equal(old.epsilon, old_before.epsilon)
    np.testing.assert_array_equal(trial.epsilon, epsilon_before)
    np.testing.assert_array_equal(trial.stress_correction, correction_before)
    for current, saved in zip(old.material_states, old_before.material_states, strict=True):
        _assert_material_state(current, saved)
    for current, saved in zip(trial.material_states, trial_before, strict=True):
        _assert_material_state(current, saved)
    for point, (committed_slot, trial_slot, saved_committed, saved_trial) in zip(points, point_snapshots, strict=True):
        assert point.committed is committed_slot and point.trial is trial_slot
        _assert_material_state(point.committed, saved_committed)
        _assert_material_state(point.trial, saved_trial)


@pytest.mark.parametrize("invalid", ["dimension", "committed_shape", "trial_shape", "committed_count", "trial_count"])
def test_material_tangent_rejects_incompatible_history(grid: pp.Grid, invalid: str) -> None:
    jacobian = AnalyticalJacobian(_assembler(grid))
    old, trial = _material_trial(jacobian, "elastic")
    if invalid == "dimension":
        old.dim = 3 if grid.dim == 2 else 2
        message = "dimension"
    elif invalid.endswith("shape"):
        if invalid.startswith("committed"):
            old.epsilon = old.epsilon[:, :, :-1]
        else:
            trial.epsilon = trial.epsilon[:, :, :-1]
        message = "shape"
    else:
        if invalid.startswith("committed"):
            old.material_states.pop()
        else:
            trial.material_states.pop()
        message = "one committed and trial material state per cell"
    with pytest.raises(ValueError, match=message):
        jacobian._build_material_correction_tangent(old, trial)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_material_correction_matrix_is_rejected(grid: pp.Grid, value: float) -> None:
    jacobian = AnalyticalJacobian(_assembler(grid))
    old, trial = _material_trial(jacobian, "elastic")
    jacobian._elastic_mandel[0][0, 0] = value
    with pytest.raises(ValueError, match="Material correction tangent.*finite"):
        jacobian._build_material_correction_tangent(old, trial)


@pytest.mark.parametrize("dim", [2, 3])
def test_single_dirichlet_cell_has_no_material_feedback_to_unknowns(dim: int) -> None:
    grid = pp.CartGrid(np.ones(dim, dtype=int))
    grid.compute_geometry()
    assembler = _assembler(grid, "dirichlet")
    jacobian, old = AnalyticalJacobian(assembler), MaterialHistory.zeros(assembler.layout)
    gradient = np.diag([0.02, -0.005, -0.003])[:dim, :dim]
    loads = LoadIncrement(
        gradient @ grid.face_centers[:dim, assembler.layout.dirichlet_faces],
        np.zeros((dim, 0)), np.zeros(assembler.layout.cell_size),
    )
    evaluation = assembler.evaluate(np.zeros(assembler.layout.size), loads, old)
    assert evaluation.material_trial is not None
    assert evaluation.material_trial.material_states[0].alpha > 0
    matrix = jacobian._build_material_correction_tangent(old, evaluation.material_trial)
    assert matrix.nnz > 0
    assert jacobian.B_z.nnz == 0
    assert (jacobian.P @ matrix @ jacobian.B_z).nnz == 0
    np.testing.assert_array_equal(jacobian.evaluate(old, evaluation).toarray(), jacobian.L.toarray())


def _evaluated_candidate(
    jacobian: AnalyticalJacobian, regime: str,
) -> tuple[MaterialHistory, FloatArray, LoadIncrement, ResidualEvaluation]:
    assembler, layout = jacobian.assembler, jacobian.layout
    grid, nd = jacobian.operators.grid, layout.dim
    old = MaterialHistory.zeros(layout)
    gradient = np.array([[0.02, 0.004, -0.003], [0.004, -0.005, 0.002], [-0.003, 0.002, -0.001]])[:nd, :nd]
    if regime == "elastic":
        gradient *= 0.001
    elif regime == "mixed":
        for point in assembler.material_points[::2]:
            point.parameters.sigma_y = point.parameters.sigma_u = 100.0
    elif regime in ("reloading", "unloading"):
        old, _ = _material_trial(jacobian, "reloading")
        if regime == "unloading":
            gradient = -0.01 * np.diag([0.02, -0.005, -0.003])[:nd, :nd]

    z = 0.01 * np.random.default_rng(53).normal(size=layout.size)
    faces = gradient @ grid.face_centers[:nd]
    z[layout.face_slice] = faces[:, layout.free_faces].ravel(order="F")
    loads = _loads(assembler)
    loads.dirichlet_increment[:] = faces[:, layout.dirichlet_faces]
    return old, z, loads, assembler.evaluate(z, loads, old)


def _reference_jacobian_action(
    jacobian: AnalyticalJacobian, old: MaterialHistory,
    evaluation: ResidualEvaluation, direction: FloatArray,
) -> FloatArray:
    """Add direct tensor/traction variations to the independently tested elastic map."""
    layout, grid = jacobian.layout, jacobian.operators.grid
    nd, nc = layout.dim, layout.num_cells
    trial = evaluation.material_trial
    assert trial is not None
    face_variation = np.zeros((nd, grid.num_faces))
    face_variation[:, layout.free_faces] = direction[layout.face_slice].reshape((nd, -1), order="F")
    gradient = np.zeros((3, 3, nc))
    incidence = grid.cell_faces.tocoo()
    for face, side_cell, sign in zip(incidence.row, incidence.col, incidence.data, strict=True):
        area_normal = sign * grid.face_normals[:nd, face]
        gradient[:nd, :nd, side_cell] += (
            np.outer(face_variation[:, face], area_normal) / grid.cell_volumes[side_cell]
        )
    strain_variation = 0.5 * (gradient + gradient.swapaxes(0, 1))
    correction = np.empty((3, 3, nc))
    for cell, point in enumerate(jacobian.assembler.material_points):
        tangent = analytical_material_tangent(
            point, old.material_states[cell], trial.epsilon[:, :, cell] - old.epsilon[:, :, cell],
            dim=3, returned=trial.material_states[cell],
        )
        correction[:, :, cell] = _hooke(point, strain_variation[:, :, cell]) - np.einsum(
            "ijkl,kl->ij", tangent, strain_variation[:, :, cell],
        )

    expected = np.asarray(jacobian.L @ direction, dtype=np.float64).copy()
    for face, side_cell, sign in zip(incidence.row, incidence.col, incidence.data, strict=True):
        traction = -correction[:nd, :nd, side_cell] @ (sign * grid.face_normals[:nd, face])
        expected[nd * side_cell:nd * (side_cell + 1)] += traction
        rows = layout.face_dofs[:, face]
        free = rows >= 0
        expected[rows[free]] += traction[free]
    return expected


@pytest.mark.parametrize("regime", ["elastic", "plastic", "mixed", "reloading", "unloading"])
def test_evaluate_matches_tensor_and_surface_actions(jacobian: AnalyticalJacobian, regime: str) -> None:
    old, _, _, evaluation = _evaluated_candidate(jacobian, regime)
    trial, layout = evaluation.material_trial, jacobian.layout
    assert trial is not None
    active = np.array([new.alpha > previous.alpha for new, previous in zip(
        trial.material_states, old.material_states, strict=True,
    )])
    expected_active = (
        np.zeros(layout.num_cells, dtype=bool) if regime in ("elastic", "unloading")
        else np.arange(layout.num_cells) % 2 == 1 if regime == "mixed"
        else np.ones(layout.num_cells, dtype=bool)
    )
    np.testing.assert_array_equal(active, expected_active)
    if regime in ("reloading", "unloading"):
        assert all(state.alpha > 0 for state in old.material_states)

    actual = jacobian.evaluate(old, evaluation)
    assert isinstance(actual, sps.csr_array)
    assert actual.shape == (layout.size, layout.size)
    assert np.all(np.isfinite(actual.data))
    assert np.all(actual.data != 0.0)
    if np.any(active):
        assert np.linalg.norm((actual - jacobian.L).data) > 1e-3
    else:
        np.testing.assert_array_equal(actual.toarray(), jacobian.L.toarray())
    # Material corrections act only on face columns and momentum/face rows.
    np.testing.assert_array_equal(actual[:, layout.cell_slice].toarray(), jacobian.L[:, layout.cell_slice].toarray())
    for block in (layout.r_slice, layout.p_slice):
        np.testing.assert_array_equal(actual[block, :].toarray(), jacobian.L[block, :].toarray())
    direction = np.random.default_rng(54).normal(size=layout.size)
    for face_only in (False, True):
        if face_only:
            direction[layout.cell_slice] = 0.0
        expected = _reference_jacobian_action(jacobian, old, evaluation, direction)
        np.testing.assert_allclose(actual @ direction, expected, rtol=1e-12, atol=2e-11)


def test_evaluate_rejects_missing_material_trial(grid: pp.Grid) -> None:
    assembler = _assembler(grid)
    jacobian, old = AnalyticalJacobian(assembler), MaterialHistory.zeros(assembler.layout)
    evaluation = assembler.evaluate_elastic(np.zeros(assembler.layout.size), _loads(assembler))
    assert evaluation.material_trial is None
    with pytest.raises(ValueError, match="requires a material trial"):
        jacobian.evaluate(old, evaluation)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_evaluate_rejects_nonfinite_assembled_jacobian(grid: pp.Grid, value: float) -> None:
    jacobian = AnalyticalJacobian(_assembler(grid))
    old, _, _, evaluation = _evaluated_candidate(jacobian, "elastic")
    # Keep the material tangent valid to exercise the final assembly guard.
    jacobian.L.data[0] = value
    with pytest.raises(ValueError, match="Analytical Jacobian must remain finite"):
        jacobian.evaluate(old, evaluation)


@pytest.mark.parametrize("regime", ["elastic", "reloading"])
def test_evaluate_reuses_material_trial_and_returns_owned_matrix(
    grid: pp.Grid, regime: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    jacobian = AnalyticalJacobian(_assembler(grid))
    assembler = jacobian.assembler
    old, z, loads, evaluation = _evaluated_candidate(jacobian, regime)
    elastic_old, _, _, elastic_evaluation = _evaluated_candidate(jacobian, "elastic")
    trial = evaluation.material_trial
    assert trial is not None
    old_before, saved_evaluation = old.copy(), deepcopy(evaluation)
    saved_trial = saved_evaluation.material_trial
    assert saved_trial is not None
    saved_z, saved_loads = z.copy(), deepcopy(loads)
    caches = {name: getattr(jacobian, name) for name in ("L", "P", "P_side", "B_z")}
    saved_caches = {name: matrix.copy() for name, matrix in caches.items()}
    for matrix in caches.values():
        for storage in (matrix.data, matrix.indices, matrix.indptr):
            storage.setflags(write=False)
    for values in (
        z, old.epsilon, evaluation.residual, evaluation.face_increment, evaluation.traction_increment,
        trial.epsilon, trial.stress_correction, loads.dirichlet_increment,
        loads.neumann_increment, loads.cell_source_increment,
    ):
        values.setflags(write=False)

    # Existing MaterialPoint slots are unrelated to the supplied global history.
    for cell, point in enumerate(assembler.material_points):
        point.update(strain(0.25 * trial.epsilon[:, :, cell]))
        point.commit()
        point.update(strain(0.03 * trial.epsilon[:, :, cell]))
    saved_points = deepcopy(assembler.material_points)
    slots = [(point.committed, point.trial) for point in assembler.material_points]

    def unexpected_evaluation(*args: object, **kwargs: object) -> None:
        pytest.fail("Public Jacobian evaluation must reuse the existing material trial.")

    monkeypatch.setattr(MaterialPoint, "update", unexpected_evaluation)
    monkeypatch.setattr(assembler, "evaluate", unexpected_evaluation)
    monkeypatch.setattr(jacobian.operators, "green_gauss_reconstruction", unexpected_evaluation)
    first = jacobian.evaluate(old, evaluation)
    expected = first.toarray().copy()
    # The same Jacobian object must handle a new history/candidate without stale tangents.
    np.testing.assert_array_equal(jacobian.evaluate(elastic_old, elastic_evaluation).toarray(), jacobian.L.toarray())
    for matrix in caches.values():
        for field in ("data", "indices", "indptr"):
            assert not np.shares_memory(getattr(first, field), getattr(matrix, field))
    first.data[:] = 0.0
    np.testing.assert_array_equal(jacobian.evaluate(old, evaluation).toarray(), expected)

    np.testing.assert_array_equal(z, saved_z)
    np.testing.assert_array_equal(old.epsilon, old_before.epsilon)
    for name in ("dirichlet_increment", "neumann_increment", "cell_source_increment"):
        np.testing.assert_array_equal(getattr(loads, name), getattr(saved_loads, name))
    for name in ("residual", "face_increment", "traction_increment"):
        np.testing.assert_array_equal(getattr(evaluation, name), getattr(saved_evaluation, name))
    for name in ("epsilon", "stress_correction"):
        np.testing.assert_array_equal(getattr(trial, name), getattr(saved_trial, name))
    for current, saved in zip(old.material_states, old_before.material_states, strict=True):
        _assert_material_state(current, saved)
    for current, saved in zip(trial.material_states, saved_trial.material_states, strict=True):
        _assert_material_state(current, saved)
    for point, saved_point, (committed_slot, trial_slot) in zip(assembler.material_points, saved_points, slots, strict=True):
        assert point.committed is committed_slot and point.trial is trial_slot
        assert point.trial is not None and saved_point.trial is not None
        _assert_material_state(point.committed, saved_point.committed)
        _assert_material_state(point.trial, saved_point.trial)
    for name, matrix in caches.items():
        assert getattr(jacobian, name) is matrix
        for field in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(matrix, field), getattr(saved_caches[name], field))
