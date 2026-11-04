"""Mixed boundary elimination, physical face laws and analytical derivatives."""

import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]

from alternatives.coupled.jacobian import CoupledAnalyticalJacobian
from alternatives.coupled.residual import CoupledOperators, evaluate_coupled_residual
from alternatives.plane_strain import PlaneStrainTpsa, _assemble_matrices, _discretize_get_matrices
from test.alternatives.coupled.test_formulation import heterogeneous_case
from utils.state import TpsaState


def mixed_case(*, flip: bool = False) -> PlaneStrainTpsa:
    case = heterogeneous_case()
    g, bc = case.grid, case.bc
    bf = g.get_all_boundary_faces()
    bc.is_dir[:, bf], bc.is_neu[:, bf] = False, True
    axes = np.argmax(abs(g.face_normals[:2, bf]), axis=0)
    bc.is_dir[axes, bf], bc.is_neu[axes, bf] = True, False
    # Include a completely traction-prescribed face as well as rollers.
    bc.is_dir[:, bf[-1]], bc.is_neu[:, bf[-1]] = False, True
    if flip:
        signs = np.ones(g.num_faces); signs[::2] = -1
        g.face_normals *= signs
        g.cell_faces = sps.csc_matrix(sps.diags(signs) @ g.cell_faces)
    case.matrices = _discretize_get_matrices(g, case.d)
    case.face_discretization, case.rhs_matrix, case.div, case.accum = _assemble_matrices(case.matrices, g, case.d)
    return case


@pytest.mark.parametrize("flip", [False, True])
def test_mixed_maps_obey_one_sided_traction_law(flip: bool) -> None:
    case = mixed_case(flip=flip)
    ops = CoupledOperators(case)
    g, nc, nf = case.grid, ops.num_cells, ops.num_faces
    rng = np.random.default_rng(31)
    x = rng.normal(size=4*nc) * np.r_[np.full(2*nc, 0.002), np.full(2*nc, 1e8)]
    q = rng.normal(size=(nc, 3)) * 1e8
    data = rng.normal(size=(2, nf))
    data[ops.dirichlet_mask] *= 0.002
    data[ops.neumann_mask] *= 1e7  # Nonzero outward integrated forces.
    face = (ops.U @ x + ops.U_g @ data.ravel("F") + ops.H_f @ q.ravel()).reshape((2, nf), order="F")
    traction = (ops.T @ x + ops.T_g @ data.ravel("F") + ops.H_t @ q.ravel()).reshape((2, nf), order="F")
    np.testing.assert_allclose(face[ops.dirichlet_mask], data[ops.dirichlet_mask])
    np.testing.assert_allclose(traction[ops.neumann_mask], (data*ops.boundary_signs)[ops.neumann_mask])
    cf = sps.csr_array(g.cell_faces)
    for f in range(nf):
        cells = cf.indices[cf.indptr[f]:cf.indptr[f+1]]
        signs = cf.data[cf.indptr[f]:cf.indptr[f+1]]
        n = ops.normal[:, f]
        weights = ops.transfer.weights[[f], :][:, cells].toarray().ravel()
        r = weights @ x[2*nc+cells]
        for cell, sign in zip(cells, signs, strict=True):
            d = np.linalg.norm(g.face_centers[:2, f]-g.cell_centers[:2, cell])
            tensor = np.array([[q[cell, 0], q[cell, 2]/np.sqrt(2)],
                               [q[cell, 2]/np.sqrt(2), q[cell, 1]]])
            expected = g.face_areas[f] * (
                sign*2*ops.mu[cell]/d*(face[:, f]-x[2*cell:2*cell+2])
                + np.array([-n[1], n[0]])*r + x[3*nc+cell]*n - tensor @ n)
            np.testing.assert_allclose(traction[:, f], expected, rtol=2e-12, atol=2e-6)
    if flip:
        original = CoupledOperators(mixed_case())
        for name in ("U", "U_g", "H_f", "A", "H", "G"):
            np.testing.assert_allclose(getattr(ops, name).toarray(), getattr(original, name).toarray(), atol=1e-5)


@pytest.mark.parametrize("plastic", [False, True])
def test_mixed_global_and_reaction_jacobians(plastic: bool) -> None:
    case = mixed_case()
    ops = CoupledOperators(case)
    nc, nf = ops.num_cells, ops.num_faces
    old = TpsaState.zeros(case.grid, store_faces=True)
    rng = np.random.default_rng(42)
    z = rng.normal(size=7*nc) * np.r_[np.full(2*nc, 0.001), np.full(2*nc, 1e8), np.full(3*nc, 1e-5)]
    if plastic:
        z[4*nc:].reshape(nc, 3)[:] = [0.004, -0.0004, 0.0005]
    data = rng.normal(size=(2, nf))
    data[ops.dirichlet_mask] *= 0.002
    data[ops.neumann_mask] *= 1e7
    value = evaluate_coupled_residual(ops, ops.transfer, case.material_points, old, z, data)
    callback = CoupledAnalyticalJacobian(ops, ops.transfer, case.material_points, old)
    direction = rng.normal(size=7*nc) * np.r_[np.full(2*nc, 1e-4), np.full(2*nc, 1e7), np.full(3*nc, 1e-4)]
    h = 1e-5
    plus = evaluate_coupled_residual(ops, ops.transfer, case.material_points, old, z+h*direction, data)
    minus = evaluate_coupled_residual(ops, ops.transfer, case.material_points, old, z-h*direction, data)
    expected = (plus.residual-minus.residual)/(2*h)
    for a, b in zip(np.split(callback(z, value) @ direction, np.array([2, 3, 4])*nc),
                    np.split(expected, np.array([2, 3, 4])*nc), strict=True):
        assert np.linalg.norm(a-b) <= 2e-6*np.linalg.norm(b)+1e-12
    np.testing.assert_allclose(callback.face_jacobian(z, value) @ direction,
                              ((plus.trial.traction-minus.trial.traction)/(2*h)).ravel("F"),
                              rtol=3e-6, atol=0.2)
    np.testing.assert_array_equal(value.trial.traction[ops.neumann_mask],
                                  (data*ops.boundary_signs)[ops.neumann_mask])
    assert any(s.alpha > 0 for s in value.trial.material_states) == plastic
    assert all(p.trial is None and p.committed.alpha == 0 for p in case.material_points)
