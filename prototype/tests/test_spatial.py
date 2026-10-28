"""Independent 2D/3D patch, TPSA, tangent and load-history checks."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
import porepy as pp
import pytest
import scipy.sparse as sps
from porepy.grids.grid import Grid
from porepy.params.bc import BoundaryConditionVectorial

from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint, MaterialState
from material.scalar_hardening import HardeningParameters, isotropic_hardening_K, kinematic_hardening_H
from material.tensor import strain
from prototype import loading
from prototype.jacobian import RetainedAnalyticalJacobian, RetainedFiniteDifferenceJacobian
from prototype.loading import LoadStepController
from prototype.newton import BacktrackingOptions, solve_newton
from prototype.operators import RetainedFaceOperators
from prototype.residual import assemble_global_residual, evaluate_global_residual, evaluate_one_sided_traction
from prototype.runner import prepare_retained_solver
from prototype.state import RetainedState, assemble_state_residual
from prototype.tpsa_state import TpsaState
from utils.load_schedule import AdaptiveLoadOptions, LoadStepError
from utils.newton_utils import NewtonConvergenceError
from utils.reconstruction import mandel
from utils.state import TpsaState as LegacyState, as_tpsa_state

FloatArray = NDArray[np.float64]
ATOL = (1e-10, 1e-12, 1e-12, 1e-10)


@dataclass
class Problem:
    grid: Grid
    bc: BoundaryConditionVectorial
    material_points: list[MaterialPoint]
    bc_values: FloatArray


def setup(dim: int = 3, *, single: bool = False, heterogeneous: bool = False,
          rotated: bool = True) -> tuple[Problem, RetainedFaceOperators]:
    axes = [np.array([0., 1.])] * 3 if single else [
        np.array([0., .3, 1.]), np.array([0., .4, 1.3]), np.array([0., .2, .8]),
    ]
    grid = pp.TensorGrid(axes[0], axes[1], axes[2] if dim == 3 else None)
    if rotated:
        c, s = np.cos(.37), np.sin(.37)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        if dim == 3:
            rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]]) @ rotation
        grid.nodes[:] = rotation @ grid.nodes
    grid.compute_geometry()
    bc = pp.BoundaryConditionVectorial(grid, grid.get_all_boundary_faces(), "dir")
    points = []
    for cell in range(grid.num_cells):
        material = _materialProperty()
        material.young_modulus = 12. + (cell if heterogeneous else 0.)
        material.poisson_ratio = .25
        points.append(MaterialPoint(material, HardeningParameters(.05, .05, 1., .4, 0.),
                                    vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H))
    ops = RetainedFaceOperators(grid, bc,
        mu=np.array([p.material.isotropic_shear_modulus for p in points]),
        lam=np.array([p.material.lame_parameter for p in points]))
    return Problem(grid, bc, points, np.zeros((dim, grid.num_faces))), ops


def gradient(dim: int, scale: float) -> FloatArray:
    return scale * np.array([[1., .3, -.2], [-.1, -.2, .4], [.15, -.25, .5]])[:dim, :dim]


def affine(ops: RetainedFaceOperators, g: FloatArray,
           shift: FloatArray | None = None) -> tuple[FloatArray, FloatArray]:
    assert ops.mu is not None and ops.lam is not None
    nd = ops.dim
    shift = np.zeros(nd) if shift is None else shift
    u = g @ ops.grid.cell_centers[:nd] + shift[:, None]
    d = g @ ops.grid.face_centers[:nd] + shift[:, None]
    spin = np.array([g[0, 1] - g[1, 0]]) if nd == 2 else np.array([
        g[1, 2] - g[2, 1], g[2, 0] - g[0, 2], g[0, 1] - g[1, 0],
    ])
    x = np.r_[u.ravel("F"), (spin[:, None] * ops.mu).ravel("F"), ops.lam * np.trace(g)]
    return ops.pack(x, d[:, ops.free_faces]), d


def assert_same_state(actual: RetainedState, expected: RetainedState) -> None:
    for name in ("side_traction", "body_force"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in ("x", "bc_values", "epsilon", "traction", "face_displacement"):
        np.testing.assert_array_equal(getattr(actual.tpsa, name), getattr(expected.tpsa, name))
    for a, b in zip(actual.tpsa.material_states, expected.tpsa.material_states, strict=True):
        assert a.alpha == b.alpha
        for name in ("stress", "plastic_strain", "backstress"):
            np.testing.assert_array_equal(getattr(a, name).to_numpy(), getattr(b, name).to_numpy())


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("field", ["translation", "rotation", "strain"])
def test_affine_patch_and_rotation_conventions(dim: int, single: bool, field: str) -> None:
    problem, ops = setup(dim, single=single)
    g = gradient(dim, .001)
    if field == "translation":
        g[:] = 0.
    elif field == "rotation":
        g = g - g.T
    z, d = affine(ops, g, np.arange(dim, dtype=np.float64) * .003)
    zero = np.zeros_like(d)
    epsilon = np.zeros((3, 3, ops.num_cells))
    epsilon[:dim, :dim] = ((g + g.T) / 2)[:, :, None]
    np.testing.assert_allclose(ops.strain_increment(z, d, zero), epsilon, atol=2e-17)
    t = evaluate_one_sided_traction(ops, z, d, zero, np.zeros_like(epsilon))
    stress = problem.material_points[0].material.elastic_tensor.double_contract(strain(epsilon[:, :, 0])).to_numpy()
    np.testing.assert_allclose(t, (stress[:dim, :dim] @ ops.side_normals) * ops.side_areas, atol=3e-16)
    np.testing.assert_allclose(assemble_global_residual(ops, z, d, zero, t), 0., atol=5e-16)
    nc = ops.num_cells
    assert ops.num_dofs == ((4 if dim == 2 else 7) * nc + dim * ops.num_free_faces)
    x, faces = ops.unpack(z)
    np.testing.assert_array_equal(ops.pack(x, faces), z)
    x[:] = 0.
    assert not np.shares_memory(faces, z)


@pytest.mark.parametrize("dim", [2, 3])
def test_state_ownership_and_legacy_compatibility(dim: int) -> None:
    _, ops = setup(dim)
    state = RetainedState.zeros(ops)
    assert isinstance(state.tpsa, LegacyState)
    assert as_tpsa_state(state) is state.tpsa
    assert state.tpsa.dim == dim
    saved = state.copy()
    snapshot = state.copy()
    snapshot.tpsa.x[:] = 1.
    snapshot.tpsa.material_states[0].alpha = 1.
    snapshot.side_traction[:] = 1.
    assert_same_state(state, saved)
    if dim == 3:
        eps = np.zeros((3, 3, ops.num_cells))
        eps[2, 2] = .01
        total = TpsaState(state.tpsa.x, state.tpsa.bc_values, eps,
                          state.tpsa.material_states, state.tpsa.traction)
        eps[:] = 0.
        np.testing.assert_array_equal(total.epsilon[2, 2], .01)
        with pytest.raises(ValueError, match="plane strain"):
            LegacyState(np.zeros(4 * ops.num_cells), np.zeros((2, ops.num_faces)),
                        total.epsilon, total.material_states, np.zeros((2, ops.num_faces)))


@pytest.mark.parametrize("defect", ["x", "face_displacement", "epsilon", "complex", "alpha"])
def test_3d_state_rejects_bad_histories(defect: str) -> None:
    problem, _ = setup()
    state = TpsaState.zeros(problem.grid, store_faces=True)
    values: dict[str, Any] = {name: deepcopy(getattr(state, name)) for name in (
        "x", "bc_values", "epsilon", "material_states", "traction", "face_displacement")}
    if defect == "x":
        values["x"] = values["x"][:-1]
    elif defect == "face_displacement":
        values[defect] = values[defect][:2]
    elif defect == "epsilon":
        values[defect][0, 2] = 1.
    elif defect == "complex":
        values["bc_values"] = values["bc_values"].astype(complex) + 1j
    else:
        values["material_states"][0].alpha = -1.
    with pytest.raises(ValueError):
        TpsaState(**values)


@pytest.mark.parametrize("rotated", [False, True])
def test_3d_elastic_schur_and_solution_match_porepy(rotated: bool) -> None:
    problem, ops = setup(heterogeneous=True, rotated=rotated)
    g, nc, nf = problem.grid, ops.num_cells, ops.num_faces
    assert ops.mu is not None and ops.lam is not None
    data: dict[str, Any] = {
        pp.PARAMETERS: {"mechanics": {"fourth_order_tensor": pp.FourthOrderTensor(ops.mu, ops.lam), "bc": problem.bc}},
        pp.DISCRETIZATION_MATRICES: {"mechanics": {}},
    }
    pp.Tpsa("mechanics").discretize(g, data)
    m = data[pp.DISCRETIZATION_MATRICES]["mechanics"]
    flux = sps.block_array([
        [m["stress"], m["stress_rotation"], m["stress_total_pressure"]],
        [m["rotation_displacement"], m["rotation_rotation"], sps.csr_array((3 * nf, nc))],
        [m["solid_mass_displacement"], sps.csr_array((nf, 3 * nc)), m["solid_mass_total_pressure"]],
    ], format="csr")
    div = sps.block_diag([g.divergence(dim=3), g.divergence(dim=3), g.divergence(dim=1)], format="csr")
    accum = sps.diags_array(np.r_[np.zeros(3 * nc), np.repeat(g.cell_volumes / ops.mu, 3), g.cell_volumes / ops.lam])
    matrix = sps.csr_array(div @ flux - accum)
    bound = sps.vstack([m["bound_stress"], m["bound_rotation_displacement"], m["bound_mass_displacement"]])
    old = TpsaState.zeros(g)
    jac = RetainedAnalyticalJacobian(ops, problem.material_points, old)
    L = jac.L.toarray()
    k = ops.cell_size
    schur = L[:k, :k] - L[:k, k:] @ np.linalg.solve(L[k:, k:], L[k:, :k])
    np.testing.assert_allclose(schur, matrix.toarray(), atol=5e-13, rtol=3e-13)
    rng = np.random.default_rng(42)
    boundary = np.zeros_like(old.bc_values)
    boundary[:, ops.boundary_faces] = rng.normal(scale=1e-5, size=(3, ops.boundary_faces.size))
    body = rng.normal(scale=1e-6, size=(3, nc))
    rhs = np.r_[body.ravel("F"), np.zeros(4 * nc)] - div @ bound @ boundary.ravel("F")
    reference = sps.linalg.spsolve(matrix, rhs)
    result = solve_newton(ops, problem.material_points, old, boundary,
                          body_force_increment=body, atol=ATOL)
    assert result.iterations == 1
    np.testing.assert_allclose(result.z[ops.cell_slice], reference, atol=2e-12, rtol=1e-8)


@pytest.mark.parametrize("regime", ["elastic", "plastic", "reload"])
@pytest.mark.parametrize("kind", ["analytical", "finite-difference"])
def test_3d_jacobian_actions_match_independent_residual_differences(regime: str, kind: str) -> None:
    problem, ops = setup(heterogeneous=True)
    old = TpsaState.zeros(problem.grid)
    if regime == "reload":
        pre_z, pre_bc = affine(ops, gradient(3, .03))
        pre = evaluate_global_residual(ops, problem.material_points, old, pre_z, pre_bc)
        old = TpsaState(pre_z[ops.cell_slice], pre_bc, pre.material_trial.epsilon,
                        pre.material_trial.material_states, old.traction)
    z, inc = affine(ops, gradient(3, .00001 if regime == "elastic" else .02))
    bc = old.bc_values + inc
    evaluation = evaluate_global_residual(ops, problem.material_points, old, z, bc)
    jac = (RetainedAnalyticalJacobian(ops, problem.material_points, old) if kind == "analytical"
           else RetainedFiniteDifferenceJacobian(ops, problem.material_points, old, step=1e-8))
    J, JT = jac(z, evaluation), jac.face_jacobian(z, evaluation)
    assert jac.B.shape == (6 * ops.num_cells, ops.num_dofs)
    rng = np.random.default_rng(11)
    for block in (ops.u_slice, ops.r_slice, ops.p_slice, ops.face_slice):
        direction = np.zeros(ops.num_dofs)
        direction[block] = rng.normal(size=direction[block].size)
        h = 1e-7
        plus = evaluate_global_residual(ops, problem.material_points, old, z + h * direction, bc)
        minus = evaluate_global_residual(ops, problem.material_points, old, z - h * direction, bc)
        np.testing.assert_allclose(J @ direction, (plus.residual - minus.residual) / (2 * h), rtol=2e-5, atol=2e-7)
        np.testing.assert_allclose(JT @ direction,
            ((plus.traction_increment - minus.traction_increment) / (2 * h)).ravel("F"), rtol=2e-5, atol=2e-7)
        eps_plus = ops.strain_increment(z + h * direction, bc, old.bc_values)
        eps_minus = ops.strain_increment(z - h * direction, bc, old.bc_values)
        np.testing.assert_allclose(np.asarray(jac.B @ direction), mandel((eps_plus - eps_minus) / (2 * h), dim=3), atol=1e-9)
    assert all(p.trial is None and p.committed.alpha == 0. for p in problem.material_points)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("line_search", [False, True])
def test_3d_plastic_loading_unloading_and_restart(single: bool, line_search: bool) -> None:
    problem, ops = setup(single=single)
    g = gradient(3, .025)
    _, boundary = affine(ops, g)
    initial = RetainedState.zeros(ops)
    controller = LoadStepController(ops, problem.material_points, initial, boundary, atol=ATOL,
        line_search=BacktrackingOptions() if line_search else None)
    point = deepcopy(problem.material_points[0])
    previous_factor = 0.
    for factor in (.2, 1., .7, 1.3, -.3):
        record = controller.advance(factor)
        state = record.state
        inc = (factor - previous_factor) * (g + g.T) / 2
        reference, _ = point.update(strain(inc))
        point.commit()
        expected_z, expected_faces = affine(ops, factor * g)
        np.testing.assert_allclose(state.tpsa.x, expected_z[ops.cell_slice], atol=2e-10)
        assert state.tpsa.face_displacement is not None
        np.testing.assert_allclose(state.tpsa.face_displacement, expected_faces, atol=2e-11)
        for cell in state.tpsa.material_states:
            np.testing.assert_allclose(cell.stress.to_numpy(), reference.stress.to_numpy(), atol=2e-10)
            assert cell.alpha == pytest.approx(reference.alpha, abs=2e-10)
        np.testing.assert_allclose(state.side_traction,
            reference.stress.to_numpy() @ ops.side_normals * ops.side_areas, atol=2e-10)
        np.testing.assert_allclose(assemble_state_residual(ops, state), 0., atol=2e-10)
        previous_factor = factor
    assert max(p.alpha for p in controller.state.tpsa.material_states) > 0.
    assert_same_state(initial, RetainedState.zeros(ops))
    assert all(p.trial is None and p.committed.alpha == 0. for p in problem.material_points)
    restarted = LoadStepController(ops, problem.material_points, controller.state, boundary,
                                   initial_load_factor=-.3, atol=ATOL)
    a, b = controller.advance(.1).state, restarted.advance(.1).state
    assert_same_state(a, b)


def test_3d_failed_step_is_atomic_and_adaptive_cutback_reuses_history(monkeypatch: pytest.MonkeyPatch) -> None:
    problem, ops = setup()
    _, boundary = affine(ops, gradient(3, .025))
    initial = RetainedState.zeros(ops)
    controller = LoadStepController(ops, problem.material_points, initial, boundary, atol=ATOL)
    original = solve_newton

    def limited(*args: Any, **kwargs: Any) -> Any:
        old, target = args[2], args[3]
        if np.max(np.abs(target - old.bc_values)) > .55 * np.max(np.abs(boundary)):
            raise NewtonConvergenceError("test increment cap")
        return original(*args, **kwargs)

    monkeypatch.setattr(loading, "solve_newton", limited)
    with pytest.raises(LoadStepError):
        controller.advance(1.)
    assert_same_state(controller.state, initial)
    assert controller.load_factor == 0. and not controller.steps
    records = controller.run_adaptive([1., .2], options=AdaptiveLoadOptions(
        initial_increment=1., max_increment=1., min_increment=.01))
    assert controller.load_factor == .2
    assert any(not attempt.accepted for attempt in controller.attempts)
    assert records[-1].load_factor == .2
    reference = LoadStepController(ops, problem.material_points, initial, boundary, atol=ATOL)
    for record in records:
        expected = reference.advance(record.load_factor)
        assert_same_state(record.state, expected.state)
    np.testing.assert_allclose(assemble_state_residual(ops, controller.state), 0., atol=2e-10)


@pytest.mark.parametrize("kind", ["analytical", "finite-difference"])
def test_3d_runner_accepts_prepared_problem_and_preserves_input(kind: str) -> None:
    problem, ops = setup()
    _, problem.bc_values = affine(ops, gradient(3, .02))
    boundary = problem.bc_values.copy()
    prepared = prepare_retained_solver(problem, kind)
    result = prepared.controller.advance(1.)
    assert prepared.operators.dim == 3
    assert result.state.tpsa.x.size == 7 * ops.num_cells
    assert max(p.alpha for p in result.state.tpsa.material_states) > 0.
    np.testing.assert_array_equal(problem.bc_values, boundary)
    residual = assemble_state_residual(prepared.operators, result.state)
    blocks = (ops.u_slice, ops.r_slice, ops.p_slice, ops.face_slice)
    norms = np.array([np.linalg.norm(residual[block]) for block in blocks])
    assert np.all(norms <= result.newton.thresholds)


def test_nonuniform_3d_plastic_solution_agrees_between_jacobians() -> None:
    problem, ops = setup(rotated=False)
    x, y, z = problem.grid.face_centers
    boundary = np.vstack((.025 * x * (1 + .35 * y * z), .005 * y * z, -.004 * x * z))
    initial = RetainedState.zeros(ops)
    controllers = []
    for kind in (RetainedAnalyticalJacobian, RetainedFiniteDifferenceJacobian):
        def factory(old: LegacyState, cls: type[RetainedAnalyticalJacobian] = kind) -> RetainedAnalyticalJacobian:
            return cls(ops, problem.material_points, old)
        controllers.append(LoadStepController(ops, problem.material_points, initial, boundary,
                           jacobian_factory=factory, atol=ATOL, line_search=BacktrackingOptions()))
    for factor in (.3, .7, 1., .85):
        a, b = [controller.advance(factor).state for controller in controllers]
        np.testing.assert_allclose(a.tpsa.x, b.tpsa.x, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(a.tpsa.epsilon, b.tpsa.epsilon, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(a.side_traction, b.side_traction, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(assemble_state_residual(ops, a), 0., atol=2e-10)
    alpha = np.array([state.alpha for state in controllers[0].state.tpsa.material_states])
    assert np.min(alpha) > 0. and np.ptp(alpha) > 1e-4


def test_3d_face_orientation_reversal_preserves_physics() -> None:
    problem, ops = setup()
    z, boundary = affine(ops, gradient(3, .02))
    old = TpsaState.zeros(problem.grid)
    expected = evaluate_global_residual(ops, problem.material_points, old, z, boundary)
    signs = np.ones(ops.num_faces)
    signs[::2] = -1.
    problem.grid.face_normals *= signs
    problem.grid.cell_faces = sps.csc_matrix(sps.diags(signs) @ problem.grid.cell_faces)
    flipped = RetainedFaceOperators(problem.grid, problem.bc, mu=ops.mu, lam=ops.lam)
    actual = evaluate_global_residual(flipped, problem.material_points, old, z, boundary)
    np.testing.assert_allclose(actual.residual, expected.residual, atol=1e-14)
    np.testing.assert_allclose(actual.traction_increment, expected.traction_increment, atol=1e-14)
    a = RetainedAnalyticalJacobian(ops, problem.material_points, old)(z, expected)
    b = RetainedAnalyticalJacobian(flipped, problem.material_points, old)(z, actual)
    np.testing.assert_allclose(a.toarray(), b.toarray(), atol=1e-12)
