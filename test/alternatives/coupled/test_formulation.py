"""Complementary face laws, monolithic derivatives, and accepted plastic histories."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import meshio  # type: ignore[import-untyped]
import numpy as np
import pytest
import scipy.sparse as sps  # type: ignore[import-untyped]
from numpy.typing import NDArray
from porepy.params.tensor import FourthOrderTensor
from porepy.utils.common_constants import PARAMETERS

from alternatives.coupled.jacobian import CoupledAnalyticalJacobian, CoupledFiniteDifferenceJacobian
from alternatives.coupled.residual import (
    CoupledOperators, assemble_coupled_residual, evaluate_coupled_residual,
)
from utils.load_schedule import AdaptiveLoadOptions, LoadStepError
from alternatives.loading import LoadStepController
from alternatives.newton import JacobianCallback, solve_newton
from utils.newton_utils import NewtonConvergenceError
from alternatives.plane_strain import PlaneStrainTpsa, KEYWORD, _assemble_matrices, _discretize_get_matrices
from examples.plastic_plane_strain import run_example as homogeneous_example
from alternatives.original.residual import TpsaResidual
from utils.state import TpsaState
from utils.transfer import CellToFaceTransfer
from examples.dirichlet_example import boundary_displacement, main, run_example


class StretchedCase(PlaneStrainTpsa):
    def set_geometry(self) -> None:
        super().set_geometry()
        self.grid.nodes[0] = 2 * self.grid.nodes[0] ** 1.4
        self.grid.nodes[1] = 0.6 * self.grid.nodes[1] ** 1.2
        self.grid.compute_geometry()


def heterogeneous_case() -> PlaneStrainTpsa:
    case = StretchedCase(cells_per_axis=2)
    case.prepare_simulation()
    for cell, point in enumerate(case.material_points):
        point.material = deepcopy(point.material)
        point.material.young_modulus *= 0.7 + 0.3 * cell
        point.parameters = replace(point.parameters, sigma_u=400e6, delta=80.0)
    case.d[PARAMETERS][KEYWORD]["fourth_order_tensor"] = FourthOrderTensor(
        mu=np.array([p.material.isotropic_shear_modulus for p in case.material_points]),
        lmbda=np.array([p.material.lame_parameter for p in case.material_points]),
    )
    case.matrices = _discretize_get_matrices(case.grid, case.d)
    case.face_discretization, case.rhs_matrix, case.div, case.accum = _assemble_matrices(
        case.matrices, case.grid, case.d,
    )
    return case


def test_complementary_weights_and_eliminated_one_sided_tractions() -> None:
    case = heterogeneous_case()
    ops = CoupledOperators(case)
    g, nc, nf = case.grid, ops.num_cells, ops.num_faces
    rng = np.random.default_rng(23)
    x = rng.normal(size=4 * nc) * np.r_[np.full(2 * nc, 0.002), np.full(2 * nc, 1e8)]
    q = rng.normal(size=(nc, 3)) * 1e8
    tensors = np.zeros((nc, 2, 2))
    tensors[:, 0, 0], tensors[:, 1, 1] = q[:, 0], q[:, 1]
    tensors[:, 0, 1] = tensors[:, 1, 0] = q[:, 2] / np.sqrt(2)
    bc = rng.normal(size=2 * nf) * 0.002
    face = (ops.U @ x + ops.U_g @ bc + ops.H_f @ q.ravel()).reshape(nf, 2)
    traction = (ops.T @ x + ops.T_g @ bc + ops.H_t @ q.ravel()).reshape(nf, 2)
    cf = sps.csr_array(g.cell_faces)
    for f in range(nf):
        cells = cf.indices[cf.indptr[f]:cf.indptr[f + 1]]
        signs = cf.data[cf.indptr[f]:cf.indptr[f + 1]]
        n = ops.normal[:, f]
        d = np.linalg.norm(g.face_centers[:2, f, None] - g.cell_centers[:2, cells], axis=0)
        a = 2 * ops.mu[cells] / d
        weights = ops.transfer.weights[[f], :][:, cells].toarray().ravel()
        np.testing.assert_allclose(weights, 1 - a / a.sum() if len(cells) == 2 else [1])
        rotation = weights @ x[2 * nc + cells]
        for cell, sign, stiffness in zip(cells, signs, a, strict=True):
            # Convert each cell's outward one-sided law back to the fixed face normal.
            expected = g.face_areas[f] * (
                sign * stiffness * (face[f] - x[2 * cell:2 * cell + 2])
                + np.array([-n[1], n[0]]) * rotation
                + x[3 * nc + cell] * n - tensors[cell] @ n
            )
            np.testing.assert_allclose(traction[f], expected, rtol=1e-12, atol=1e-5)
    boundary_rows = (2 * ops.boundary_faces[:, None] + np.arange(2)).ravel()
    np.testing.assert_array_equal(ops.H_f[boundary_rows].toarray(), 0)
    # Plastic face feedback enters all mechanical equations, even for traceless Q.
    assert ops.H[2 * nc:3 * nc].nnz and ops.H[3 * nc:].nnz


@pytest.mark.parametrize("regime", ["elastic", "plastic", "mixed"])
@pytest.mark.parametrize("jacobian_type", [CoupledAnalyticalJacobian, CoupledFiniteDifferenceJacobian])
def test_full_residual_and_reaction_derivatives(
    regime: str, jacobian_type: type[CoupledAnalyticalJacobian],
) -> None:
    case = heterogeneous_case()
    ops, old = CoupledOperators(case), TpsaState.zeros(case.grid, store_faces=True)
    nc, nf = ops.num_cells, ops.num_faces
    rng = np.random.default_rng(12)
    z = rng.normal(size=7 * nc) * np.r_[np.full(2 * nc, 0.001), np.full(2 * nc, 1e8), np.full(3 * nc, 1e-5)]
    eps = z[4 * nc:].reshape(nc, 3)
    if regime != "elastic":
        eps[:, :] = [0.004, -0.0004, 0.0005]
    if regime == "mixed":
        eps[::2] *= 0.01
    sources = rng.normal(size=4 * nc) * np.r_[np.full(2 * nc, 1e4), np.full(2 * nc, 1e-6)]

    def evaluate(candidate: NDArray[np.float64]) -> TpsaResidual:
        return evaluate_coupled_residual(ops, ops.transfer, case.material_points, old,
                                         candidate, case.bc_values, sources=sources)

    evaluation = evaluate(z)
    callback = jacobian_type(ops, ops.transfer, case.material_points, old)
    matrix, face_matrix = callback(z, evaluation), callback.face_jacobian(z, evaluation)
    assert matrix.shape == (7 * nc, 7 * nc) and face_matrix.shape == (2 * nf, 7 * nc)
    if regime == "elastic":
        np.testing.assert_allclose(matrix[:4 * nc, :4 * nc].toarray(), ops.A.toarray())
        np.testing.assert_allclose(matrix[:4 * nc, 4 * nc:].toarray(), 0, atol=1e3 if jacobian_type is CoupledFiniteDifferenceJacobian else 0)
    for _ in range(3):
        direction = rng.normal(size=7 * nc) * np.r_[np.full(2 * nc, 1e-4), np.full(2 * nc, 1e7), np.full(3 * nc, 1e-4)]
        h = 1e-5
        plus, minus = evaluate(z + h * direction), evaluate(z - h * direction)
        expected = (plus.residual - minus.residual) / (2 * h)
        for actual_block, expected_block in zip(
            np.split(matrix @ direction, [2 * nc, 3 * nc, 4 * nc]),
            np.split(expected, [2 * nc, 3 * nc, 4 * nc]), strict=True,
        ):
            assert np.linalg.norm(actual_block - expected_block) < 2e-6 * np.linalg.norm(expected_block) + 1e-13
        expected_face = ((plus.trial.traction - minus.trial.traction) / (2 * h)).ravel("F")
        np.testing.assert_allclose(face_matrix @ direction, expected_face, rtol=3e-6, atol=2)
    np.testing.assert_array_equal(old.face_displacement, 0)
    assert all(point.trial is None and point.committed.alpha == 0 for point in case.material_points)


def test_elastic_solution_and_strain_constraint_are_both_enforced() -> None:
    case = PlaneStrainTpsa(cells_per_axis=2, displacement_gradient=np.array([[1e-5, 2e-5], [-1e-5, 0]]))
    case.prepare_simulation()
    ops, old = CoupledOperators(case), TpsaState.zeros(case.grid, store_faces=True)
    callback = CoupledAnalyticalJacobian(ops, ops.transfer, case.material_points, old)
    result = solve_newton(ops, ops.transfer, case.material_points, old, case.bc_values, callback)
    expected = sps.linalg.spsolve(ops.A, ops.assemble_rhs(case.bc_values))
    np.testing.assert_allclose(result.trial.x, expected, rtol=1e-10, atol=1e-8)
    assert result.residual_norms.shape == (2, 4)
    np.testing.assert_allclose(result.trial.epsilon[:, :, 0], case.reference_strain().to_numpy(), atol=1e-14)
    # Mechanical equations alone would accept this elastic, zero-load candidate.
    z = ops.unknowns(old)
    z[4 * ops.num_cells:] = 1e-6
    evaluation = evaluate_coupled_residual(ops, ops.transfer, case.material_points, old, z, old.bc_values)
    np.testing.assert_allclose(evaluation.residual[:4 * ops.num_cells], 0, atol=1e-9)
    with pytest.raises(NewtonConvergenceError, match="strain"):
        solve_newton(ops, ops.transfer, case.material_points, old, old.bc_values, callback,
                     initial_x=z, max_iterations=0, atol=(1e-5, 1e-12, 1e-12, 1e-12))


@pytest.mark.parametrize("jacobian", ["analytical", "finite-difference"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_homogeneous_reference_with_both_load_controllers(jacobian: str, adaptive: bool) -> None:
    result = homogeneous_example(cells_per_axis=2, formulation="coupled", jacobian=jacobian,
                                 adaptive=AdaptiveLoadOptions() if adaptive else None)
    assert result.formulation == "coupled"
    assert isinstance(result.controller, LoadStepController)
    assert result.controller.load_factor == 0.98
    for record in result.controller.steps:
        assert record.newton.residual_norms.shape[1] == 4
        assert np.all(record.newton.residual_norms[-1] <= record.newton.thresholds)
        assert record.state.face_displacement is not None


@pytest.mark.parametrize("adaptive", [False, True])
def test_nonuniform_plasticity_jacobians_agree(adaptive: bool) -> None:
    a = run_example(cells_per_axis=3, formulation="coupled", adaptive=adaptive)
    b = run_example(cells_per_axis=3, formulation="coupled", adaptive=adaptive, jacobian="finite-difference")
    for result in (a, b):
        assert isinstance(result.controller, LoadStepController)
        assert isinstance(result.case, PlaneStrainTpsa)
        ops = CoupledOperators(result.case)
        records = result.controller.steps
        alpha = np.array([[s.alpha for s in record.state.material_states] for record in records])
        assert np.ptp(alpha[-1]) > 1e-5 and np.all(np.diff(alpha, axis=0) >= -1e-14)
        np.testing.assert_allclose(alpha[-1], alpha[-2], atol=1e-11)
        for record in records:
            residual = assemble_coupled_residual(ops, record.state, sources=record.sources)
            blocks = np.split(residual, [2 * ops.num_cells, 3 * ops.num_cells, 4 * ops.num_cells])
            assert np.all(np.array([np.linalg.norm(v) for v in blocks]) <= record.newton.thresholds)
            ops.validate_committed(record.state, result.case.material_points)
            # The pressure equation follows reconstructed divergence, including H_f Q.
            expected_p = ops.lam * np.trace(record.state.epsilon[:2, :2], axis1=0, axis2=1)
            np.testing.assert_allclose(record.state.x[3 * ops.num_cells:], expected_p, rtol=2e-7, atol=2)
    assert isinstance(a.controller, LoadStepController)
    assert isinstance(b.controller, LoadStepController)
    np.testing.assert_allclose(a.controller.state.x, b.controller.state.x, rtol=2e-6, atol=0.5)
    np.testing.assert_allclose(a.controller.state.epsilon, b.controller.state.epsilon, rtol=1e-6, atol=1e-9)


def test_restart_and_failed_step_preserve_plastic_face_history() -> None:
    case = PlaneStrainTpsa(cells_per_axis=3, boundary_displacement=boundary_displacement)
    case.prepare_simulation()
    ops, zero = CoupledOperators(case), TpsaState.zeros(case.grid, store_faces=True)

    def controller(old: TpsaState, factor: float, max_iterations: int = 20) -> LoadStepController:
        return LoadStepController(ops, ops.transfer, case.material_points, old, case.bc_values,
            initial_load_factor=factor, max_iterations=max_iterations,
            jacobian_factory=lambda state: CoupledAnalyticalJacobian(ops, ops.transfer, case.material_points, state))

    full = controller(zero, 0)
    full.run([0.1, 0.3, 0.5, 0.7])
    saved = full.state
    assert saved.face_displacement is not None
    elastic_face = (ops.U @ saved.x + ops.U_g @ saved.bc_values.ravel("F")).reshape(saved.face_displacement.shape, order="F")
    assert np.linalg.norm(saved.face_displacement - elastic_face) > 1e-5
    failed = controller(saved, 0.7, max_iterations=0)
    with pytest.raises(LoadStepError):
        failed.advance(0.9)
    np.testing.assert_array_equal(failed.state.face_displacement, saved.face_displacement)
    assert failed.load_factor == 0.7 and failed.steps == ()
    restarted = controller(failed.state, 0.7)
    for factor in (0.9, 1.0, 0.98):
        actual, expected = restarted.advance(factor), full.advance(factor)
        for field in ("x", "epsilon", "traction", "face_displacement"):
            np.testing.assert_array_equal(getattr(actual.state, field), getattr(expected.state, field))
    snapshot = restarted.state
    assert snapshot.face_displacement is not None
    snapshot.face_displacement.fill(0)
    retained_faces = restarted.state.face_displacement
    assert retained_faces is not None and np.linalg.norm(retained_faces) > 0


def test_analytical_taylor_remainders_after_plastic_preload() -> None:
    case = heterogeneous_case()
    ops, zero = CoupledOperators(case), TpsaState.zeros(case.grid, store_faces=True)
    old = solve_newton(ops, ops.transfer, case.material_points, zero, 20 * case.bc_values,
        CoupledAnalyticalJacobian(ops, ops.transfer, case.material_points, zero)).trial
    assert any(s.alpha > 0 for s in old.material_states)
    z, bc = 2 * ops.unknowns(old), 40 * case.bc_values
    evaluation = evaluate_coupled_residual(ops, ops.transfer, case.material_points, old, z, bc)
    assert all(a.alpha > b.alpha for a, b in zip(evaluation.trial.material_states, old.material_states, strict=True))
    callback = CoupledAnalyticalJacobian(ops, ops.transfer, case.material_points, old)
    nc = ops.num_cells
    rng = np.random.default_rng(38)
    direction = rng.normal(size=7 * nc) * np.r_[np.full(2 * nc, 0.002), np.full(2 * nc, 5e8), np.full(3 * nc, 0.002)]
    action = callback(z, evaluation) @ direction
    errors = []
    for h in (0.01, 0.005, 0.0025):
        perturbed = evaluate_coupled_residual(ops, ops.transfer, case.material_points, old, z + h * direction, bc)
        remainder = perturbed.residual - evaluation.residual - h * action
        errors.append([np.linalg.norm(b) for b in np.split(remainder, [2 * nc, 3 * nc, 4 * nc])])
    values = np.array(errors)
    # Every nonlinear block has O(h^2) remainder, including rotation and pressure.
    assert np.all(values[-1] > 1e-13)
    np.testing.assert_allclose(values[:-1] / values[1:], 4, rtol=0.08)


def test_adaptive_cutback_keeps_committed_face_history() -> None:
    case = PlaneStrainTpsa(cells_per_axis=2, boundary_displacement=boundary_displacement)
    case.prepare_simulation()
    ops, zero = CoupledOperators(case), TpsaState.zeros(case.grid, store_faces=True)
    initial = LoadStepController(ops, ops.transfer, case.material_points, zero, case.bc_values)
    initial.run([0.1, 0.3, 0.5])
    saved = initial.state

    def factory(old: TpsaState) -> JacobianCallback:
        exact = CoupledAnalyticalJacobian(ops, ops.transfer, case.material_points, old)

        def slowed(z: NDArray[np.float64], evaluation: TpsaResidual) -> sps.csr_array:
            increment = np.max(np.abs(evaluation.trial.bc_values - old.bc_values)) / np.max(np.abs(case.bc_values))
            # Deliberately prevent large attempts from converging within four corrections.
            return exact(z, evaluation) * (2 if increment > 0.11 else 1)

        return slowed

    controller = LoadStepController(ops, ops.transfer, case.material_points, saved, case.bc_values,
        initial_load_factor=0.5, jacobian_factory=factory, max_iterations=4)
    records = controller.run_adaptive([0.9, 0.85], options=AdaptiveLoadOptions(initial_increment=0.2))
    assert any(not attempt.accepted for attempt in controller.attempts)
    assert controller.load_factor == 0.85
    reference = LoadStepController(ops, ops.transfer, case.material_points, saved, case.bc_values,
        initial_load_factor=0.5, jacobian_factory=lambda old: CoupledAnalyticalJacobian(ops, ops.transfer, case.material_points, old))
    for record in records:
        expected = reference.advance(record.load_factor)
        np.testing.assert_array_equal(record.state.face_displacement, expected.state.face_displacement)
        np.testing.assert_array_equal(record.state.epsilon, expected.state.epsilon)


def test_face_orientation_changes_flux_sign_but_not_solution_maps() -> None:
    case = heterogeneous_case()
    before = CoupledOperators(case)
    signs = np.ones(case.grid.num_faces)
    signs[::2] = -1
    case.grid.face_normals *= signs
    case.grid.cell_faces = sps.csc_matrix(sps.diags(signs) @ case.grid.cell_faces)
    case.matrices = _discretize_get_matrices(case.grid, case.d)
    case.face_discretization, case.rhs_matrix, case.div, case.accum = _assemble_matrices(case.matrices, case.grid, case.d)
    after = CoupledOperators(case)
    for field in ("U", "H_f", "G", "H", "A"):
        np.testing.assert_allclose(getattr(after, field).toarray(), getattr(before, field).toarray(), rtol=1e-12, atol=1e-4)
    np.testing.assert_allclose(after.H_t.toarray(), np.repeat(signs, 2)[:, None] * before.H_t.toarray())


@pytest.mark.parametrize("defect", ["history", "weights", "material", "centroid"])
def test_incompatible_inputs_are_rejected(defect: str) -> None:
    case = heterogeneous_case()
    if defect == "centroid":
        case.grid.cell_centers[0, 0] += 0.01
        with pytest.raises(ValueError, match="centroids"):
            CoupledOperators(case)
        return
    ops = CoupledOperators(case)
    old = TpsaState.zeros(case.grid, store_faces=defect != "history")
    transfer = CellToFaceTransfer(case.grid) if defect == "weights" else ops.transfer
    if defect == "material":
        case.material_points[0].material.young_modulus *= 2
    with pytest.raises(ValueError, match={"history": "face_displacement", "weights": "complementary", "material": "coefficients"}[defect]):
        CoupledAnalyticalJacobian(ops, transfer, case.material_points, old)


def test_cli_exports_coupled_history_and_strain_block_heatmap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--formulation", "coupled", "--cells-per-axis", "2", "--adaptive", "--output-dir", str(tmp_path)])
    output = capsys.readouterr().out
    assert "Formulation: coupled" in output and "||R_eps||" in output and "Converged:" in output
    prefix = "dirichlet_example_coupled_analytical"
    collection = tmp_path / "pvd" / f"{prefix}.pvd"
    datasets = list(ET.parse(collection).iter("DataSet"))
    assert len(datasets) > 1
    for dataset in datasets:
        mesh = meshio.read(collection.parent / dataset.attrib["file"])
        assert {"displacement", "strain", "stress", "alpha"} <= set(mesh.cell_data)
    for suffix in ("stress_strain", "alpha", "final_displacement", "final_epsilon_xx", "jacobian"):
        assert (tmp_path / f"{prefix}_{suffix}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
