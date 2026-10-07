"""Trial reconstruction, material response, and history isolation.

The candidate x stores the current coupled fields [u, r, p].
The evaluator returns total strain, owned material histories, and the incremental
stress correction from the supplied committed state. Face traction, equilibrium,
commit, and Jacobian are separate steps.
"""

from copy import copy, deepcopy
from dataclasses import replace
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from coupling.plane_strain import PlaneStrainTpsa
from coupling.postprocessing import TpsaPostprocessing
from coupling.residual import TpsaOperators, TpsaState, evaluate_material_trial as evaluate
from material_state import MaterialPoint, MaterialState
from tensor import strain, stress

FloatArray: TypeAlias = NDArray[np.float64]

Setup: TypeAlias = tuple[PlaneStrainTpsa, TpsaOperators, TpsaState]


@pytest.fixture
def setup() -> Setup:
    case = PlaneStrainTpsa(cells_per_axis=2)
    case.prepare_simulation()
    return case, TpsaOperators(case), TpsaState.zeros(case.grid)


def affine_candidate(
    case: PlaneStrainTpsa, operators: TpsaOperators, gradient: FloatArray,
    translation: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    shift = np.zeros((2, 1)) if translation is None else translation[:, None]
    u = (gradient @ case.grid.cell_centers[:2] + shift).ravel(order="F")
    bc_values = np.zeros_like(case.bc_values)
    bf = operators.boundary_faces
    bc_values[:, bf] = gradient @ case.grid.face_centers[:2, bf] + shift
    r, p = operators.solve_auxiliary(u, operators.assemble_rhs(bc_values))
    return np.concatenate((u, r, p)), bc_values


def exact_strain(gradient: FloatArray) -> FloatArray:
    epsilon = np.zeros((3, 3))
    epsilon[:2, :2] = 0.5 * (gradient + gradient.T)
    return epsilon


def hooke(point: MaterialPoint, epsilon: FloatArray) -> FloatArray:
    return np.asarray(
        2 * point.material.isotropic_shear_modulus * epsilon
        + point.material.lame_parameter * np.trace(epsilon) * np.eye(3),
        dtype=np.float64,
    )


def linear_hardening_reference(
    point: MaterialPoint, old: MaterialState, increment: FloatArray,
) -> MaterialState:
    """Closed-form J2 oracle for the fixture's linear mixed hardening.

    Uses NumPy and the analytical plastic multiplier, never MaterialPoint.update
    or the implementation's local Newton/return-map routines.
    """
    parameters = point.parameters
    assert parameters.sigma_u == parameters.sigma_y
    predictor = old.stress.to_numpy() + hooke(point, increment)
    backstress = old.backstress.to_numpy()
    shifted = predictor - np.trace(predictor) / 3 * np.eye(3)
    shifted -= backstress - np.trace(backstress) / 3 * np.eye(3)
    norm = float(np.linalg.norm(shifted))
    c = np.sqrt(2 / 3)
    radius = c * (parameters.sigma_y + parameters.theta * parameters.H_bar * old.alpha)
    gamma = max(0.0, norm - radius) / (
        2 * point.material.isotropic_shear_modulus + 2 / 3 * parameters.H_bar
    )
    plastic_increment = np.zeros((3, 3)) if gamma == 0 else gamma * shifted / norm
    return MaterialState(
        stress=stress(predictor - hooke(point, plastic_increment)),
        plastic_strain=strain(old.plastic_strain.to_numpy() + plastic_increment),
        backstress=stress(
            backstress + 2 / 3 * (1 - parameters.theta) * parameters.H_bar * plastic_increment
        ),
        alpha=old.alpha + c * gamma,
    )


def assert_history(actual: MaterialState, expected: MaterialState) -> None:
    assert actual.alpha == pytest.approx(expected.alpha, rel=1e-10, abs=1e-14)
    for name, atol in (("stress", 1e-5), ("plastic_strain", 1e-13), ("backstress", 1e-5)):
        np.testing.assert_allclose(
            getattr(actual, name).to_numpy(), getattr(expected, name).to_numpy(),
            rtol=1e-10, atol=atol,
        )


def assert_unchanged(actual: TpsaState, saved: TpsaState) -> None:
    for name in ("x", "bc_values", "epsilon", "traction"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(saved, name))
    for history, old in zip(actual.material_states, saved.material_states, strict=True):
        assert history.alpha == old.alpha
        for name in ("stress", "plastic_strain", "backstress"):
            np.testing.assert_array_equal(
                getattr(history, name).to_numpy(), getattr(old, name).to_numpy(),
            )


def preloaded_state(case: PlaneStrainTpsa, operators: TpsaOperators) -> TpsaState:
    """Supply a compatible homogeneous plastic state independently of the new API."""
    gradient = np.array([[0.004, 0.0], [0.0, 0.0]])
    epsilon = exact_strain(gradient)
    x, bc_values = affine_candidate(case, operators, gradient)
    histories = [
        linear_hardening_reference(point, MaterialState(), epsilon)
        for point in case.material_points
    ]
    return TpsaState(
        x=x, bc_values=bc_values,
        epsilon=np.repeat(epsilon[:, :, None], case.grid.num_cells, axis=2),
        material_states=histories,
        traction=histories[0].stress.to_numpy()[:2, :2] @ case.grid.face_normals[:2],
    )


@pytest.mark.parametrize(
    "gradient, translation",
    [
        (np.zeros((2, 2)), np.zeros(2)),
        (np.zeros((2, 2)), np.array([0.012, -0.008])),
        (np.array([[0.0, -0.003], [0.003, 0.0]]), np.zeros(2)),
        (np.array([[1e-4, 2e-4], [-3e-4, 4e-4]]), np.array([1e-5, -2e-5])),
    ],
    ids=["zero", "translation", "infinitesimal-rotation", "elastic-tensor-shear"],
)
def test_affine_strain_and_elastic_stress(
    setup: Setup, gradient: FloatArray, translation: FloatArray,
) -> None:
    case, operators, committed = setup
    x, bc_values = affine_candidate(case, operators, gradient, translation)
    trial = evaluate(operators, case.material_points, committed, x, bc_values)
    epsilon = exact_strain(gradient)
    assert trial.epsilon.shape == trial.stress_correction.shape == (3, 3, case.grid.num_cells)
    assert len(trial.material_states) == case.grid.num_cells
    np.testing.assert_allclose(
        trial.epsilon, np.repeat(epsilon[:, :, None], case.grid.num_cells, axis=2),
        rtol=1e-10, atol=1e-14,
    )
    np.testing.assert_allclose(trial.stress_correction, 0.0, atol=1e-5)
    for point, history in zip(case.material_points, trial.material_states, strict=True):
        assert_history(history, MaterialState(stress=stress(hooke(point, epsilon))))
    assert case.x is None  # Evaluation must work without solving/publishing a case solution.


def test_plastic_trial_matches_closed_form_and_retains_3d_history(setup: Setup) -> None:
    case, operators, committed = setup
    gradient = np.array([[0.004, 0.001], [0.0, -0.001]])
    epsilon = exact_strain(gradient)
    x, bc_values = affine_candidate(case, operators, gradient)
    trial = evaluate(operators, case.material_points, committed, x, bc_values)
    np.testing.assert_allclose(trial.epsilon[2], 0.0, atol=1e-14)
    np.testing.assert_allclose(trial.epsilon[:, 2], 0.0, atol=1e-14)
    for cell, point in enumerate(case.material_points):
        expected = linear_hardening_reference(point, MaterialState(), epsilon)
        actual = trial.material_states[cell]
        assert_history(actual, expected)
        assert actual.alpha > 0.0
        assert abs(actual.stress.to_numpy()[2, 2]) > 1e6
        assert abs(actual.plastic_strain.to_numpy()[2, 2]) > 1e-5
        correction = hooke(point, epsilon) - expected.stress.to_numpy()
        np.testing.assert_allclose(trial.stress_correction[:, :, cell], correction, atol=1e-5)
        np.testing.assert_allclose(
            correction, hooke(point, expected.plastic_strain.to_numpy()), atol=1e-5,
        )


@pytest.mark.parametrize("factor", [1.0, 0.99, 1.2], ids=["no-increment", "unload", "reload"])
def test_increment_is_measured_from_committed_total_strain(setup: Setup, factor: float) -> None:
    case, operators, _ = setup
    committed = preloaded_state(case, operators)
    saved = committed.copy()
    gradient = np.array([[factor * 0.004, 0.0], [0.0, 0.0]])
    x, bc_values = affine_candidate(case, operators, gradient)
    trial = evaluate(operators, case.material_points, committed, x, bc_values)
    np.testing.assert_allclose(trial.epsilon, factor * committed.epsilon, atol=1e-14)
    for cell, point in enumerate(case.material_points):
        old = committed.material_states[cell]
        increment = exact_strain(gradient) - committed.epsilon[:, :, cell]
        expected = linear_hardening_reference(point, old, increment)
        assert_history(trial.material_states[cell], expected)
        correction = old.stress.to_numpy() + hooke(point, increment) - expected.stress.to_numpy()
        np.testing.assert_allclose(trial.stress_correction[:, :, cell], correction, atol=1e-5)
        if factor <= 1.0:
            assert trial.material_states[cell].alpha == pytest.approx(old.alpha, abs=1e-14)
            np.testing.assert_allclose(trial.stress_correction[:, :, cell], 0.0, atol=1e-5)
        else:
            assert trial.material_states[cell].alpha > old.alpha
    assert_unchanged(committed, saved)
    # The explicit committed argument is authoritative; the case's points are still virgin.
    assert all(point.committed.alpha == 0.0 for point in case.material_points)


def test_nonaffine_candidate_matches_existing_green_gauss_reconstruction(setup: Setup) -> None:
    case, operators, committed = setup
    rng = np.random.default_rng(15)
    u = 1e-5 * rng.normal(size=2 * case.grid.num_cells)
    bc_values = np.zeros_like(case.bc_values)
    bf = operators.boundary_faces
    xf, yf = case.grid.face_centers[:2, bf]
    bc_values[:, bf] = 1e-5 * np.array([xf * xf + xf * yf, yf * yf - xf * yf])
    r, p = operators.solve_auxiliary(u, operators.assemble_rhs(bc_values))
    x = np.concatenate((u, r, p))
    # Compare with the existing, separately tested reconstruction on an isolated case view.
    reference_case = copy(case)
    reference_case.x = x.copy()
    reference_case.bound_vec = bc_values.ravel(order="F").copy()
    reference_case.bc_values = reference_case.bound_vec.reshape(bc_values.shape, order="F")
    expected = TpsaPostprocessing(reference_case).strain_green_gauss()
    trial = evaluate(operators, case.material_points, committed, x, bc_values)
    np.testing.assert_allclose(trial.epsilon, expected, rtol=1e-10, atol=1e-14)
    for cell, point in enumerate(case.material_points):
        assert_history(
            trial.material_states[cell], MaterialState(stress=stress(hooke(point, expected[:, :, cell]))),
        )
    assert case.x is None


def test_constitutive_parameters_are_selected_by_cell(setup: Setup) -> None:
    case, operators, committed = setup
    for point, yield_stress in zip(case.material_points, [100e6, 250e6, 800e6, 1000e6], strict=True):
        point.parameters = replace(point.parameters, sigma_y=yield_stress, sigma_u=yield_stress)
    gradient = np.array([[0.004, 0.0], [0.0, 0.0]])
    x, bc_values = affine_candidate(case, operators, gradient)
    trial = evaluate(operators, case.material_points, committed, x, bc_values)
    for cell, point in enumerate(case.material_points):
        expected = linear_hardening_reference(point, MaterialState(), exact_strain(gradient))
        assert_history(trial.material_states[cell], expected)
    assert [history.alpha > 0 for history in trial.material_states] == [True, True, False, False]


def test_rejected_candidate_does_not_change_repeated_evaluation(setup: Setup) -> None:
    case, operators, _ = setup
    committed = preloaded_state(case, operators)
    saved = committed.copy()
    x, bc_values = affine_candidate(case, operators, np.array([[0.00396, 0.0], [0.0, 0.0]]))
    first = evaluate(operators, case.material_points, committed, x, bc_values)
    first_saved = deepcopy(first)
    rejected_x, rejected_bc = affine_candidate(case, operators, np.array([[0.008, 0.003], [0.0, 0.0]]))
    rejected = evaluate(operators, case.material_points, committed, rejected_x, rejected_bc)
    assert rejected.material_states[0].alpha > committed.material_states[0].alpha
    repeated = evaluate(operators, case.material_points, committed, x, bc_values)
    for result in (first, repeated):
        np.testing.assert_array_equal(result.epsilon, first_saved.epsilon)
        np.testing.assert_array_equal(result.stress_correction, first_saved.stress_correction)
        for actual, expected in zip(result.material_states, first_saved.material_states, strict=True):
            assert_history(actual, expected)
    assert_unchanged(committed, saved)


def test_outputs_are_owned_and_inputs_are_preserved(setup: Setup) -> None:
    case, operators, committed = setup
    x, bc_values = affine_candidate(case, operators, np.array([[0.004, 0.0], [0.0, 0.0]]))
    saved_x, saved_bc = x.copy(), bc_values.copy()
    saved = committed.copy()
    # Even pre-existing material-point trial data belong to the caller.
    for point in case.material_points:
        point.update(strain(np.diag([1e-4, 0.0, 0.0])))
    point_snapshots = [(point.committed.copy(), deepcopy(point.trial)) for point in case.material_points]
    x.flags.writeable = bc_values.flags.writeable = False
    trial = evaluate(operators, case.material_points, committed, x, bc_values)
    np.testing.assert_array_equal(x, saved_x)
    np.testing.assert_array_equal(bc_values, saved_bc)
    assert_unchanged(committed, saved)
    for point, (old, old_trial) in zip(case.material_points, point_snapshots, strict=True):
        assert_history(point.committed, old)
        assert point.trial is not None and old_trial is not None
        assert_history(point.trial, old_trial)
    for cell, history in enumerate(trial.material_states):
        for name in ("stress", "plastic_strain", "backstress"):
            output = getattr(history, name).to_numpy(copy=False)
            originals = [committed.material_states[cell], case.material_points[cell].committed]
            point_trial = case.material_points[cell].trial
            assert point_trial is not None
            originals.append(point_trial)
            for original in originals + trial.material_states[:cell]:
                assert not np.shares_memory(output, getattr(original, name).to_numpy(copy=False))
    other = deepcopy(trial.material_states[1])
    # Tensor.to_numpy(copy=False) is read-only; replace the tensor through its API.
    trial.material_states[0].stress = stress.zeros((3, 3))
    trial.material_states[0].alpha = -1.0
    trial.epsilon[:] = -1.0
    trial.stress_correction[:] = -1.0
    assert_history(trial.material_states[1], other)
    assert_unchanged(committed, saved)


def test_interior_boundary_storage_is_ignored(setup: Setup) -> None:
    case, operators, committed = setup
    x, bc_values = affine_candidate(case, operators, np.array([[1e-4, 0.0], [0.0, 0.0]]))
    expected = evaluate(operators, case.material_points, committed, x, bc_values)
    interior = np.setdiff1d(np.arange(case.grid.num_faces), operators.boundary_faces)
    bc_values[:, interior] = np.nan
    trial = evaluate(operators, case.material_points, committed, x, bc_values)
    np.testing.assert_array_equal(trial.epsilon, expected.epsilon)
    np.testing.assert_array_equal(trial.stress_correction, expected.stress_correction)
    for actual, reference in zip(trial.material_states, expected.material_states, strict=True):
        assert_history(actual, reference)
    assert np.all(np.isnan(bc_values[:, interior]))


def test_local_failure_propagates_without_partial_commit(
    setup: Setup, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, operators, committed = setup
    saved = committed.copy()
    x, bc_values = affine_candidate(case, operators, np.array([[0.004, 0.0], [0.0, 0.0]]))

    def fail(**kwargs: object) -> None:
        raise RuntimeError("Injected local return-map failure")

    # Fail after at least one successful cell update to expose partial commits.
    monkeypatch.setattr(case.material_points[1].model, "radial_return_map", fail)
    with pytest.raises(RuntimeError, match="Injected local return-map failure"):
        evaluate(operators, case.material_points, committed, x, bc_values)
    assert_unchanged(committed, saved)
    for point in case.material_points:
        assert_history(point.committed, MaterialState())
        assert point.trial is None
