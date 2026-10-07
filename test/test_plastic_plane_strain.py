"""Runnable plastic benchmark, independent reference, and numerical-check failures."""

from pathlib import Path

import numpy as np
import pytest

from coupling.plane_strain import PlaneStrainTpsa
from coupling.plastic_plane_strain import (
    NumericalCheckError, PlasticPlaneStrainResult, check_step,
    homogeneous_reference, main, run_example,
)


@pytest.fixture(scope="module")
def example() -> PlasticPlaneStrainResult:
    return run_example()


def test_default_runner_reaches_yield_and_retains_every_checked_step(example: PlasticPlaneStrainResult) -> None:
    assert example.outputs == {}  # The Python default does not export.
    records = example.controller.steps
    assert example.case.grid.num_cells == 9
    assert len(records) == len(example.checks) == 22
    np.testing.assert_allclose([record.load_factor for record in records], np.r_[np.linspace(0.05, 1, 20), 0.99, 0.98])
    assert all(check.phase == "loading" for check in example.checks[:20])
    assert all(check.phase == "unloading" for check in example.checks[20:])
    assert example.checks[0].alpha_max == 0.0
    peak = records[19].state
    mu, hardening = example.case.material.isotropic_shear_modulus, example.case.hardening_parameters
    # Scalar constrained-extension formula, separate from the tensor reference.
    alpha_exact = (2 * mu * 0.004 - hardening.sigma_y) / (3 * mu + hardening.H_bar)
    for history in peak.material_states:
        assert history.alpha == pytest.approx(alpha_exact, rel=1e-7)
        sigma = history.stress.to_numpy()
        assert sigma[2, 2] > 0.0
        assert sigma[1, 1] == pytest.approx(sigma[2, 2], rel=1e-8)
        assert sigma[0, 0] - sigma[1, 1] == pytest.approx(hardening.sigma_y + hardening.H_bar * alpha_exact, rel=1e-8)
        assert history.plastic_strain.to_numpy()[2, 2] < 0.0
    for record, report in zip(records, example.checks, strict=True):
        assert report.iterations == record.newton.iterations
        np.testing.assert_array_equal(report.residual_norms, record.newton.residual_norms[-1])
        assert np.all(report.residual_norms <= record.newton.thresholds)
        assert len(report.errors) == len(report.limits) == 9
        assert all(np.isfinite(error) and error <= report.limits[name] for name, error in report.errors.items())
        np.testing.assert_array_equal(record.state.epsilon[2], 0.0)
    for record in records[20:]:
        for history, old in zip(record.state.material_states, peak.material_states, strict=True):
            assert history.alpha == pytest.approx(old.alpha, abs=1e-11)
            np.testing.assert_allclose(history.plastic_strain.to_numpy(), old.plastic_strain.to_numpy(), atol=1e-11)
    assert example.case.x is None  # The runner never replaces its history with case.solve().
    assert all(point.trial is None and point.committed.alpha == 0 for point in example.case.material_points)


def test_reference_covers_zero_elastic_plastic_and_unloading_states() -> None:
    case = PlaneStrainTpsa(displacement_gradient=np.diag([0.004, 0.0]))
    zero = homogeneous_reference(case, 0)
    np.testing.assert_array_equal(zero.stress, 0.0)
    assert zero.alpha == 0.0
    elastic = homogeneous_reference(case, 0.1)
    np.testing.assert_allclose(elastic.stress, 0.1 * case.reference_stress(), rtol=1e-14)
    assert elastic.alpha == 0.0
    peak = homogeneous_reference(case, 1)
    assert peak.alpha > 0.0
    unloaded = homogeneous_reference(case, 0.98, unloading=True)
    depsilon = unloaded.epsilon - peak.epsilon
    expected = peak.stress + 2 * case.material.isotropic_shear_modulus * depsilon
    expected += case.material.lame_parameter * np.trace(depsilon) * np.eye(3)
    np.testing.assert_allclose(unloaded.stress, expected)
    np.testing.assert_array_equal(unloaded.plastic_strain, peak.plastic_strain)
    np.testing.assert_array_equal(unloaded.backstress, peak.backstress)
    assert unloaded.alpha == peak.alpha


def test_cli_with_alternate_grid_and_schedule(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(["--cells-per-axis", "1", "--loading-steps", "8", "--unloading-steps", "3", "--unload-fraction", "0.03",
          "--output-dir", str(tmp_path)])
    output = capsys.readouterr().out
    assert "1 cells, 4 unknowns" in output
    assert "final load factor=0.97" in output
    assert "||R_u||" in output and "||R_r||" in output and "||R_p||" in output
    assert "alpha_max" in output
    assert "ratio_u" in output and "order_u" in output
    assert output.count("iter       ||R_u||") == 11  # A separate history for every load step.
    assert "Maximum absolute discrete L2 errors" in output
    assert "stress:" in output and "[Pa m]" in output
    assert "Numerical checks: PASS" in output
    assert (tmp_path / "pvd" / "plastic_plane_strain.pvd").is_file()
    assert (tmp_path / "plastic_plane_strain_stress_strain.png").is_file()
    assert (tmp_path / "plastic_plane_strain_alpha.png").is_file()
    assert sum(line.split()[1:2] == ["loading"] for line in output.splitlines()) == 8
    assert sum(line.split()[1:2] == ["unloading"] for line in output.splitlines()) == 3


@pytest.mark.parametrize("arguments, message", [
    (["--max-iterations", "0"], "Load step 1"),
    (["--unload-fraction", "1"], "reverse plastic yield"),
    (["--peak-strain", "0.0001"], "produce plastic loading"),
])
def test_cli_fails_visibly_without_success_message(
    capsys: pytest.CaptureFixture[str], arguments: list[str], message: str,
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(arguments)
    output = capsys.readouterr()
    assert caught.value.code == 1
    assert "Example failed:" in output.err and message in output.err
    assert "Numerical checks: PASS" not in output.out


@pytest.mark.parametrize("field, message", [
    ("stress", "stress L2 error"),
    ("plastic_strain", "plastic_strain L2 error"),
    ("backstress", "backstress L2 error"),
    ("alpha", "alpha L2 error"),
    ("strain", "violates plane strain"),
    ("traction", "residual norms"),
])
def test_checks_reject_corrupted_accepted_fields(
    example: PlasticPlaneStrainResult, field: str, message: str,
) -> None:
    records = example.controller.steps  # Owned snapshots; module fixture remains untouched.
    peak = records[19]
    if field == "alpha":
        peak.state.material_states[0].alpha += 1e-3
    elif field == "strain":
        peak.state.epsilon[2, 2, 0] = 1e-3
    elif field == "traction":
        peak.state.traction[0, 0] += 1e6
        # Force checks must reassemble from state, rather than trust saved diagnostics.
        peak.newton.residual_norms[-1] = 0.0
        peak.newton.evaluation.residual[:] = 0.0
    else:
        tensor = getattr(peak.state.material_states[0], field)
        values = tensor.to_numpy()
        values[2, 2] += 1e-3 if field == "plastic_strain" else 1e6
        setattr(peak.state.material_states[0], field, type(tensor)(values))
    with pytest.raises(NumericalCheckError, match=message):
        check_step(example.case, example.operators, peak, records[18].state, index=20, unloading=False)


@pytest.mark.parametrize("field", ["alpha", "plastic_strain", "backstress"])
def test_unloading_checks_detect_small_history_drift(example: PlasticPlaneStrainResult, field: str) -> None:
    records = example.controller.steps
    current = records[20]
    # Below the relative reference-error allowance, but above the stricter
    # no-change tolerance for elastic unloading from the numerical peak.
    if field == "alpha":
        current.state.material_states[0].alpha += 5e-11
    else:
        tensor = getattr(current.state.material_states[0], field)
        values = tensor.to_numpy()
        values[0, 0] += 5e-11 if field == "plastic_strain" else 0.02
        setattr(current.state.material_states[0], field, type(tensor)(values))
    with pytest.raises(NumericalCheckError, match=f"{field} changed during elastic unloading"):
        check_step(
            example.case, example.operators, current, records[19].state,
            index=21, unloading=True, peak=records[19].state,
        )


def test_accumulated_plastic_strain_cannot_decrease(example: PlasticPlaneStrainResult) -> None:
    records = example.controller.steps
    records[19].state.material_states[0].alpha = 0.0
    with pytest.raises(NumericalCheckError, match="accumulated plastic strain decreased"):
        check_step(example.case, example.operators, records[19], records[18].state, index=20, unloading=False)


@pytest.mark.parametrize("count", [0, -1, True])
def test_invalid_increment_counts_are_rejected(count: int) -> None:
    with pytest.raises(ValueError, match="loading_steps.*positive integer"):
        run_example(loading_steps=count)
    with pytest.raises(ValueError, match="unloading_steps.*positive integer"):
        run_example(unloading_steps=count)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_strain_and_unload_fraction_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="peak_strain.*positive"):
        run_example(peak_strain=value)
    with pytest.raises(ValueError, match="unload_fraction.*lie in"):
        run_example(unload_fraction=value)


def test_reference_rejects_nonlinear_hardening() -> None:
    case = PlaneStrainTpsa()
    case.hardening_parameters.sigma_u = 400e6
    case.hardening_parameters.delta = 10.0
    with pytest.raises(ValueError, match="linear mixed hardening"):
        homogeneous_reference(case, 1.0)


def test_cli_no_export_overrides_output_directory(tmp_path: Path) -> None:
    folder = tmp_path / "not_created"
    main(["--cells-per-axis", "1", "--loading-steps", "8", "--no-export",
          "--output-dir", str(folder)])
    assert not folder.exists()


def test_runner_returns_generated_paths(tmp_path: Path) -> None:
    result = run_example(cells_per_axis=1, loading_steps=8, output_dir=tmp_path)
    assert set(result.outputs) == {"pvd", "stress_strain", "alpha"}
    for name, path in result.outputs.items():
        assert path.parent == (tmp_path / "pvd" if name == "pvd" else tmp_path)
        assert path.is_file()
