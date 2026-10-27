"""End-to-end checks of the nonuniform Dirichlet plasticity example."""

from pathlib import Path
import xml.etree.ElementTree as ET

import meshio  # type: ignore[import-untyped]
import numpy as np
import pytest

from examples.dirichlet_example import main, run_example
from examples._solver_setup import prepare_solver
from alternatives.original.residual import TpsaOperators, assemble_global_residual
from alternatives.loading import LoadStepController
from alternatives.plane_strain import PlaneStrainTpsa


@pytest.mark.parametrize("adaptive", [False, True], ids=["fixed", "adaptive"])
def test_nonuniform_example_equilibrates_and_jacobians_agree(adaptive: bool) -> None:
    analytical = run_example(formulation="original", adaptive=adaptive)
    numerical = run_example(formulation="original", adaptive=adaptive, jacobian="finite-difference")
    for result in (analytical, numerical):
        assert isinstance(result.controller, LoadStepController)
        assert result.outputs == {}
        assert result.case.x is None  # All steps went through the coupled Newton solver.
        assert result.controller.load_factor == 0.98
        records = result.controller.steps
        assert isinstance(result.case, PlaneStrainTpsa)
        ops = TpsaOperators(result.case)
        alpha = np.array([[s.alpha for s in record.state.material_states] for record in records])
        assert np.all(alpha[0] == 0)
        assert np.all(np.diff(alpha, axis=0) >= -1e-14)
        assert np.max(alpha[-1]) > 0
        peak_index = next(i for i, record in enumerate(records) if record.load_factor == 1)
        assert np.ptp(alpha[peak_index]) > 1e-5  # Spatially varying plastic response.
        assert np.ptp(records[peak_index].state.epsilon[0, 0]) > 1e-6
        np.testing.assert_allclose(alpha[-1], alpha[peak_index], rtol=0, atol=1e-11)
        for record in records:
            state = record.state
            np.testing.assert_array_equal(state.bc_values, record.load_factor * result.case.bc_values)
            np.testing.assert_array_equal(state.epsilon[2], 0)
            residual = assemble_global_residual(
                ops, state.x, state.bc_values, state.traction, sources=record.sources,
            )
            blocks = np.split(residual, [2 * ops.num_cells, 3 * ops.num_cells])
            assert np.all(np.array([np.linalg.norm(block) for block in blocks]) <= record.newton.thresholds)
    assert isinstance(analytical.controller, LoadStepController)
    assert isinstance(numerical.controller, LoadStepController)
    analytical_steps = analytical.controller.steps
    numerical_steps = numerical.controller.steps
    if adaptive:
        # Newton iteration counts can produce different adaptive increments.
        # Plastic history depends on those increments, so compare both Jacobians
        # on the same accepted path, starting from a fresh initial state.
        replay = prepare_solver(numerical.case, "original", "analytical").controller
        assert isinstance(replay, LoadStepController)
        analytical_steps = replay.run([record.load_factor for record in numerical_steps])
    nc = analytical.case.grid.num_cells
    np.testing.assert_array_equal([step.load_factor for step in analytical_steps],
                                  [step.load_factor for step in numerical_steps])
    a, b = analytical_steps[-1].state, numerical_steps[-1].state
    np.testing.assert_allclose(a.x, b.x, rtol=1e-6, atol=1e-3)
    np.testing.assert_allclose(a.x[:2 * nc], b.x[:2 * nc], rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(a.epsilon, b.epsilon, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose([s.alpha for s in a.material_states],
                               [s.alpha for s in b.material_states], rtol=1e-6, atol=1e-9)


def test_cli_exports_accepted_states_and_all_plots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--formulation", "original", "--adaptive", "--output-dir", str(tmp_path)])
    output = capsys.readouterr().out
    assert "Converged:" in output and "ratio_u" in output
    assert "Formulation: original" in output
    assert "L2" not in output
    prefix = "dirichlet_example_original_analytical"
    collection = tmp_path / "pvd" / f"{prefix}.pvd"
    datasets = list(ET.parse(collection).iter("DataSet"))
    assert len(datasets) > 1
    for index, dataset in enumerate(datasets, 1):
        assert float(dataset.attrib["timestep"]) == index
        vtu = collection.parent / dataset.attrib["file"]
        assert vtu.is_file()
        mesh = meshio.read(vtu)
        assert set(["displacement", "strain", "stress", "alpha", "load_factor"]) <= set(mesh.cell_data)
        np.testing.assert_array_equal(np.concatenate(mesh.cell_data["load_step"]), index)
    np.testing.assert_array_equal(np.concatenate(mesh.cell_data["load_factor"]), 0.98)
    for suffix in ("stress_strain", "alpha", "final_displacement", "final_epsilon_xx",
                   "final_epsilon_yy", "final_epsilon_xy", "final_principal_strain", "jacobian"):
        png = tmp_path / f"{prefix}_{suffix}.png"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(tmp_path.glob("*.vtu"))
    assert not list(tmp_path.glob("*.pvd"))


def test_cli_reports_invalid_case_without_exporting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--cells-per-axis", "0", "--output-dir", str(tmp_path)])
    assert caught.value.code == 1
    assert "Example failed:" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
