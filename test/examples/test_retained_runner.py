"""Retained formulation dispatch, full-state checks, and reusable example exports."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeAlias
import xml.etree.ElementTree as ET

from matplotlib.figure import Figure
import meshio  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
import pytest

from examples import plastic_plane_strain as homogeneous
from utils.load_schedule import AdaptiveLoadOptions, LoadStepError
from utils.visualization import export_jacobian_png
from examples import dirichlet_example as nonuniform
from prototype.jacobian import RetainedAnalyticalJacobian, RetainedFiniteDifferenceJacobian
from prototype.loading import LoadStepController
from prototype.operators import RetainedFaceOperators
from prototype.residual import evaluate_global_residual
from prototype.state import assemble_state_residual

Runner: TypeAlias = Callable[..., homogeneous.PlasticPlaneStrainResult | nonuniform.DirichletResult]


@pytest.mark.parametrize("jacobian", ["analytical", "finite-difference"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_retained_homogeneous_reference(jacobian: str, adaptive: bool) -> None:
    result = homogeneous.run_example(
        cells_per_axis=2, loading_steps=8, formulation="retained", jacobian=jacobian,
        adaptive=AdaptiveLoadOptions() if adaptive else None,
    )
    ops, controller = result.operators, result.controller
    assert isinstance(ops, RetainedFaceOperators)
    assert isinstance(controller, LoadStepController)
    assert result.formulation == "retained" and result.outputs == {}
    assert result.case.x is None and controller.load_factor == 0.98
    assert result.transfer is ops.rotation_transfer
    records = controller.steps
    assert len(records) == len(result.checks)
    for record, check in zip(records, result.checks, strict=True):
        assert record.newton.z.shape == (ops.num_dofs,)
        assert check.residual_norms.shape == (4,)
        residual = assemble_state_residual(ops, record.state)
        blocks = np.split(residual, [2 * ops.num_cells, 3 * ops.num_cells, 4 * ops.num_cells])
        np.testing.assert_array_equal(check.residual_norms, [np.linalg.norm(v) for v in blocks])
        assert np.all(check.residual_norms <= record.newton.thresholds)
        assert all(error <= check.limits[name] for name, error in check.errors.items())
        reference = homogeneous.homogeneous_reference(
            result.case, record.load_factor, unloading=check.phase == "unloading",
        )
        exact = reference.stress[:2, :2] @ (ops.side_normals * ops.side_areas)
        np.testing.assert_allclose(record.state.side_traction, exact, rtol=1e-7, atol=1)
    assert all(point.trial is None and point.committed.alpha == 0 for point in result.case.material_points)


@pytest.mark.parametrize("adaptive", [False, True])
def test_retained_nonuniform_jacobians_agree(adaptive: bool) -> None:
    analytical = nonuniform.run_example(cells_per_axis=3, formulation="retained", adaptive=adaptive)
    numerical = nonuniform.run_example(cells_per_axis=3, formulation="retained", adaptive=adaptive,
                                      jacobian="finite-difference")
    for result in (analytical, numerical):
        assert isinstance(result.controller, LoadStepController)
        case = result.case
        ops = RetainedFaceOperators(case.grid, case.bc, mu=case.material.isotropic_shear_modulus,
                                   lam=case.material.lame_parameter)
        records = result.controller.steps
        assert records[-1].load_factor == 0.98
        alpha = np.array([[s.alpha for s in r.state.tpsa.material_states] for r in records])
        assert np.all(alpha[0] == 0) and np.ptp(alpha[-1]) > 1e-5
        assert np.all(np.diff(alpha, axis=0) >= -1e-14)
        np.testing.assert_allclose(alpha[-1], alpha[-2], rtol=0, atol=1e-11)
        for record in records:
            state = record.state.tpsa
            assert state.face_displacement is not None
            np.testing.assert_array_equal(state.bc_values, record.load_factor * case.bc_values)
            np.testing.assert_array_equal(state.face_displacement[:, ops.boundary_faces],
                                          state.bc_values[:, ops.boundary_faces])
            residual = assemble_state_residual(ops, record.state)
            blocks = np.split(residual, [2 * ops.num_cells, 3 * ops.num_cells, 4 * ops.num_cells])
            assert np.all(np.array([np.linalg.norm(v) for v in blocks]) <= record.newton.thresholds)
    assert isinstance(analytical.controller, LoadStepController)
    assert isinstance(numerical.controller, LoadStepController)
    a, b = analytical.controller.state, numerical.controller.state
    nc = analytical.case.grid.num_cells
    np.testing.assert_allclose(a.tpsa.x[:2 * nc], b.tpsa.x[:2 * nc], rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(a.tpsa.x[2 * nc:], b.tpsa.x[2 * nc:], rtol=1e-6, atol=1)
    np.testing.assert_allclose(a.tpsa.epsilon, b.tpsa.epsilon, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(a.side_traction, b.side_traction, rtol=1e-6, atol=1)


def test_checks_use_both_side_histories_and_ignore_saved_norms() -> None:
    result = homogeneous.run_example(cells_per_axis=2, loading_steps=8, formulation="retained")
    ops, controller = result.operators, result.controller
    assert isinstance(ops, RetainedFaceOperators)
    assert isinstance(controller, LoadStepController)
    records = controller.steps
    peak_index = next(i for i, r in enumerate(records) if r.load_factor == 1)
    record = records[peak_index]
    old = records[peak_index - 1].state
    sides = np.flatnonzero(ops.side_faces == ops.free_faces[0])
    # Equal outward perturbations cancel in the oriented average, but not R_f.
    record.state.side_traction[0, sides] += 1e6
    record.newton.residual_norms[-1] = 0
    record.newton.evaluation.residual[:] = 0
    with pytest.raises(homogeneous.NumericalCheckError, match="residual norms"):
        homogeneous.check_step(result.case, ops, record, old, index=peak_index + 1, unloading=False)


@pytest.mark.parametrize("runner", [homogeneous.run_example, nonuniform.run_example],
                         ids=["homogeneous", "nonuniform"])
@pytest.mark.parametrize("jacobian", ["analytical", "finite-difference"])
def test_retained_exports_use_total_fields_and_peak_increment(
    runner: Runner, jacobian: Literal["analytical", "finite-difference"],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plotted: list[NDArray[np.float64]] = []
    labels: list[str] = []
    savefig = Figure.savefig

    def capture(figure: Figure, path: str | Path, **kwargs: Any) -> None:
        if str(path).endswith("_jacobian.png"):
            axes = figure.axes[0]
            plotted.append(np.asarray(axes.images[0].get_array(), dtype=np.float64).copy())
            labels.extend(label.get_text() for label in axes.get_xticklabels() + axes.get_yticklabels())
        savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    adaptive = True if runner is nonuniform.run_example else AdaptiveLoadOptions()
    result = runner(cells_per_axis=2, formulation="retained", jacobian=jacobian,
                    adaptive=adaptive, output_dir=tmp_path)
    assert isinstance(result.controller, LoadStepController)
    records = result.controller.steps
    case = result.case
    ops = RetainedFaceOperators(case.grid, case.bc, mu=case.material.isotropic_shear_modulus,
                               lam=case.material.lame_parameter)
    peak_index = next(i for i, r in enumerate(records) if r.load_factor == 1)
    peak, old = records[peak_index], records[peak_index - 1].state
    assert peak_index > 0
    tangent_type = RetainedAnalyticalJacobian if jacobian == "analytical" else RetainedFiniteDifferenceJacobian
    expected = tangent_type(ops, case.material_points, old.tpsa)(peak.newton.z, peak.newton.evaluation)
    assert len(plotted) == 1 and plotted[0].shape == (ops.num_dofs, ops.num_dofs)
    np.testing.assert_allclose(plotted[0], expected.toarray(), rtol=1e-13, atol=1e-12)
    assert r"$d_f$" in labels and r"$R_f$" in labels and r"$\varepsilon$" not in labels

    # Independent directional derivative at the accepted increment and OLD history.
    direction = np.random.default_rng(23).normal(size=ops.num_dofs)
    direction *= np.r_[np.full(2 * ops.num_cells, 1e-4), np.full(2 * ops.num_cells, 1e7),
                       np.full(2 * ops.num_free_faces, 1e-4)]

    def residual(z: NDArray[np.float64]) -> NDArray[np.float64]:
        return evaluate_global_residual(ops, case.material_points, old.tpsa, z,
                                        peak.state.tpsa.bc_values).residual

    h = 1e-5
    numerical = (residual(peak.newton.z + h * direction) - residual(peak.newton.z - h * direction)) / (2 * h)
    predicted = plotted[0] @ direction
    for block in (ops.u_slice, ops.r_slice, ops.p_slice, ops.face_slice):
        assert np.linalg.norm(predicted[block] - numerical[block]) < 1e-5 * np.linalg.norm(numerical[block])

    datasets = ET.parse(result.outputs["pvd"]).findall(".//DataSet")
    assert len(datasets) == len(records) == len(list((tmp_path / "vtu").glob("*.vtu")))
    for index, (dataset, record) in enumerate(zip(datasets, records, strict=True), 1):
        assert float(dataset.attrib["timestep"]) == index
        mesh = meshio.read(result.outputs["pvd"].parent / dataset.attrib["file"])
        np.testing.assert_array_equal(np.concatenate(mesh.cell_data["load_factor"]), record.load_factor)
        np.testing.assert_allclose(np.concatenate(mesh.cell_data["alpha"]),
                                   [s.alpha for s in record.state.tpsa.material_states])
        displacement = np.concatenate(mesh.cell_data["displacement"])
        np.testing.assert_allclose(displacement[:, :2].T,
                                   record.state.tpsa.x[:2 * ops.num_cells].reshape(2, -1, order="F"))
    for name, path in result.outputs.items():
        assert f"_retained_{jacobian}" in path.name
        if name != "pvd":
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(tmp_path.glob("*.pvd")) and not list(tmp_path.glob("*.vtu"))


@pytest.mark.parametrize("main", [homogeneous.main, nonuniform.main], ids=["homogeneous", "nonuniform"])
def test_retained_cli_flag_and_diagnostics(
    main: Callable[[list[str]], None], capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    main(["--formulation", "retained", "--jacobian", "finite-difference", "--adaptive",
          "--cells-per-axis", "2", "--no-export", "--output-dir", str(tmp_path)])
    output = capsys.readouterr().out
    assert "Formulation: retained" in output and "4 cells, 24 unknowns" in output
    assert "||R_f||" in output and "ratio_f" in output and "order_f" in output
    assert "||R_eps||" not in output and "L2" not in output
    assert not list(tmp_path.iterdir())


def test_retained_failure_propagates_without_export(tmp_path: Path) -> None:
    with pytest.raises(LoadStepError):
        homogeneous.run_example(formulation="retained", max_iterations=0, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("size,num_cells", [(24, None), (24, 0), (24, True), (23, 4), (12, 4)])
def test_retained_heatmap_requires_unambiguous_block_sizes(
    size: int, num_cells: int | None, tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="num_cells"):
        export_jacobian_png(np.eye(size), formulation="retained", num_cells=num_cells, folder_name=tmp_path)
    assert not list(tmp_path.iterdir())


def test_single_cell_retained_has_no_free_face_block(tmp_path: Path) -> None:
    result = homogeneous.run_example(cells_per_axis=1, loading_steps=4, formulation="retained", output_dir=tmp_path)
    assert isinstance(result.operators, RetainedFaceOperators)
    assert isinstance(result.controller, LoadStepController)
    assert result.operators.num_dofs == 4 and result.operators.num_free_faces == 0
    assert all(r.newton.residual_norms[-1, 3] == 0 for r in result.controller.steps)
    assert result.outputs["jacobian"].is_file()
