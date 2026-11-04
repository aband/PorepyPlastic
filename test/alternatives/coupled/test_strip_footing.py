"""Coupled benchmark acceptance, prototype comparison and labelled exports."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from alternatives.coupled.jacobian import CoupledAnalyticalJacobian
from alternatives.loading import LoadStepController
from alternatives.strip_footing import StripFootingCase, StripFootingResult, run_case
from alternatives.strip_footing_plots import export_results
from prototype.strip_footing import YIELD_STRESS, make_case, run_case as run_prototype
from utils.load_schedule import LoadStepError


@pytest.fixture(scope="module")
def solved() -> StripFootingResult:
    return run_case(StripFootingCase(1, growth=2), steps=4)


def test_same_benchmark_and_solution_as_retained_prototype(solved: StripFootingResult) -> None:
    reference = run_prototype(make_case(1, growth=2), steps=4)
    case, ops = solved.case, solved.case.operators
    np.testing.assert_array_equal(case.grid.nodes, reference.case.grid.nodes)
    np.testing.assert_array_equal(case.bc.is_dir, reference.case.bc.is_dir)
    np.testing.assert_array_equal(case.bc.is_neu, reference.case.bc.is_neu)
    np.testing.assert_array_equal(case.bc_values, reference.case.bc_values)
    assert len(solved.steps) == 4
    assert not any(not a.accepted for a in solved.attempts)
    for actual, expected in zip(solved.steps, reference.steps, strict=True):
        for name in ("x", "epsilon", "face_displacement"):
            np.testing.assert_allclose(getattr(actual.state, name), getattr(expected.state.tpsa, name), rtol=1e-5, atol=1e-8)
        ops.validate_committed(actual.state, case.material_points)
        assert np.all(actual.newton.residual_norms[-1] <= actual.newton.thresholds)
    np.testing.assert_allclose(solved.response[:, 4:7], reference.response[:, 4:7], rtol=1e-7, atol=1e-6)
    assert solved.response[:, 12].max() < 1e-5
    state = solved.state
    assert state.face_displacement is not None
    np.testing.assert_array_equal(state.traction[ops.neumann_mask], 0.0)
    np.testing.assert_array_equal(state.face_displacement[ops.dirichlet_mask], case.bc_values[ops.dirichlet_mask])
    assert np.max(abs(state.face_displacement[0, case.footing_faces])) > 1e-8
    reaction = -np.sum((state.traction*ops.boundary_signs)[1, case.footing_faces])
    assert solved.response[-1, 4] == pytest.approx(reaction/0.5)
    assert max(s.alpha for s in state.material_states) > 0
    for s in state.material_states:
        sigma = s.stress.to_numpy()
        dev = sigma-np.trace(sigma)*np.eye(3)/3
        assert np.sqrt(1.5*np.sum(dev**2)) <= YIELD_STRESS*(1+1e-9)
    assert all(p.trial is None and p.committed.alpha == 0 for p in case.material_points)


def test_failed_step_preserves_mixed_boundary_plastic_history(solved: StripFootingResult) -> None:
    case, ops = solved.case, solved.case.operators
    old = solved.steps[0].state
    controller = LoadStepController(ops, ops.transfer, case.material_points, old, case.bc_values,
        initial_load_factor=0.25, max_iterations=0,
        jacobian_factory=lambda s: CoupledAnalyticalJacobian(ops, ops.transfer, case.material_points, s))
    with pytest.raises(LoadStepError):
        controller.advance(0.5)
    assert controller.load_factor == 0.25 and not controller.steps
    for name in ("x", "epsilon", "traction", "face_displacement", "bc_values"):
        np.testing.assert_array_equal(getattr(controller.state, name), getattr(old, name))
    np.testing.assert_array_equal([s.alpha for s in controller.state.material_states], [s.alpha for s in old.material_states])


def test_exports_identify_alternatives_and_strain_residual(tmp_path: Path, solved: StripFootingResult) -> None:
    paths = export_results(solved, tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.name.startswith("alternatives_")
            if path.suffix == ".png":
                with Image.open(path) as im:
                    im.verify()
    data = json.loads(paths["summary"].read_text())
    assert data["solver_package"] == "alternatives"
    assert data["formulation"] == "coupled cell/strain TPSA"
    assert data["mesh"]["dofs"] == 7*solved.case.grid.num_cells
    assert "R_strain" in paths["csv"].read_text().splitlines()[0]
    assert "R_face" not in paths["csv"].read_text().splitlines()[0]
    for dataset in ET.parse(paths["vtk"]).iter("DataSet"):
        assert (paths["vtk"].parent/dataset.attrib["file"]).is_file()
