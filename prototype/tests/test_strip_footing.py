"""Textbook material/BC conventions and a numerical prototype smoke test."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from prototype.strip_footing import (
    COHESION, DOMAIN_SIZE, HALF_WIDTH, LIMIT_FACTOR, POISSON_RATIO, YIELD_STRESS,
    YOUNG_MODULUS, StripFootingResult, make_case, run_case,
)


@pytest.mark.parametrize("n", [1, 3, 6])
def test_geometry_material_and_smooth_boundary_conditions(n: int) -> None:
    c = make_case(n)
    g, ops = c.grid, c.operators
    assert g.cell_volumes.sum() == pytest.approx(25.0)
    np.testing.assert_allclose(g.nodes[:2].min(axis=1), [0.0, -DOMAIN_SIZE])
    np.testing.assert_allclose(g.nodes[:2].max(axis=1), [DOMAIN_SIZE, 0.0])
    assert c.footing_faces.size == n
    assert g.face_areas[c.footing_faces].sum() == pytest.approx(HALF_WIDTH)
    bf = ops.boundary_faces
    x, y = g.face_centers[:2, bf]
    expected = np.vstack((np.isclose(x, 0.0) | np.isclose(x, 5.0),
                          np.isclose(y, -5.0) | (np.isclose(y, 0.0) & (x < HALF_WIDTH))))
    np.testing.assert_array_equal(ops.dirichlet_mask[:, bf], expected)
    assert not np.any(ops.dirichlet_mask[0, c.footing_faces])
    np.testing.assert_array_equal(c.bc_values[1, c.footing_faces], -0.002)
    for p in c.material_points:
        assert p.material.young_modulus == YOUNG_MODULUS == 1e7
        assert p.material.poisson_ratio == POISSON_RATIO == 0.48
        assert p.K_law(0.5, p.parameters) == (YIELD_STRESS, 0.0)
        assert p.H_law(0.5, p.parameters) == (0.0, 0.0)
    assert COHESION == pytest.approx(490.0, rel=1e-5)
    assert LIMIT_FACTOR == pytest.approx(5.141592653589793)


@pytest.fixture(scope="module")
def solved() -> StripFootingResult:
    return run_case(make_case(1, growth=2.0), steps=4)


def test_settlement_reaction_and_plasticity(solved: StripFootingResult) -> None:
    c, s = solved.case, solved.state
    ops = c.operators
    assert len(solved.steps) == 4
    assert s.tpsa.face_displacement is not None
    np.testing.assert_allclose(s.tpsa.face_displacement[ops.dirichlet_mask], c.bc_values[ops.dirichlet_mask], atol=1e-13)
    assert np.max(abs(s.tpsa.face_displacement[0, c.footing_faces])) > 1e-8  # Smooth, not clamped.
    boundary_sides = np.isin(ops.side_faces, ops.boundary_faces)
    mask = ~ops.dirichlet_mask[:, ops.side_faces[boundary_sides]]
    np.testing.assert_allclose(s.side_traction[:, boundary_sides][mask], 0.0, atol=2e-5)
    foot_sides = np.isin(ops.side_faces, c.footing_faces)
    reaction = -s.side_traction[1, foot_sides].sum()
    assert reaction > 0
    assert solved.response[-1, 4] == pytest.approx(reaction / 0.5)
    assert solved.response[-1, 5] == pytest.approx(solved.response[-1, 4] / COHESION)
    assert solved.response[-1, 3] == pytest.approx(0.002)
    assert solved.response[:, 12].max() < 1e-5
    assert max(p.alpha for p in s.tpsa.material_states) > 0
    np.testing.assert_array_equal(s.tpsa.epsilon[2], 0.0)
    for p in s.tpsa.material_states:
        sigma = p.stress.to_numpy()
        deviator = sigma - np.trace(sigma)*np.eye(3)/3
        assert np.sqrt(1.5*np.sum(deviator**2)) <= YIELD_STRESS*(1+1e-9)
        assert abs(np.trace(p.plastic_strain.to_numpy())) < 1e-12


def test_exports_label_prototype_and_report_mesh_error(tmp_path: Path, solved: StripFootingResult) -> None:
    from prototype.strip_footing_plots import export_results
    paths = export_results(solved, tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert path.name.startswith("prototype_")
            if path.suffix == ".png":
                with Image.open(path) as im:
                    im.verify()
    data = json.loads(paths["summary"].read_text())
    assert data["solver_package"] == "prototype"
    assert data["material"]["hardening"] == "none"
    assert data["reference"]["printed_pages"] == [252, 253, 254, 255]
    assert data["results"]["relative_difference_from_limit"] == pytest.approx(solved.response[-1, 5]/LIMIT_FACTOR-1)
    for dataset in ET.parse(paths["vtk"]).iter("DataSet"):
        assert (paths["vtk"].parent / dataset.attrib["file"]).is_file()
