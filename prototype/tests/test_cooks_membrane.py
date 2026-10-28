"""Cook's membrane geometry, units, affine consistency and prototype outputs."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import meshio  # type: ignore[import-untyped]
import numpy as np
from PIL import Image
import pytest
from porepy.grids.structured import TensorGrid
from porepy.params.bc import BoundaryConditionVectorial

from material.tensor import strain
from prototype.cooks_membrane import CooksMembraneResult, make_case, run_case, top_right_displacement
from prototype.cooks_membrane_plots import export_results
from prototype.operators import RetainedFaceOperators
from prototype.residual import evaluate_global_residual
from prototype.state import RetainedState
from prototype.tests.test_spatial import affine


@pytest.mark.parametrize("level", [0, 1, 2])
def test_staircase_geometry_preserves_loaded_and_clamped_edges(level: int) -> None:
    c = make_case(level)
    g, o = c.grid, c.operators
    assert not isinstance(g, TensorGrid)  # extract_subgrid returns a general Grid.
    np.testing.assert_allclose(np.ptp(g.nodes, axis=1), [48., 60., 0.])
    np.testing.assert_allclose(g.cell_volumes, c.spacing**2)
    assert g.face_areas[o.dirichlet_faces].sum() == pytest.approx(44.)
    assert g.face_areas[c.tip_faces].sum() == pytest.approx(16.)
    assert g.face_centers[1, c.tip_faces[0]] - c.spacing/2 == pytest.approx(44.)
    assert g.face_centers[1, c.tip_faces[-1]] + c.spacing/2 == pytest.approx(60.)
    np.testing.assert_allclose(c.reference_neumann.sum(axis=1), [0., 100.])
    # The centre-selection rule introduces a measurable boundary geometry error.
    assert g.cell_volumes.sum() == pytest.approx(1440. + 8.*c.spacing)
    offsets = g.face_centers[:2, o.side_faces] - g.cell_centers[:2, o.side_cells]
    np.testing.assert_allclose(offsets, o.side_normals*o.side_distances, atol=1e-14)
    m = c.material_points[0].material
    assert m.young_modulus == 240.565 and m.poisson_ratio == .4999


def test_affine_patch_on_cartesian_subgrid_and_corner_extrapolation() -> None:
    c = make_case(0)
    g, o = c.grid, c.operators
    gradient = np.array([[2e-5, 3e-5], [-1e-5, 4e-5]])
    z, boundary = affine(o, gradient)
    eps = np.zeros((3, 3)); eps[:2, :2] = .5*(gradient+gradient.T)
    stress = c.material_points[0].material.elastic_tensor.double_contract(strain(eps)).to_numpy()
    sides = np.array([np.flatnonzero(o.side_faces == f)[0] for f in o.neumann_faces])
    forces = stress[:2, :2] @ o.side_normals[:, sides] * o.side_areas[sides]
    trial = evaluate_global_residual(o, c.material_points, RetainedState.zeros(o).tpsa,
                                    z, boundary, neumann_increment=forces)
    np.testing.assert_allclose(trial.residual, 0., atol=2e-11)
    np.testing.assert_allclose(trial.material_trial.epsilon, np.repeat(eps[:, :, None], g.num_cells, axis=2), atol=1e-17)
    assert top_right_displacement(c, boundary) == pytest.approx(float(gradient[1] @ np.array([48., 60.])))
    # A corner value must not silently become the mean displacement on the tip.
    assert not np.isclose(top_right_displacement(c, boundary), boundary[1, c.tip_faces].mean())


def test_general_grid_still_rejects_nonorthogonal_traction_geometry() -> None:
    c = make_case(0)
    c.grid.cell_centers[0, 0] += .1
    bc = BoundaryConditionVectorial(c.grid, c.operators.dirichlet_faces, "dir")
    with pytest.raises(ValueError, match="orthogonal"):
        RetainedFaceOperators(c.grid, bc, mu=80., lam=400000.)
    with pytest.raises(ValueError, match="nonnegative integer"):
        make_case(-1)


@pytest.fixture(scope="module")
def solved() -> CooksMembraneResult:
    return run_case(make_case(0))


def test_elastic_load_path_and_plane_strain_stress(solved: CooksMembraneResult) -> None:
    split = run_case(make_case(0), steps=4)
    np.testing.assert_allclose(split.state.tpsa.x, solved.state.tpsa.x, rtol=2e-9, atol=2e-8)
    np.testing.assert_allclose(split.response[1:, 2], split.response[1:, 1]*solved.response[-1, 2], atol=2e-8)
    assert np.max(split.response[:, 9]) < 1e-7
    o, s = solved.case.operators, solved.state.tpsa
    assert o.lam is not None and o.mu is not None
    stress = np.stack([p.stress.to_numpy() for p in s.material_states], axis=2)
    np.testing.assert_allclose(np.trace(stress, axis1=0, axis2=1), (3.+2.*o.mu/o.lam)*s.x[o.p_slice], rtol=1e-7, atol=1e-6)
    np.testing.assert_array_equal(s.epsilon[2], 0.)
    assert np.max(abs(stress[2, 2])) > 0.
    assert all(p.alpha == 0. for p in s.material_states)
    root_sides = np.isin(o.side_faces, o.dirichlet_faces)
    np.testing.assert_allclose(solved.state.side_traction[:, root_sides].sum(axis=1), [0., -100.], atol=1e-7)


def test_all_exported_files_identify_prototype_and_vtk_fields(tmp_path: Path, solved: CooksMembraneResult) -> None:
    paths = export_results([solved], tmp_path)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files and all(p.name.startswith("prototype_") for p in files)
    assert len([p for p in files if p.suffix == ".png"]) == 3
    for p in files:
        if p.suffix == ".png":
            with Image.open(p) as im:
                im.verify()
    for dataset in ET.parse(paths["vtk_l0"]).iter("DataSet"):
        assert (paths["vtk_l0"].parent / dataset.attrib["file"]).is_file()
    mesh = meshio.read(next((tmp_path/"vtu").glob("*.vtu")))
    assert mesh.cell_data["displacement"][0].shape == (solved.case.grid.num_cells, 3)
    assert mesh.cell_data["stress"][0].shape[1] == 9
    metadata = json.loads(paths["summary"].read_text())
    assert metadata["solver_package"] == "prototype" and metadata["plane_strain"]
    assert metadata["traction_MPa"] == [0., 6.25]
    assert "staircase" in metadata["geometry_approximation"]
    assert metadata["reference"]["uy_bbar_mm"][-1] == pytest.approx(7.999718483861992)
