"""Independent physical checks of Galin fields and the staircase benchmark."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import meshio  # type: ignore[import-untyped]
import numpy as np
from PIL import Image
import pytest

from prototype.yarushina import (
    CircularHoleResult, analytical_elastic_displacement, analytical_stress,
    load_path, make_case, plastic_axes, run_case, stress_measures,
)
from prototype.yarushina_plots import export_results


@pytest.mark.parametrize("pressure", [.2, 1., 1.4, 2.])
def test_hole_traction_and_plastic_shear_strength(pressure: float) -> None:
    theta = np.linspace(0., 2.*np.pi, 101)
    normals = np.stack((np.cos(theta), np.sin(theta)))
    stress = analytical_stress(normals, pressure)
    np.testing.assert_allclose(np.einsum("ijn,jn->in", stress, normals), 0., atol=2e-14)
    mean, shear = stress_measures(stress)
    np.testing.assert_allclose(shear, min(pressure, 1.), atol=1e-14)
    np.testing.assert_allclose(mean, -min(pressure, 1.), atol=1e-14)


def test_elastic_pressure_limit_matches_independent_lame_solution() -> None:
    theta = np.linspace(0., 2.*np.pi, 37)
    r, pressure, nu, young = 1.7, .7, .4999, 100.
    n = np.stack((np.cos(theta), np.sin(theta)))
    rr, hoop = -pressure*(1.-1./r**2), -pressure*(1.+1./r**2)
    exact = hoop*np.eye(2)[:, :, None]+(rr-hoop)*np.einsum("in,jn->ijn", n, n)
    np.testing.assert_allclose(analytical_stress(r*n, pressure), exact, atol=2e-15)
    ur = -pressure*(1.+nu)/young*((1.-2.*nu)*r+1./r)
    np.testing.assert_allclose(analytical_elastic_displacement(r*n, pressure), ur*n, atol=2e-17)
    np.testing.assert_allclose(analytical_stress(r*n, 1.+1e-9), analytical_stress(r*n, 1.), atol=5e-9)
    np.testing.assert_allclose(analytical_elastic_displacement(r*n, 1.+1e-9), analytical_elastic_displacement(r*n, 1.), atol=5e-11)


def test_interface_continuity_and_admissible_loading() -> None:
    theta = np.linspace(0., 2.*np.pi, 81)
    previous = np.array([1., 1.])
    for pressure in np.linspace(1., 2., 11):
        ax, ay = plastic_axes(float(pressure))
        tau = load_path(float(pressure))
        assert ax >= 1. and ay >= 1. and abs(tau) < np.sqrt(2.)-1.
        assert np.all(np.array([ax, ay]) >= previous-1e-14)
        previous = np.array([ax, ay])
        # dP >= 2*|dtau|/(1-|tau|), with |dtau/dP|=0.3.
        assert 2.*.3/(1.-abs(tau)) <= 1.
        if pressure > 1.:
            boundary = np.stack((ax*np.cos(theta), ay*np.sin(theta)))
            inside = analytical_stress(boundary*(1.-1e-7), float(pressure))
            outside = analytical_stress(boundary*(1.+1e-7), float(pressure))
            np.testing.assert_allclose(inside, outside, atol=3e-6)


@pytest.mark.parametrize("pressure", [.5, 2.])
def test_elastic_displacement_derivative_and_stress_equilibrium(pressure: float) -> None:
    xy = np.array([[2.7, -2.7, -2.7, 2.7], [1.8, 1.8, -1.8, -1.8]])
    gradient = np.empty((2, 2, 4))
    divergence = np.zeros((2, 4))
    for j in range(2):
        shift = np.zeros_like(xy); shift[j] = 1e-5
        gradient[:, j] = (analytical_elastic_displacement(xy+shift, pressure)-analytical_elastic_displacement(xy-shift, pressure))/2e-5
        divergence += (analytical_stress(xy+shift, pressure)[:, j]-analytical_stress(xy-shift, pressure)[:, j])/2e-5
    nu, young = .4999, 100.
    mu, lam = young/(2.*(1.+nu)), young*nu/((1.+nu)*(1.-2.*nu))
    epsilon = .5*(gradient+gradient.swapaxes(0, 1))
    hooke = 2.*mu*epsilon+lam*np.trace(epsilon, axis1=0, axis2=1)[None, None, :]*np.eye(2)[:, :, None]
    np.testing.assert_allclose(hooke, analytical_stress(xy, pressure), atol=2e-6)
    np.testing.assert_allclose(divergence, 0., atol=2e-9)


def test_far_field_branches_and_plastic_equilibrium() -> None:
    theta = np.linspace(0., 2.*np.pi, 53)
    xy = 1e5*np.stack((np.cos(theta), np.sin(theta)))
    stress = analytical_stress(xy, 2.)
    np.testing.assert_allclose(stress[0, 0], -1.7, atol=1e-8)
    np.testing.assert_allclose(stress[1, 1], -2.3, atol=1e-8)
    np.testing.assert_allclose(stress[0, 1], 0., atol=1e-8)
    # Independent radial equilibrium: d sigma_rr/dr + (sigma_rr-sigma_tt)/r=0.
    r = 1.1; delta = 1e-5
    radial = (analytical_stress(np.array([[r+delta], [0.]]), 2.)[0, 0]
              -analytical_stress(np.array([[r-delta], [0.]]), 2.)[0, 0])/(2.*delta)
    middle = analytical_stress(np.array([[r], [0.]]), 2.)
    np.testing.assert_allclose(radial+(middle[0, 0]-middle[1, 1])/r, 0., atol=1e-9)


@pytest.mark.parametrize("level", [0, 1])
def test_staircase_geometry_bc_and_material(level: int) -> None:
    case = make_case(level)
    g, o = case.grid, case.operators
    assert np.all(np.linalg.norm(g.cell_centers[:2], axis=0) >= 1.)
    np.testing.assert_allclose(g.cell_volumes, case.spacing**2)
    np.testing.assert_allclose(g.face_centers[:2, o.side_faces]-g.cell_centers[:2, o.side_cells], o.side_normals*o.side_distances, atol=1e-14)
    assert np.all(np.any(np.isclose(abs(g.face_centers[:2, o.dirichlet_faces]), case.half_width), axis=0))
    assert np.all(abs(np.linalg.norm(g.face_centers[:2, o.neumann_faces], axis=0)-1.) <= case.spacing)
    assert o.neumann_faces.size > 0
    p = case.material_points[0]
    assert p.parameters.sigma_y == pytest.approx(np.sqrt(3.))
    assert p.parameters.H_bar == 0. and p.parameters.sigma_u == p.parameters.sigma_y
    assert p.material.poisson_ratio == .4999


def test_reference_rejects_plastic_displacement_and_invalid_path() -> None:
    with pytest.raises(ValueError, match="elastic region"):
        analytical_elastic_displacement(np.array([[1.1], [0.]]), 2.)
    with pytest.raises(ValueError, match="r>=R"):
        analytical_stress(np.zeros((2, 1)), 2.)
    with pytest.raises(ValueError, match="pressure"):
        load_path(2.1)
    with pytest.raises(ValueError, match="nu"):
        make_case(nu=.5)


@pytest.fixture(scope="module")
def solved() -> CircularHoleResult:
    return run_case(make_case(0), steps=4)


def test_nonlinear_run_equilibrium_yield_and_no_material_point_mutation(solved: CircularHoleResult) -> None:
    r, o = solved, solved.case.operators
    assert r.response[-1, 1] == 2. and r.response[-1, 2] == pytest.approx(-.3)
    assert np.max(r.response[:, 4:8]) < 2e-8
    assert 0. < r.response[-1, 8] < .15 and 0. < r.response[-1, 9] < .15
    assert any(s.alpha > 1e-5 for s in r.state.tpsa.material_states)
    assert all(p.committed.alpha == 0. for p in r.case.material_points)
    np.testing.assert_array_equal(r.state.neumann_values, 0.)
    boundary = np.isin(o.side_faces, o.boundary_faces)
    np.testing.assert_allclose(r.state.side_traction[:, boundary].sum(axis=1), 0., atol=1e-9)
    dev = r.stress-np.trace(r.stress, axis1=0, axis2=1)[None, None, :]*np.eye(3)[:, :, None]/3.
    mises = np.sqrt(1.5*np.einsum("ijn,ijn->n", dev, dev))
    assert np.max(mises) <= np.sqrt(3.)+2e-9


def test_exports_and_error_definitions(tmp_path: Path, solved: CircularHoleResult) -> None:
    paths = export_results([solved], tmp_path)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert all(p.name.startswith("prototype_") for p in files)
    pngs = [p for p in files if p.suffix == ".png"]
    assert len(pngs) == 4
    for path in pngs:
        with Image.open(path) as im:
            im.verify()
    for dataset in ET.parse(paths["vtk_l0"]).iter("DataSet"):
        assert (paths["vtk_l0"].parent/dataset.attrib["file"]).exists()
    mesh = meshio.read(next((tmp_path/"vtu").glob("*.vtu")))
    assert mesh.cell_data["analytical_mean_stress"][0].size == solved.case.grid.num_cells
    summary = json.loads(paths["summary"].read_text())
    assert summary["solver_package"] == "prototype"
    assert len(summary["comparison_limits"]) >= 4
    values, exact = stress_measures(solved.stress)[0], stress_measures(solved.reference_stress)[0]
    metric = 100.*np.max(abs(values-exact))/np.max(abs(exact))
    assert summary["cases"][0]["mean_stress_max_error_percent_eq41"] == pytest.approx(metric)
