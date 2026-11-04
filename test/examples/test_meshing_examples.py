"""utils.meshing versions of the self-convergence and Yarushina examples match the originals."""

import numpy as np
import pytest

from examples import core_self_convergence_meshing as convergence


@pytest.mark.parametrize("kind", convergence.MESH_KINDS)
def test_self_convergence_meshes_are_identical(kind: str) -> None:
    checks = convergence.compare_meshes(kind, (3, 2), L=1.5, H=1.0)
    assert all(checks.values()), checks


def test_self_convergence_run_uses_all_sides_as_dirichlet() -> None:
    result = convergence.run(mesh_kind="cartesian", resolution=(3, 3), load_factors=(0.5,), export_mesh=False)
    grid = result.grid
    np.testing.assert_array_equal(result.assembler.layout.dirichlet_faces, np.sort(grid.get_all_boundary_faces()))
    assert result.controller.load_factor == 0.5 and result.outputs == {}


def test_self_convergence_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        convergence.build_mesh("hexagonal", (2, 2), 1.0, 1.0)


def test_yarushina_boundary_forces_balance() -> None:
    pytest.importorskip("gmsh")
    from examples import yarushina_meshing as yarushina

    mesh = yarushina.build_yarushina_mesh(level=0)
    bc, forces = yarushina.assign_boundary(mesh, far_field_pressure=10.0, far_field_shear=3.0, hole_pressure=2.0)
    assert np.all(bc.is_neu[:, mesh.boundary])
    # A uniform stress on a closed boundary gives zero net force, separately on outer and hole.
    for group in ("outer", "hole"):
        assert np.allclose(forces[:, mesh.groups[group]].sum(axis=1), 0.0, atol=1e-10)
    # Compression on the right side pushes in -x: sigma_xx = -(p + s).
    np.testing.assert_allclose(
        forces[0, mesh.groups["xmax"]].sum(), -(10.0 + 3.0) * 8.0, rtol=1e-12,
    )
    interior = np.setdiff1d(np.arange(mesh.grid.num_faces), mesh.boundary)
    assert not forces[:, interior].any()


def test_yarushina_compare_with_original_when_importable() -> None:
    pytest.importorskip("gmsh")
    from examples import yarushina_meshing as yarushina

    checks = yarushina.compare(0)
    if checks is not None:          # examples/yarushina.py imports
        assert all(checks.values()), checks
