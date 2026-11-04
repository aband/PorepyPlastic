"""Yarushina benchmark problem."""

from __future__ import annotations

# Standard library: command-line argument
import argparse
from collections.abc import Sequence
from pathlib import Path

# Third-party packages
import numpy as np
import porepy as pp
from porepy.grids.grid import Grid
from dataclasses import field,dataclass

import gmsh

# Core solver
from core.layout import FloatArray, IntArray
from core.loading import LoadStepController, LoadStepResult
from core.residual import ResidualAssembler
from core.state import SolutionState

# Material definition
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import(
    HardeningParameters,
    isotropic_hardening_K,
    kinematic_hardening_H,
)

# Adaptive stepping and Newton diagnostics
from utils.load_schedule import AdaptiveLoadOptions
from utils.newton_utils import BacktrackingOptions, print_newton_convergence
from utils.geometry import preprocess_simplex_grid

# Visualization module
from utils.visualization import (
    export_png,
    export_strain_png,
    export_vtk,
    export_mesh_png,
)

import matplotlib.pyplot as plt

@dataclass(eq=False)
class YarushinaMesh:
    grid: pp.TriangleGrid
    hole_faces: IntArray
    outer_faces: IntArray
    max_nonorthogonality: float
    capped_cells:int

    @property
    def tpsa_admissible(self) -> bool:
        # Positive signed distances are checked during construction
        return self.max_nonorthogonality <= 1e-10

def build_yarushina_mesh(
    *,
    radius: float = 1.0,
    half_width: float = 4.0,
    level: int = 0,
) -> YarushinaMesh:
    """Triangulate [-L, L]^2 minus a circular hole of radius R."""

    if (
        isinstance(level, bool)
        or not isinstance(level, (int, np.integer))
        or level < 0
    ):
        raise ValueError("level must be a nonnegative integer.")

    if (
        not np.all(np.isfinite([radius, half_width]))
        or not 0 < 2.5 * radius < half_width
    ):
        raise ValueError("Require 0 < 2.5*radius < half_width.")

    if gmsh.isInitialized():
        raise RuntimeError(
            "This standalone generator requires an inactive Gmsh session."
        )

    h_near = radius / (2 * 2**int(level))
    h_far = 4 * h_near
    fine_radius = 2.5 * radius

    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("yarushina")

        # Geometry: square minus disk.
        occ = gmsh.model.occ
        square = occ.addRectangle(
            -half_width, -half_width, 0,
            2 * half_width, 2 * half_width,
        )
        disk = occ.addDisk(0, 0, 0, radius, radius)
        surfaces, _ = occ.cut([(2, square)], [(2, disk)])
        occ.synchronize()

        # Constant fine size near the hole, then a gradual increase.
        field = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            field,
            "F",
            f"{h_near} + {h_far - h_near} * "
            f"Min(1, Max(0, (Sqrt(x*x+y*y)-{fine_radius})/"
            f"{half_width - fine_radius}))",
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(field)

        for option in (
            "MeshSizeFromPoints",
            "MeshSizeFromCurvature",
            "MeshSizeExtendFromBoundary",
            "RecombineAll",
        ):
            gmsh.option.setNumber(f"Mesh.{option}", 0)

        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay.
        gmsh.model.mesh.generate(2)

        # Gmsh node tags need not be contiguous or sorted.
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        _, connectivity = gmsh.model.mesh.getElementsByType(
            2, surfaces[0][1],  # Type 2: three-node triangles.
        )

        order = np.argsort(tags)
        points = coordinates.reshape(-1, 3)[order, :2].T
        triangles = np.searchsorted(
            tags[order], connectivity,
        ).reshape(-1, 3).T

    finally:
        gmsh.finalize()

    # PorePy expects counterclockwise triangle connectivity.
    a = points[:, triangles[1]] - points[:, triangles[0]]
    b = points[:, triangles[2]] - points[:, triangles[0]]
    clockwise = a[0] * b[1] - a[1] * b[0] < 0
    triangles[:, clockwise] = triangles[[0, 2, 1]][:, clockwise]

    grid = pp.TriangleGrid(points, triangles)
    grid.compute_geometry()
    _, shift, _ = preprocess_simplex_grid(grid)

    # Identify boundaries using the outer square coordinates.
    boundary = grid.get_all_boundary_faces()
    outer = np.any(
        np.isclose(
            np.abs(grid.face_centers[:2, boundary]),
            half_width,
            rtol=0,
            atol=1e-10 * half_width,
        ),
        axis=0,
    )
    outer_faces = boundary[outer]
    hole_faces = boundary[~outer]

    # Check actual geometry after circumcenter preprocessing.
    incidence = grid.cell_faces.tocoo()
    normals = (
        grid.face_normals[:2, incidence.row]
        * incidence.data[None, :]
        / grid.face_areas[incidence.row][None, :]
    )
    offset = (
        grid.face_centers[:2, incidence.row]
        - grid.cell_centers[:2, incidence.col]
    )
    distance = np.sum(offset * normals, axis=0)
    skew = (
        np.linalg.norm(offset - normals * distance[None, :], axis=0)
        / np.linalg.norm(offset, axis=0)
    )

    if (
        not np.all(np.isfinite(distance))
        or np.any(distance <= 0)
        or not np.all(np.isfinite(skew))
    ):
        raise ValueError("Invalid cell-to-face geometry.")

    return YarushinaMesh(
        grid=grid,
        hole_faces=hole_faces,
        outer_faces=outer_faces,
        max_nonorthogonality=float(skew.max()),
        capped_cells=int(
            np.count_nonzero((shift > 0) & (shift < 1 - 1e-12))
        ),
    )

def assign_boundary(
    mesh: YarushinaMesh,
    *,
    far_field_pressure: float,
    far_field_shear: float = 0.0,
    hole_pressure: float = 0.0,
)-> tuple[pp.BoundaryConditionVectorial, FloatArray]:
    """Return all-Neumann BCs and outward integrated tractions.

    Values have shape (2, grid.num_faces). 
    """
    grid = mesh.grid

    if not np.all(np.isfinite([
        far_field_pressure, far_field_shear, hole_pressure,
    ])):
        raise ValueError("Boundary stresses must be finite.")

    bc = pp.BoundaryConditionVectorial(
        grid, grid.get_all_boundary_faces(), "neu",
    )

    sigma_inf = np.diag([
        -far_field_pressure - far_field_shear,
        -far_field_pressure + far_field_shear,
    ])

    forces = np.zeros((2, grid.num_faces), dtype=np.float64)

    for faces, stress in (
        (mesh.outer_faces, sigma_inf),
        (mesh.hole_faces, -hole_pressure * np.eye(2)),
    ):
        signs, _ = grid.signs_and_cells_of_boundary_faces(faces=faces)

        # PorePy weighted normals
        area_normals = (
            grid.face_normals[:2,faces] * signs[None, :]
        )
        forces[:, faces] = stress @ area_normals

    return bc, forces

def make_material_points(
        num_cells: int,
        gamma: float = 0.3, #poisson_ratio
        E: float = 210.0e9, #young_modulus
    ) -> list[MaterialPoint]:
    material = _materialProperty()
    material.young_modulus = E       # Pa
    material.poisson_ratio = gamma

    parameters = HardeningParameters(
        sigma_y=250.0e6,                  # Pa
        sigma_u=250.0e6,                  # No saturation contribution
        H_bar=1.0e9,                      # Pa
        theta=0.4,                       # Mixed isotropic/kinematic hardening
        delta=0.0,
    )

    return [
        MaterialPoint(
            material=material,
            parameters=parameters,
            model=vonMisesModel("J2"),
            K_law=isotropic_hardening_K,
            H_law=kinematic_hardening_H,
        )
        for _ in range(num_cells)
    ]

def run_mesh(level:int) -> None:

    mesh = build_yarushina_mesh(level=level)

    print(f"Cells: {mesh.grid.num_cells}")
    print(f"Capped cells: {mesh.capped_cells}")
    print(f"Maximum nonorthogonality: {mesh.max_nonorthogonality:.3e}")
    print(f"TPSA-admissible: {mesh.tpsa_admissible}")

    path = export_mesh_png(
        mesh.grid,
        folder_name="results/yarushina/meshes",
        file_name=f"yarushina_triangular_level{level}",
        title=f"Yarushina mesh at level{level}",
    )
    print(path)

if __name__ == "__main__":
    run_mesh(0)
    run_mesh(1)
    run_mesh(2)