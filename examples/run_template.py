"""Run-script template for the core elastoplastic TPSA solver: every parameter in one place.

Copy this file, edit the PARAMETERS section, and run from the repository root:

    python -m examples.run_template

Nothing is chosen automatically: the linear solver, its options, the predictor,
Newton tolerances, line search and load stepping are all set explicitly below.
Every option is spelled out with its current default value, so changing one is
a one-word edit. Units are whatever you use consistently (SI below).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import porepy as pp

from core.layout import FloatArray, IntArray
from core.linear_solvers import CondensedDirectSolver, DirectSolver, LinearSolver
from core.loading import LoadStepController, LoadStepResult
from core.residual import ResidualAssembler
from core.state import SolutionState
from material.J2 import _materialProperty, vonMisesModel
from material.material_state import MaterialPoint
from material.scalar_hardening import (
    HardeningParameters,
    isotropic_hardening_K,
    kinematic_hardening_H,
)
from utils.load_schedule import AdaptiveLoadOptions, LoadStepError
from utils import meshing
from utils.meshing import Mesh
from utils.newton_utils import BacktrackingOptions, print_newton_convergence

# =============================================================================
# PARAMETERS
# =============================================================================

# --- Mesh ----------------------------------------------------------------------
# Every builder in utils/meshing.py returns a Mesh: the grid, named boundary-face
# groups and TPSA quality measures. Groups: "xmin", "xmax", "ymin", "ymax"
# ("zmin", "zmax" in 3D) for the bounding-box sides; plate_with_hole adds
# "outer" and "hole"; from_grid accepts your own groups.


def build_mesh() -> Mesh:
    return meshing.structured("cartesian", cells=(32, 32), size=(1.0, 1.0))
    # Alternatives (keyword arguments shown with their defaults):
    # return meshing.structured("graded", cells=(32, 32), size=(1.0, 1.0), origin=None, grading=0.6)
    # return meshing.structured("perturbed", cells=(32, 32), size=(1.0, 1.0), perturbation=0.15, seed=0)
    # return meshing.structured("cartesian", cells=(12, 12, 12), size=(1.0, 1.0, 1.0))      # 3D
    # return meshing.simplex_rectangle(cells=(16, 16), size=(1.0, 1.0))   # nearly square blocks
    # return meshing.plate_with_hole(radius=1.0, half_width=4.0, level=1, near_size=None,
    #                                far_factor=4.0, fine_radius_factor=2.5)
    # Any 2D domain bounded by lines and arcs (< 180 degrees each); curve names become
    # the boundary groups. Strictly acute (TPSA-orthogonal) unless acute=False.
    # Example: Cook's membrane (mm), groups "clamped", "bottom", "load", "top".
    # return meshing.planar_domain(
    #     [meshing.Line((0, 0), (48, 44), "bottom"), meshing.Line((48, 44), (48, 60), "load"),
    #      meshing.Line((48, 60), (0, 44), "top"), meshing.Line((0, 44), (0, 0), "clamped")],
    #     size=2.0,                                # number, or function h(x, y)
    #     holes=(), acute=True, max_angle=80.0, min_angle=20.0, smoothing=10)
    # return meshing.from_grid(my_grid, groups={"loaded": my_faces})   # computed PorePy grid


# --- Boundary conditions -----------------------------------------------------
# Boundary groups map to functions of face centers x (shape (3, n)). Functions
# return 3 components (x, y, z); the z row is ignored in 2D. Values are the
# REFERENCE loads reached at load factor 1; the load path scales them.
# Every boundary face is fully Dirichlet or fully Neumann; faces in no group
# below are traction-free. At least one Dirichlet group is needed: with
# tractions only, rigid-body motion makes the Jacobian singular.
Field = Callable[[FloatArray], FloatArray]


def zero(x: FloatArray) -> FloatArray:
    return np.zeros((3, x.shape[1]))


def stretch(x: FloatArray) -> FloatArray:
    """u_x = 0.2 % * x: uniaxial stretch at load factor 1."""
    u = zero(x)
    u[0] = 2e-3 * x[0]
    return u


# Group -> prescribed displacement u(x).
DIRICHLET: dict[str, Field] = {
    "xmin": zero,
    "xmax": stretch,
}

# Group -> traction t(x, n) = sigma n (force per area); n: outward unit normals (dim, n).
# Example, pressure p on a hole: {"hole": lambda x, n: -p * np.vstack((n, np.zeros((3 - len(n), n.shape[1]))))}
NEUMANN: dict[str, Callable[[FloatArray, FloatArray], FloatArray]] = {}


def body_force(x: FloatArray) -> FloatArray:
    """Body force density b (force per volume) at cell centers x (3, n); 3 components."""
    return zero(x)


# --- Material ----------------------------------------------------------------
YOUNG_MODULUS = 210e9
POISSON_RATIO = 0.3
HARDENING = HardeningParameters(
    sigma_y=250e6,       # initial yield stress
    sigma_u=400e6,       # saturation yield stress (= sigma_y for no saturation)
    H_bar=1e9,           # total hardening modulus
    theta=0.5,           # isotropic/kinematic mix: 1 = isotropic only, 0 = kinematic only
    delta=30.0,          # saturation rate (0 = linear hardening)
)

# --- Load path -----------------------------------------------------------------
# Absolute load factors; 1.0 = the reference loads above. Going down unloads.
LOAD_FACTORS = [0.25, 0.5, 0.75, 1.0, 0.6]
ADAPTIVE = True                          # False: exactly LOAD_FACTORS, stop at first failure
ADAPTIVE_OPTIONS = AdaptiveLoadOptions(  # used only if ADAPTIVE
    initial_increment=0.25,              # first increment of load factor
    min_increment=1e-3,                  # give up below this increment
    max_increment=0.25,
    cutback_factor=0.5,                  # increment *= this after a failed step
    growth_factor=1.5,                   # increment *= this after an easy step
    target_iterations=6,                 # "easy" = at most this many Newton iterations
    max_retries=8,                       # cutbacks allowed at one committed state
    max_steps=1000,                      # accepted substeps in one run
)

# --- Newton ------------------------------------------------------------------
MAX_ITERATIONS = 20
# Per block [u, r, p, face]: converged when ||R_block|| <= atol + rtol * ||R_block at start||.
# atol is in residual units (forces for u and face): scale it with your loads.
ATOL = (1e-5, 1e-12, 1e-12, 1e-5)
RTOL = (1e-8, 1e-8, 1e-8, 1e-8)
LINE_SEARCH: BacktrackingOptions | None = BacktrackingOptions(
    factor=0.5,          # step length *= factor per backtrack
    armijo=1e-4,         # sufficient-decrease coefficient
    max_backtracks=12,
)

# --- Starting guess for each load step ------------------------------------------
# "zero": Newton starts from zero increments.
# "elastic": starts from the elastic solution of the increment (needs a solver
#            with solve_elastic: all solvers below). Measured: helps in 3D
#            (Newton iterations 36 -> 25), slower in 2D.
PREDICTOR: Literal["zero", "elastic"] = "zero"

# --- Linear solver -------------------------------------------------------------
# "default"           full Jacobian + UMFPACK inside solve_newton (no solver object)
# "direct"            DirectSolver: same, as a solver object (supports PREDICTOR="elastic")
# "condensed-direct"  CondensedDirectSolver: UMFPACK on the condensed face system
# "krylov"            KrylovSolver (petsc4py) with KRYLOV_OPTIONS below
LINEAR_SOLVER = "krylov"

# Measured choices (examples/benchmark_linear_solvers.py):
#   2D: preconditioner="gamg", matrix_update="in_place", forcing="eisenstat-walker"
#   3D: preconditioner="ilu", matrix_update="rebuild", forcing="eisenstat-walker"
#   simplex meshes: forcing="fixed" (loose EW steps cost Newton iterations there)
KRYLOV_OPTIONS = dict(
    system="condensed",            # "condensed" (face system) or "full" (equilibrated full system)
    method="gmres",                # "gmres", "fgmres", "bcgs"
    preconditioner="gamg",         # "gamg", "ilu", "asm", "jacobi"; never "ilu" in 2D
    ilu_levels=0,                  # ILU fill level
    rtol=1e-3,                     # fixed: tolerance; eisenstat-walker: lower bound
    max_iterations=500,
    restart=200,                   # GMRES restart length
    right_preconditioning=True,    # True: rtol bounds the true residual
    fallback_to_direct=False,      # on Krylov failure, retry with CondensedDirectSolver
    forcing="eisenstat-walker",    # "fixed" or "eisenstat-walker"
    rtol_max=1e-2,                 # EW: tolerance of the first correction of a step
    ew_gamma=0.9,
    ew_alpha=0.5 * (1.0 + 5.0 ** 0.5),
    matrix_update="in_place",      # "rebuild" or "in_place" (condensed only; best with GAMG)
    refresh_interpolation="never", # "never" or "step" (in_place only)
    petsc_options={},              # raw PETSc options, e.g. {"pc_gamg_threshold": "0.02"}
)

# --- Output --------------------------------------------------------------------
PRINT_NEWTON_HISTORY = False             # per-iteration block norms and rates per step
OUTPUT_FILE: Path | None = Path("results/run_template.npz")   # None: do not save


# =============================================================================
# SETUP (normally no edits needed below)
# =============================================================================


def _assigned(groups: dict[str, object], mesh: Mesh, kind: str) -> IntArray:
    faces = [mesh.faces(name) for name in groups]
    every = np.concatenate(faces) if faces else np.zeros(0, dtype=np.int64)
    if np.unique(every).size != every.size:
        raise ValueError(f"{kind} groups {list(groups)} overlap: a face may get one condition only.")
    return every


def build_assembler(mesh: Mesh) -> ResidualAssembler:
    grid = mesh.grid
    dirichlet = _assigned(dict(DIRICHLET), mesh, "Dirichlet")
    if dirichlet.size == 0:
        raise ValueError("No Dirichlet faces: rigid-body motion makes the problem singular.")
    neumann = _assigned(dict(NEUMANN), mesh, "Neumann")
    if np.intersect1d(dirichlet, neumann).size:
        raise ValueError("A face is in both a Dirichlet and a Neumann group.")
    bc = pp.BoundaryConditionVectorial(grid, dirichlet, "dir")
    material = _materialProperty()
    material.young_modulus, material.poisson_ratio = YOUNG_MODULUS, POISSON_RATIO
    points = [
        MaterialPoint(material, HARDENING, vonMisesModel("J2"), isotropic_hardening_K, kinematic_hardening_H)
        for _ in range(grid.num_cells)
    ]
    return ResidualAssembler(grid, bc, points)


def _evaluate(function: Field, x: FloatArray, dim: int, what: str) -> FloatArray:
    values = np.asarray(function(x), dtype=np.float64)
    if values.shape != (3, x.shape[1]) or not np.all(np.isfinite(values)):
        raise ValueError(f"{what} must return finite values of shape (3, {x.shape[1]}).")
    return values[:dim]


def reference_loads(
    mesh: Mesh, assembler: ResidualAssembler,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Dirichlet values, Neumann face forces and cell sources at load factor 1."""
    grid, layout, dim = mesh.grid, assembler.layout, mesh.dim

    # Per face, then picked in the order the controller expects.
    displacement = np.zeros((dim, grid.num_faces))
    for name, function in DIRICHLET.items():
        faces = mesh.faces(name)
        displacement[:, faces] = _evaluate(function, grid.face_centers[:, faces], dim, f"DIRICHLET[{name!r}]")
    dirichlet = displacement[:, layout.dirichlet_faces]

    # Neumann values are face-integrated forces: traction * face area.
    force = np.zeros((dim, grid.num_faces))
    for name, traction in NEUMANN.items():
        faces = mesh.faces(name)
        x, normals = grid.face_centers[:, faces], mesh.outward_normals(faces)
        force[:, faces] = _evaluate(lambda y: traction(y, normals), x, dim, f"NEUMANN[{name!r}]") * grid.face_areas[faces]
    neumann = force[:, assembler.neumann_faces]

    # Cell sources in the momentum rows (u stored cell by cell). The cell residual is
    # sum of face tractions - ... - source, and equilibrium is sum of tractions + b V = 0,
    # so the source is MINUS the cell-integrated body force.
    source = np.zeros(layout.cell_size)
    body = _evaluate(body_force, grid.cell_centers, dim, "body_force") * grid.cell_volumes
    source[layout.u_slice] = -body.ravel(order="F")
    return dirichlet, neumann, source


def build_linear_solver() -> LinearSolver | None:
    if LINEAR_SOLVER == "default":
        if PREDICTOR == "elastic":
            return DirectSolver()   # the built-in path has no solve_elastic of its own
        return None
    if LINEAR_SOLVER == "direct":
        return DirectSolver()
    if LINEAR_SOLVER == "condensed-direct":
        return CondensedDirectSolver()
    if LINEAR_SOLVER == "krylov":
        from core.krylov import KrylovOptions, KrylovSolver   # needs petsc4py

        return KrylovSolver(KrylovOptions(**KRYLOV_OPTIONS))
    raise ValueError(f"Unknown LINEAR_SOLVER {LINEAR_SOLVER!r}.")


def report(step: LoadStepResult, elapsed: float) -> None:
    newton = step.newton
    plastic = np.count_nonzero(step.state.material.alpha > 0.0)
    krylov = int(newton.linear_iterations.sum())
    print(
        f"  factor {step.load_factor:8.4f}  Newton {newton.iterations:3d}  "
        f"backtracks {int(np.count_nonzero(newton.step_length < 1.0)):2d}  "
        f"Krylov {krylov:5d}  plastic cells {plastic:6d}  "
        f"max alpha {step.state.material.alpha.max():.3e}  ({elapsed:.2f} s)",
        flush=True,
    )
    if PRINT_NEWTON_HISTORY:
        print_newton_convergence(newton, extra_block="f")


def main() -> LoadStepController:
    mesh = build_mesh()
    grid = mesh.grid
    assembler = build_assembler(mesh)
    layout = assembler.layout
    print(mesh.summary())
    print(
        f"{layout.size} unknowns ({layout.dirichlet_faces.size} Dirichlet, "
        f"{assembler.neumann_faces.size} Neumann faces); solver {LINEAR_SOLVER}, predictor {PREDICTOR}",
        flush=True,
    )

    initial = SolutionState.zeros(
        layout, num_sides=assembler.operators.num_sides,
        num_neumann_faces=assembler.neumann_faces.size,
    )
    dirichlet, neumann, source = reference_loads(mesh, assembler)
    solver = build_linear_solver()
    try:
        controller = LoadStepController(
            assembler,
            initial,
            dirichlet,
            reference_neumann=neumann,
            reference_cell_source=source,
            max_iterations=MAX_ITERATIONS,
            atol=ATOL,
            rtol=RTOL,
            line_search=LINE_SEARCH,
            linear_solver=solver,
            predictor=PREDICTOR,
        )

        start = last = time.perf_counter()
        try:
            if ADAPTIVE:
                steps = controller.iter_adaptive(LOAD_FACTORS, options=ADAPTIVE_OPTIONS)
            else:
                steps = (controller.advance(factor) for factor in LOAD_FACTORS)
            for step in steps:
                now = time.perf_counter()
                report(step, now - last)
                last = now
        except LoadStepError as error:
            print(f"Stopped: {error}")
        total = time.perf_counter() - start
    finally:
        close = getattr(solver, "close", None)
        if callable(close):
            close()   # frees PETSc objects of a KrylovSolver

    failed = sum(1 for attempt in controller.attempts if not attempt.accepted)
    print(
        f"Reached load factor {controller.load_factor:.4f} in {len(controller.attempts) - failed} "
        f"steps ({failed} rejected attempts), {total:.2f} s.",
    )

    if OUTPUT_FILE is not None:
        state = controller.state
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            OUTPUT_FILE,
            load_factor=controller.load_factor,
            cell_centers=grid.cell_centers[: mesh.dim],
            face_centers=grid.face_centers[: mesh.dim],
            x=state.x,                                   # [u, r, p] per cell
            face_displacement=state.face_displacement,   # (dim, num_faces)
            stress=state.material.stress,                # (3, 3, num_cells)
            plastic_strain=state.material.plastic_strain,
            alpha=state.material.alpha,                  # equivalent plastic strain
        )
        print(f"Saved {OUTPUT_FILE}")
    return controller


if __name__ == "__main__":
    main()
