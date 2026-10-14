# PorepyPlastic

An elastoplasticity extension for PorePy.

Constitutive code lives in `material/`: `J2.py`, `material_state.py`,
`scalar_hardening.py`, and `tensor.py`. Import these modules through the package,
for example `from material.material_state import MaterialPoint, MaterialState`.

Shared code lives in `utils/`: problem preparation, total-field state,
cellwise material evaluation, Green–Gauss/Mandel reconstruction, transfer weights,
Newton utilities, fixed/adaptive load schedules, and VTK/PNG exports. The local
analytical and finite-difference tangent helpers live in `material/tangent.py`.
The original formulation lives in `alternatives/original/`, and the coupled
formulation in `alternatives/coupled/`. They reuse the Newton driver and load
controller in `alternatives/newton.py` and `alternatives/loading.py`, alongside
the elastic TPSA setup and face recovery in `alternatives/plane_strain.py` and
`alternatives/postprocessing.py`.

`core/` is the main solver package.
The plastic examples now default to `--formulation retained`. The `original`
and `coupled` formulations remain available through explicit flags, and their
modules are imported only when selected. The mathematical formulation names and
output suffixes have not changed.

Run scripts live in `examples/`: `dirichlet_example.py` for nonuniform loading,
`plastic_plane_strain.py` for the checked homogeneous loading/unloading case,
`elastic_plane_strain.py` for the elastic TPSA baseline, and
`visualize_newton_stress.py` for the local return-map animation. Their shared
setup is in `examples/_solver_setup.py`. Import solver and utility code directly
from its owning package, for example `from core.operators import RetainedFaceOperators`.
Run examples from the repository root:

```bash
python -m examples.dirichlet_example --adaptive --no-export
python -m examples.plastic_plane_strain --no-export
python -m examples.elastic_plane_strain
python -m examples.visualize_newton_stress --help
```

`utils.problem.PlaneStrainProblem.prepare_simulation()` prepares geometry,
material points, and reference Dirichlet data without assembling elastic TPSA
matrices. `alternatives.plane_strain.PlaneStrainTpsa` extends that preparation with
elastic discretization and solving. The retained setup accepts either prepared
case. Shared modules import no solver packages, and retained solver modules
depend on `material/` and `utils/`, without importing `alternatives/`.

`utils.state.as_tpsa_state()` exposes common total fields for reporting;
equilibrium, commit/rollback, and restart still use each complete solver state.
`utils.visualization.export_plastic_history()` accepts records exposing
`load_factor` and `state`, without importing a concrete controller or solver.

The retained-face implementation in `core/` provides incremental
kinematics, nested constitutive updates, one-sided tractions, global residuals,
sparse Jacobians, nested Newton, and load stepping from `Plasticity_revised.pdf`,
Section 4.2. It is the default in both plastic examples:

```bash
python -m examples.dirichlet_example --adaptive
python -m examples.plastic_plane_strain --jacobian finite-difference --no-export
```

To evaluate a candidate directly:

```python
import numpy as np
from utils.problem import PlaneStrainProblem
from utils.state import TpsaState
from core.operators import RetainedFaceOperators
from core.residual import evaluate_global_residual

case = PlaneStrainProblem(cells_per_axis=2)
case.prepare_simulation()
ops = RetainedFaceOperators(
    case.grid, case.bc,
    mu=case.material.isotropic_shear_modulus, lam=case.material.lame_parameter,
)
z = ops.pack(np.zeros(4 * ops.num_cells), np.zeros((2, ops.num_free_faces)))
committed = TpsaState.zeros(case.grid)
evaluation = evaluate_global_residual(ops, case.material_points, committed, z, case.bc_values)
R = evaluation.residual
Q = evaluation.material_trial.stress_correction
delta_traction = evaluation.traction_increment
```

Each call reconstructs total strain from the retained face increments and
committed strain, then reuses `evaluate_material_at_strain` and
`MaterialPoint.update`. The correction is
`Q = sigma_n + C_e : delta_epsilon - sigma_returned`. All local updates use the
explicit `committed` history, including after rejected candidates; the supplied
material points' own committed and trial slots remain untouched. Material
elastic coefficients are checked against the operators cell by cell.

`evaluation.material_trial` is the existing `TpsaMaterialTrial` container with
`epsilon`, `material_states`, and `stress_correction`. Its histories preserve
full 3D stress, plastic strain, backstress, and accumulated plastic strain.
Returned arrays and histories are independent of the inputs. A failed local
return map propagates an exception without partial acceptance. Cell `u,r,p`
remain independent unknowns; only the retained face increments determine strain.
`evaluation.z` and `evaluation.bc_values` store independent snapshots of the
candidate and absolute boundary data; unused interior boundary entries are zero.

`z = [delta_u, delta_r, delta_p, d_free]` stores increments from the committed
load step. Boundary inputs are absolute displacements; the evaluator inserts
target minus committed values. `mu` and `lam` may each be a positive
`(num_cells,)` array. Supplying only `mu` supports traction evaluation; omitting
both leaves the operators available for kinematics only. Traction uses
orthogonal 2D tensor grids with full Dirichlet boundaries, including rotated
rectangles. Rebuild the operators after geometry or elastic coefficient changes.

`delta_traction` has shape `(2, ops.num_sides)`. Column `s` belongs to
`ops.side_faces[s]` and `ops.side_cells[s]` and already uses that cell's outward
normal and face area. Interior faces have two independent columns; their sum is
the face-balance residual. Boundary columns are reaction increments.
`Q` is a symmetric `(3,3,num_cells)` tensor and is applied separately on each
side; only rotation uses complementary cell-to-face weights.

`R = [R_u, R_r, R_p, R_f]` has the same ordering and length as `z`. Cell momentum
sums outward traction increments; rotation and pressure use the same face
displacement increments as strain reconstruction, minus `(V/mu)*delta_r` and
`(V/lam)*delta_p`. Plastic corrections enter through traction only. Residuals
are integrated and unscaled, and Dirichlet faces have no face-balance rows.
The operators expose `rotation_accumulation` and `pressure_accumulation` for
the two cell coefficients.

Pass `body_force_increment=delta_b` to subtract integrated cell force increments
of shape `(2,num_cells)` exactly once, using `div(sigma) = body_force`. These
are load increments, not force densities or a boundary-shifted right-hand side.
Auxiliary sources are zero. Assembly assumes the committed step is equilibrated;
pass its known `committed_residual` of shape `(ops.num_dofs,)` to retain any
previous imbalance. These functions preserve all inputs and do not commit
material history. The lower-level `evaluate_one_sided_traction` and
`assemble_global_residual` remain available for a supplied `Q` or traction.
The high-level result contains traction increments, not total one-sided
traction history or an accepted global state. The load controller below accepts
the converged candidate and accumulates total one-sided traction history.

Construct the Jacobian once per load step, then pass each matching residual
evaluation to it:

```python
from core.jacobian import RetainedAnalyticalJacobian

jacobian = RetainedAnalyticalJacobian(ops, case.material_points, committed)
K = jacobian(z, evaluation)
J_side = jacobian.face_jacobian(z, evaluation)
```

This implements Equation (28), `K = L + P @ D_Q @ B`, keeping all four unknown
blocks. The sparse maps are exposed on the callback: `B` reconstructs cell-major
Mandel strain `[xx, yy, sqrt(2)*xy]`; `L` is the elastic residual derivative;
`P` inserts the cell stress corrections. `D_Q = blockdiag(C_e - C_alg)` is
available through `jacobian.plastic_tangent(z, evaluation)`. The analytical
tangent reuses the residual's returned material histories without new local
return maps. Cellwise constant corrections cancel from cell momentum, so the
nonlinear correction affects only the interior face-balance rows and free-face
displacement columns.

`J_side` has shape `(2*ops.num_sides, ops.num_dofs)` with component-interleaved
rows. It includes both interior sides and Dirichlet reaction derivatives;
`T` and `P_side` expose its fixed elastic and correction maps. Prescribed face
displacements have zero variation during differentiation. Boundary values,
body-force increments, and any committed residual are held fixed.

`RetainedFiniteDifferenceJacobian` has the same interface and approximates only
the local material tangent using the existing forward-difference routine
(`step=1e-10` by default). Both classes copy the committed history; keep their
operators and material parameters fixed. Candidate and reconstructed-strain
checks reject mismatched evaluations. At a yield switch, the analytical tangent
uses the branch selected by the return map.

Solve one target load with all four blocks retained:

```python
from core.newton import solve_newton, BacktrackingOptions
from utils.newton_utils import newton_convergence_rates

result = solve_newton(ops, case.material_points, committed, case.bc_values)
z = result.z
evaluation = result.evaluation
ratios, orders = newton_convergence_rates(result.residual_norms)
```

The default Jacobian is analytical. Supply `jacobian=jacobian` to use a custom
callback, including `RetainedFiniteDifferenceJacobian`. `initial_z` defaults to
zero **load-step increments**, including when the previous state is nonzero.
Each correction solves the full system; every candidate reruns the local return
maps from a snapshot of the same committed history. Boundary data, optional
`body_force_increment`, and optional `committed_residual` are also fixed snapshots.

Convergence requires all four Euclidean block norms `[u,r,p,face]` to satisfy
`atol + rtol * initial_norm`. Defaults are `atol=(1e-5,1e-12,1e-12,1e-5)` and
`rtol=(1e-8,1e-8,1e-8,1e-8)`; pass four entries to customize them. The fourth
column measures interior face balance, and is zero for a single-cell grid.
`max_iterations=20` counts corrections; zero only checks the initial candidate.

Full steps are the default. Enable optional Armijo backtracking with
`line_search=BacktrackingOptions()`; its merit starts with scales equal to the
larger of each block's initial norm and absolute tolerance. The cell-momentum
and face-balance blocks share the larger of their two force scales; zero scales
become one. Convergence still checks each block's own tolerance. Rejected
trials never replace committed material history. `result.step_lengths` records
accepted damping factors; `result.residual_evaluations` includes rejected trials,
while `result.residual_norms` includes only the initial and accepted iterates.
`NewtonConvergenceError` reports iteration or search exhaustion; material,
invalid-data, and linear-solve errors propagate immediately.

The returned `RetainedNewtonResult` owns a converged candidate, including total
strain and material histories but incremental cell/face unknowns and tractions.
No input state or `MaterialPoint` slot is committed by Newton. The calling load
controller accepts the full state together and accumulates histories after success.
The implementation shares tolerance validation, block norms, linear-solve checks,
and convergence-rate calculation with the existing Newton solver.

Use `core.loading.LoadStepController` to advance the retained
formulation through prescribed proportional loads:

```python
from core.state import RetainedState, assemble_state_residual
from core.loading import LoadStepController
from utils.load_schedule import AdaptiveLoadOptions

controller = LoadStepController(
    ops, case.material_points, RetainedState.zeros(ops), case.bc_values,
)
steps = controller.run([0.25, 0.5, 1.0, 0.99, 1.1])
state = controller.state
total_cell_fields = state.tpsa.x
total_face_displacement = state.tpsa.face_displacement
total_side_traction = state.side_traction
residual = assemble_state_residual(ops, state)

# Optional adaptive continuation from the last accepted state:
more_steps = controller.run_adaptive([1.2], options=AdaptiveLoadOptions())
```

`RetainedState` wraps the existing `TpsaState`, adding total outward integrated
`side_traction` of shape `(2,num_sides)` and absolute integrated `body_force` of
shape `(2,num_cells)`. Its constructor validates and copies the complete state,
including agreement of stored face displacement with boundary data and total
Green–Gauss strain. `state.tpsa.traction` is a derived signed average in the grid's
normal orientation for compatibility and visualization. Equilibrium always uses
both stored sides, so averaging cannot erase an interior imbalance.

Targets are `factor*reference_bc_values` and `factor*reference_body_force`
(zero body force by default). Newton receives absolute boundary targets, the
body-force **difference from the accepted load**, and the old residual assembled
from total histories. After convergence, the controller adds cell/face and side
traction increments exactly once, adopts the returned total strain/material
histories, and checks equilibrium of the accumulated state before acceptance.
`MaterialPoint` slots remain untouched; the controller owns the accepted histories.

`advance(factor)` accepts one step; `run(factors)` accepts a fixed schedule.
Increasing, decreasing, repeated, and negative factors are supported. Adaptive
`iter_adaptive` and `run_adaptive` reuse the existing cutback/growth scheduler;
repeated adaptive targets are skipped. Only `NewtonConvergenceError` triggers
cutback. Every retry builds a fresh Jacobian from the last accepted history.
Pass the same Newton tolerances, iteration limit, and optional `line_search` to
the controller. To use the numerical tangent, pass
`jacobian_factory=lambda old: RetainedFiniteDifferenceJacobian(ops, case.material_points, old)`
after importing it from `core.jacobian`.

`state`, `steps`, and returned `RetainedLoadStepResult` records are independent
snapshots; each record contains its factor, accumulated state, and incremental
Newton diagnostics. Failed attempts retain the previous accepted state and
appear in `attempts`, with the underlying exception chained through `LoadStepError`.
Restart by supplying `controller.state` and `initial_load_factor=controller.load_factor`
to a new controller with the same operators, material parameters, and reference
loads. Both the stored boundary values and stored body force must match that factor.

The first global example is a small-strain, static, **elastic TPSA plane-strain
problem** on an unfractured 1 m by 1 m Cartesian square. Run it from this directory
in an environment with PorePy installed:

```bash
python -m alternatives.plane_strain
```

The example writes `results/vtu/displacement_2.vtu` and
`results/pvd/displacement.pvd` using `pp.Exporter`, including the reconstructed
strain tensor. It also writes three strain component PNGs and one principal-strain
PNG, described below. The reusable `export_png`
function shown below writes `results/displacement.png` using PorePy's
native `pp.save_img`. The PNG shows displacement vectors at cell centers on the undeformed mesh,
using `vector_value` in `pp.save_img`. Arrow direction and relative length
represent the vector field; a common scale factor is printed below the plot.
The title is "Displacement", with SI coordinate labels and equal spatial scales. Open the `.pvd` file in
ParaView and click **Apply**. Select `displacement_magnitude`, `rotation_stress`,
or `total_pressure` under **Color By**. The three-component `displacement`
vector has zero out-of-plane displacement. For a deformed view, apply **Cell
Data to Point Data**, followed by **Warp By Vector** using `displacement` and
an appropriate scale factor; the exported mesh itself is undeformed.

Choose a different output directory with:

```bash
python -m alternatives.plane_strain --output-dir /tmp/plane_strain_vtk
```

Remove the project's entire `results/` directory, including VTK, PVD, and PNG
outputs, with:

```bash
./clean_results.sh
```

The script locates `results/` relative to itself, so it can be called from any
working directory. It also succeeds when `results/` is already absent. Exporters
recreate the directory on the next run. Custom output directories are not cleaned.

The reusable visualization module is `utils/visualization.py`:

```python
from alternatives.plane_strain import PlaneStrainTpsa
from alternatives.postprocessing import TpsaPostprocessing
from utils.visualization import export_png, export_strain_png, export_vtk

case = PlaneStrainTpsa()
x = case.solve()
epsilon = TpsaPostprocessing(case).strain_green_gauss()
pvd = export_vtk(case.grid, x, epsilon=epsilon, folder_name="results")
strain_pngs = export_strain_png(case.grid, epsilon, folder_name="results")
png = export_png(case.grid, x, folder_name="results")
```

`export_vtk(..., epsilon=epsilon)` adds dimensionless cell fields `strain`,
`strain_xx`, `strain_yy`, `strain_xy`, and `strain_zz`. The full `strain` field has
nine components in row-major tensor order: xx, xy, xz, yx, yy, yz, zx, zy, zz.
The shear component is tensor strain, so engineering shear is twice `strain_xy`.
Omitting `epsilon` preserves the original displacement/rotation/pressure export.

`export_strain_png()` returns a dictionary with keys `xx`, `yy`, `xy`, and
`principal`, pointing to:

- `plane_strain_epsilon_xx.png`
- `plane_strain_epsilon_yy.png`
- `plane_strain_epsilon_xy.png`
- `plane_strain_principal_strain.png`

The component maps use PorePy's `pp.save_img` on the undeformed mesh and share
one color range centered on zero. The principal-strain plot overlays centered
segments in the two in-plane eigenvector directions. Lengths are proportional
to absolute eigenvalues with a common scale; red means extension and blue means
contraction. Equal principal strains appear as circles, whose diameter follows
the same scale, because there is no preferred direction. Zero strain produces
no glyph. Only roundoff relative to the largest principal strain is suppressed
in this plot; the VTK tensor retains the supplied values. The default example
shows uniform horizontal extension.

The solved data are available as class members. `solve()` still returns the
solution vector and also stores it in `case.x`. Preparation exposes `case.bc`
(the PorePy boundary-condition object) and `case.bc_values` (shape
`(2, num_faces)`), sharing storage with the flat `case.bound_vec`.

Supply `boundary_displacement` to prescribe spatially varying displacement on
all boundary faces. The callback receives a **copy** of boundary-face centers
with shape `(3, number_of_boundary_faces)` in
`grid.get_all_boundary_faces()` order. The third coordinate is zero for this
plane-strain grid. Return finite real displacements of shape
`(2, number_of_boundary_faces)`; coordinates and displacements are in metres.

```python
import numpy as np
from numpy.typing import NDArray
from alternatives.plane_strain import PlaneStrainTpsa


def displacement(centers: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y = centers[:2]  # Unit-square coordinates in metres.
    return 1e-4 * np.vstack((x * (1 + 0.2 * np.sin(np.pi * y)), np.zeros_like(x)))


case = PlaneStrainTpsa(boundary_displacement=displacement)
x = case.solve()  # Elastic solution with the custom boundary data.
```

`case.bc_values_displacement(grid)` follows PorePy's displacement-value naming,
using the standalone TPSA grid rather than a `BoundaryGrid`. It inserts values
on boundary faces, keeps interior entries zero, and returns a fresh flat vector
ordered `[ux_face0, uy_face0, ux_face1, uy_face1, ...]` (`ravel("F")`). There is no
face-area scaling. Boundary types remain full Dirichlet, set separately through
`_set_uniform_bc()`. Callback output is copied into the boundary storage.

Omitting the callback preserves `displacement_gradient @ x + translation` and
all existing affine reference solutions. Explicit `displacement_gradient` or
`translation` cannot be combined with a callback. With a callback, the
`reference_*()` methods and the plastic `homogeneous_reference()` raise an error:
a prescribed boundary function alone does not define an exact interior solution,
even if that function happens to be affine. The existing
`plastic_plane_strain.run_example()` remains the verified homogeneous benchmark.

For a custom plastic case, call `case.prepare_simulation()` and pass
`case.bc_values` as `reference_bc_values` to `LoadStepController`, as described
below. Both fixed and adaptive stepping apply `g = load_factor * reference_bc_values`
to the entire spatial pattern, including any translation. The callback is
sampled once per preparation, not during Newton iterations or load steps.
Changing a callback's captured parameters requires re-preparing the case and
creating a new controller; existing controllers retain their copied reference
loads. Re-preparation also resets the case geometry and material points.

A complete runnable example of this workflow is `examples/dirichlet_example.py`:

```bash
python -m examples.dirichlet_example
python -m examples.dirichlet_example --adaptive
python -m examples.dirichlet_example --adaptive --jacobian finite-difference --no-export
```

Both this example and `examples.plastic_plane_strain` accept an independent
`--formulation` option (Python: `run_example(formulation="retained")`). The
selection is retained in `result.formulation` and printed with verbose output.

| Formulation | Status | Newton unknowns |
|---|---|---|
| `original` | Implemented, with either Jacobian type | `u, r, p` |
| `coupled` | Updated complementary-weight formulation, with either Jacobian type | `u, r, p, strain` |
| `retained` (default) | Retained face displacement with nested material updates, with either Jacobian type | `du, dr, dp, d_free` |

For example:

```bash
python -m examples.dirichlet_example --formulation coupled --jacobian analytical --adaptive
python -m examples.plastic_plane_strain --formulation coupled --jacobian finite-difference --no-export
```

The `original` option preserves the previous equations for comparison.
The `coupled` option implements equations (4.7), (6.7), and (7.2) of
`Plasticity_TPSA_weights_marked_full.pdf`. It solves seven unknowns per cell:
two displacement components, rotation stress, pressure, and three independent
in-plane strain components. All three options support fixed and adaptive load
steps, loading/unloading, and VTK/PNG output. The homogeneous runner checks the
analytical J2 reference after every accepted step; the nonuniform example has no
exact interior reference.

The `retained` option connects `core/` to the same boundary
callback, material points, load targets, and exports. It solves increments with
`4*num_cells + 2*num_free_faces` unknowns, using full Dirichlet boundary data on
the supported orthogonal tensor grids. Constitutive updates are nested inside
global Newton; this option does not select the Newton–Schur algorithm.
`--jacobian analytical` uses its consistent retained-face Jacobian, while
`--jacobian finite-difference` approximates the local constitutive tangent.
The default is `retained`; explicitly select `original` or `coupled` for comparisons.

For retained runs, `result.controller.state` and each accepted `record.state`
are `RetainedState`: `.tpsa` contains total cell fields, strain, material history,
and face displacement; `.side_traction` retains both outward numerical forces;
`.body_force` stores the total integrated load. `record.newton.z` contains the
increment solved at that step. The homogeneous checks reassemble all four total
residual blocks using the side histories. Newton output labels the fourth block
`R_f` and reports its convergence rates. Jacobian PNGs include the free-face block
and use the peak increment with the preceding committed history. Cell-field and
history exports use total fields, with filenames containing `_retained_<jacobian>`.
To plot a retained matrix directly, call
`export_jacobian_png(matrix, formulation="retained", num_cells=grid.num_cells)`.

The coupled implementation is in `alternatives/coupled/residual.py` and
`alternatives/coupled/jacobian.py`. For an interior face with one-sided elastic
stiffnesses `a_i = 2*mu_i/d_i`, it uses displacement weights `a_i/(a_i+a_j)`
and complementary traction weights `a_j/(a_i+a_j)`. Boundary traction weights
are one. These weights use fixed elastic coefficients. The plastic correction
`Q = sigma_n + C_e:(epsilon-epsilon_n) - sigma_trial` contributes to both
face traction and shared face displacement. The latter includes
`(Q_i-Q_j) n/(a_i+a_j)` as well as the reference displacement/pressure terms.
Rotation and pressure balances use that same face displacement. Prescribed
boundary displacement is exact and has zero Newton variation.

Strain is encoded cell by cell as Mandel components `[xx, yy, sqrt(2)*xy]`.
The fourth residual enforces symmetric Green–Gauss reconstruction from the
shared face displacement. With `B` the reference reconstruction map,
`H_epsilon` its plastic feedback, and `H` all three mechanical plastic-source
maps, the sparse monolithic Jacobian is

```text
J = [ A          H D_Q             ]
    [ -B         I - H_epsilon D_Q ]
D_Q = blockdiag(C_e - C_alg)  # in matching Mandel coordinates
```

`CoupledAnalyticalJacobian` uses the consistent J2 return-map tangent;
`CoupledFiniteDifferenceJacobian` approximates only the local material tangent
and assembles the same global chain rule. Strain is solved simultaneously with
`u,r,p`; no auxiliary elimination or strain fixed-point iteration is used.
Stress, plastic strain, backstress, and accumulated plastic strain retain the
existing full 3D plane-strain material histories.

For lower-level use, construct `operators = CoupledOperators(case)`, select
`transfer = operators.transfer`, and initialize
`state = TpsaState.zeros(case.grid, store_faces=True)`. Pass those to the existing
`LoadStepController`; its default Jacobian dispatches to the coupled finite
difference class. For analytical derivatives, supply
`jacobian_factory=lambda old: CoupledAnalyticalJacobian(operators, transfer, case.material_points, old)`.
Returned `state.x` keeps its existing `4*nc` layout; `operators.unknowns(state)`
packs the full `7*nc` Newton vector. Restart states must retain
`state.face_displacement`, since plastic face history cannot be recovered with
the elastic interpolation alone. Acceptance and rollback include this field.

Newton checks four residual norms independently and reports `||R_eps||`.
Three-entry tolerances append strain defaults `atol=1e-12`, `rtol=1e-8`;
four-entry tuples set all tolerances explicitly. For standalone heatmaps, pass
`export_jacobian_png(matrix, formulation="coupled")` to label all four blocks.
The example runners select this automatically.

Scope: full displacement Dirichlet conditions on Cartesian tensor-product grids
with cell/face centroids, positive finite elastic coefficients, and the retained
TPSA rotation closure at interpolation parameter zero. Unsupported geometry or
inconsistent material coefficients/transfer weights are rejected. Tests in
`test/alternatives/coupled/test_formulation.py` cover one-sided traction elimination, full
residual derivatives, heterogeneous stiffness, elastic recovery, plastic
loading/unloading, restart, cutback, and exports. These checks do not establish
spatial convergence. The detailed original-formulation APIs below continue to
use `TpsaOperators`, `AnalyticalJacobian`, and `FiniteDifferenceJacobian`.

Run these commands from the repository root. The Dirichlet example CLI uses a 4-by-4 grid,
zero body force, and the default J2 material. On every boundary face it prescribes
`u_x = factor * 0.004 m * (x/L) * [1 + 0.2 sin(pi*y/L)]`, `u_y = 0`, with
`L = 1 m`. Edit `boundary_displacement()` in the example to change this spatial
pattern; every face retains full Dirichlet constraints. The fixed schedule has
20 loading steps to factor 1 and one unloading step to 0.98. `--adaptive` instead
uses these two turning points with the default `AdaptiveLoadOptions`.
`--cells-per-axis` changes mesh resolution. Its Python `run_example()` default
is 10 cells per axis; supply `cells_per_axis` to choose another size.

The console prints Newton residuals, convergence-rate estimates, and counts of
cells with accumulated plastic strain. This nonuniform problem uses no affine
reference solution or L2 error report. The VTK series contains every accepted
state, including displacement, strain, stress, and material history. PNGs show
the stress/strain and accumulated-plastic-strain histories, final displacement,
final strain components and principal directions, and the Jacobian at converged
peak loading. Final field plots represent the state after unloading to 0.98.

Outputs use the prefix `dirichlet_example_<formulation>_<jacobian>` under `results`, with
VTUs in `results/vtu` and PVD collections in `results/pvd`. For example, open
`results/pvd/dirichlet_example_original_analytical.pvd` in ParaView. Change the destination
with `--output-dir`; `--no-export` suppresses all file output. Fixed and adaptive
runs with the same formulation and Jacobian share filenames, so use separate output directories
to retain both. Python callers can use `run_example()` from this module, which
returns `.case`, `.controller` (including accepted states), `.formulation`, and `.outputs`;
files are written only when `output_dir` is supplied.

```python
from alternatives.plane_strain import PlaneStrainTpsa
from alternatives.postprocessing import TpsaPostprocessing

case = PlaneStrainTpsa()
case.solve()
post = TpsaPostprocessing(case)

u = post.u_cell                      # (2, num_cells)
r, p = post.r, post.p                # rotation stress and total pressure
stress_matrix = post.matrices["stress"]
bc = post.bc                        # is_dir, is_neu, is_rob, etc.
boundary_values = post.bc_values     # (2, num_faces); also post.u_boundary
boundary_faces = post.boundary_faces

epsilon = post.strain_green_gauss()  # (3, 3, num_cells), total plane strain
u_face = post.u_face                 # (2, num_faces), recovered displacement
grad_u = post.grad_u                 # (2, 2, num_cells), du_i/dx_j
```

`post` also exposes `grid`, `x`, the PorePy data dictionary `d`,
`bound_vec`, and the assembled `face_discretization`, `rhs_matrix`, `div`,
and `accum` operators. These members share the data from that solve.
Re-preparing the case clears `case.x`; create a new postprocessing instance
after solving again. Existing instances continue to reference their previous
solve. Green–Gauss reconstruction currently supports this 2D Cartesian case
with prescribed displacement on every boundary face.

`reconstruct_face_displacement()` recovers normal and tangential displacement
from the stored TPSA mass and rotation flux operators. This retains the
material/distance weighting and pressure-jump correction on interior faces;
boundary faces use the prescribed displacement. `gradient_green_gauss()` sums
face displacement times outward, area-weighted normals and divides by cell
area. `strain_green_gauss()` symmetrizes this gradient and embeds it in a 3D
tensor with zero out-of-plane strain. Shear entries are tensor shear strains,
so `epsilon_xy = (du_x/dy + du_y/dx) / 2`.

Each method recomputes its inputs through the preceding stages. The result
members `u_face`, `grad_u`, and `epsilon` start as `None`; recompute after
changing shared inputs in place. Recomputing an earlier stage clears later
results. Reconstruction does not update material histories. It returns total
strain; a future material update needs the increment from the last converged
load step. Least-squares reconstruction and the method comparison remain to
be implemented.

The exported fields are numerical cell data: displacement and its magnitude
in metres, rotation stress and total pressure in pascals, and reconstructed total
strain as dimensionless tensors and components. A numerical cell
stress tensor is not yet reconstructed, so analytical reference stresses and
initial material histories are not exported as simulation results. This
example remains **plane strain**, with nonzero out-of-plane stress allowed.

The default 4 by 4 grid has 64 global unknowns: two displacement components, one
rotation stress, and one TPSA total pressure per cell. Affine displacement
`u_x = 1e-4 * x, u_y = 0` is prescribed on every boundary, with zero body force.
This is constrained extension, so transverse stresses need not vanish. The
example compares the numerical fields, reconstructed Green–Gauss strain, and
integrated face tractions against
the exact homogeneous elastic solution. The L2 error report is hidden by default;
use `python -m alternatives.plane_strain --show-l2-errors` to print it.
Reported errors are **absolute discrete
L2 norms**, computed with PorePy's `ConvergenceAnalysis.lp_error(p=2,
relative=False)`. Cell fields use cell-area weights,
`sqrt(sum_K(area_K * ||numerical_K - exact_K||^2))`; integrated face forces use
PorePy's face dual-area weights. Both displacement components contribute to
one vector norm; strain uses the tensor Frobenius norm, including both
symmetric shear entries. Absolute norms remain defined for zero reference fields.
In this 2D problem the area weights add a metre to the norm units: displacement
errors are in m², strain errors in m, pressure and rotation-stress errors in
Pa·m, and errors of face force per thickness in N. Reconstructed strain is
available through `post.epsilon`, the console error report, VTK fields, and PNGs.

Material constants use SI units: Young's modulus is 210 GPa, Poisson's ratio is
0.3, and the initial J2 yield stress is 250 MPa. Plane strain sets the out-of-plane
total strains to zero while retaining the full 3D stress and history tensors.
In this TPSA formulation, total pressure is `lambda * div(u)`; it is not the mean
Cauchy stress. PorePy's 2D rotation stress uses the convention
`mu * (du_x/dy - du_y/dx)`.

`PlaneStrainTpsa` in `alternatives/plane_strain.py` allocates one independent
`MaterialPoint` per cell during case preparation. These histories remain at
their initial values during the elastic global solve. Green–Gauss strain
reconstruction is implemented. Work on the global residual starts with the
`TpsaState` container in `utils/state.py`. It stores `x`, `bc_values`,
`epsilon`, `material_states`, and integrated numerical face `traction`. The
`reference_*` methods supply analytical elastic values for verification only.

```python
from utils.state import TpsaState

case = PlaneStrainTpsa()
case.prepare_simulation()
state_n = TpsaState.zeros(case.grid)  # undeformed state, including zero old boundary data
trial = state_n.copy()               # independent arrays and per-cell material histories
```

The prepared case contains the target boundary displacement; `state_n.bc_values`
belongs to the old converged state and starts at zero. Construction validates and
copies the supplied state arrays and each material history. Total strain has
shape `(3, 3, num_cells)` with zero out-of-plane components; material stress and
plastic history retain all 3D components. `x` follows the existing `[u, r, p]`
ordering, and `traction` has shape `(2, num_faces)` in PorePy's fixed normal
orientation. Keep `state_n` unchanged during trial evaluations. The container is
mutable and provides independent storage, not automatic convergence or commit
logic. Nonzero initial states require compatible supplied data.

`TpsaOperators` stores copies of the fixed TPSA matrices, their cell blocks,
the traction operators `T` and `T_g`, and the force divergence `D_u`. It factors
the rotation and pressure blocks once, including the pressure stabilization:

```python
from alternatives.original.residual import TpsaOperators

operators = TpsaOperators(case)  # after prepare_simulation()
rhs = operators.assemble_rhs(case.bc_values)  # once per target load
x_trial = state_n.x.copy()  # current coupled Newton candidate [u, r, p]
```

Trial evaluation accepts all three candidate fields without solving auxiliary
equations internally. The existing elastic example still solves them together.
For an optional reduced formulation, `operators.solve_auxiliary(u_trial, rhs)`
recovers fields satisfying `A_rr @ r = rhs_r - A_ru @ u_trial` and
`A_pp @ p = rhs_p - A_pu @ u_trial`; force equilibrium is still to be solved.
The right-hand side is `sources - div @ rhs_matrix @ boundary_values`.
Optional `sources` contains integrated body-force, rotation, and pressure sources
in the same block ordering as `x`, and defaults to zero. Only boundary entries
of `bc_values` are used. The matrices and grid must remain fixed; rebuild the
operators after changes to geometry, elastic coefficients, or boundary types.
This stage supports full Dirichlet boundaries and finite positive elastic
coefficients. Inputs and material histories remain unchanged by these solves.

`evaluate_material_trial` reconstructs total Green–Gauss strain from a candidate
`[u, r, p]`, applies each cell's material model to the strain increment from
`state_n`, and returns a `TpsaMaterialTrial`:

```python
import numpy as np
from alternatives.original.residual import evaluate_material_trial

material_trial = evaluate_material_trial(
    operators, case.material_points, state_n, x_trial, case.bc_values,
)
# material_trial.epsilon, .material_states, .stress_correction
```

Each call starts from the supplied committed history and preserves the case's
material-point histories, including any existing trials. The returned histories
are independent. The full 3D stress correction is the elastic predictor minus
returned stress for this increment; it is zero during elastic unloading even
when the committed state contains plastic strain. The reconstruction helpers
in `alternatives/postprocessing.py` also serve the existing solved-case interface.
The numerical and history-isolation tests are in `test/alternatives/original/test_material_trial.py`.

`CellToFaceTransfer` in `utils/transfer.py` supplies a separate, configurable
transfer of cell stress corrections to faces. It stores a sparse matrix `W` of
shape `(num_faces, num_cells)` and applies it to every tensor component:

```python
from utils.transfer import CellToFaceTransfer

transfer = CellToFaceTransfer(case.grid)
face_correction = transfer.apply(material_trial.stress_correction)  # (3, 3, nf)
W = transfer.weights  # independent sparse matrix copy
```

The default uses equal interior weights and the single adjacent cell on boundary
faces. Change only the weight rule to explore another transfer. For example,
an inverse-distance rule on a grid with computed geometry can be supplied as:

```python
def distance_weights(grid, face, cells):
    distances = np.linalg.norm(
        grid.cell_centers[:, cells] - grid.face_centers[:, face, None], axis=0,
    )
    weights = 1.0 / distances  # requires strictly positive distances
    return weights / weights.sum()

transfer = CellToFaceTransfer.from_rule(case.grid, distance_weights)
```

Rules receive adjacent cell IDs in ascending order, run only on interior faces,
and return weights in that order. Fixed coefficient arrays can be captured by
the rule. Alternatively, supply an explicit sparse matrix with
`CellToFaceTransfer(case.grid, weights=W)`. Weights must be finite, nonnegative,
use adjacent cells, and sum to one on each face; invalid weights raise an error
rather than being silently normalized. Neither face areas nor orientation signs
are included. One shared tensor is produced per face.

Weights are built once and remain fixed during Newton iterations. The matrix
property returns a copy, so edits take effect only when supplied to a new
transfer object. Changing a rule therefore does not require changes to material
evaluation or traction/residual assembly. The same stored matrix
are used by both the analytical and finite-difference Jacobians. These transfer choices
still require spatial verification for the intended grid and coefficients.

`evaluate_face_traction` in `alternatives/original/residual.py` updates the numerical face
traction for the same candidate and material-trial result:

```python
from alternatives.original.residual import evaluate_face_traction

traction = evaluate_face_traction(
    operators, transfer, state_n, x_trial, case.bc_values, material_trial,
)  # (2, num_faces), integrated force per unit out-of-plane thickness
```

It evaluates `t_n + T @ (x - x_n) + T_g @ (g - g_n) - Q_face @ N_face`, with
vector matrix products reshaped in Fortran order. Here `x` is the complete cell
vector `[u, r, p]`, and `g` is the prescribed boundary displacement, flattened in
the same interleaved component order. PorePy's `N_face` already includes face
area, so it is used exactly once. The in-plane block of `Q_face` is contracted
with this normal; full 3D material data remain unchanged. The divergence later
supplies the orientation signs for each adjacent cell.

Both old and target boundary arrays use only boundary-face entries. The stored
`t_n` carries previous plastic loading and is never recomputed from a cell-stress
average. Trial evaluation returns an independent array and does not solve,
commit, or overwrite any cell field or material history. The caller must use
operators and transfer weights with matching grid ordering, and a material trial
computed from the same candidate and committed state. Tests cover elastic
recovery, boundary increments, custom transfer, shared-face forces, and plastic
loading/unloading in `test/alternatives/original/test_face_traction.py`.

The fully coupled global residual is now available through
`evaluate_global_residual` in `alternatives/original/residual.py`. It evaluates the material
response and numerical face tractions for the supplied candidate, then assembles
all three equation blocks:

```python
from alternatives.original.residual import evaluate_global_residual

result = evaluate_global_residual(
    operators, transfer, case.material_points, state_n, x_trial, case.bc_values,
)
R = result.residual            # (4 * num_cells,), ordered [R_u, R_r, R_p]
candidate = result.trial       # independent TpsaState for exactly this evaluation
Q = result.stress_correction   # (3, 3, num_cells), incremental cell correction
```

For `f = b - div @ rhs_matrix @ g`, the assembled equations are
`R_u = D_u @ t - b_u`, `R_r = A_ru @ u + A_rr @ r - f_r`, and
`R_p = A_pu @ u + A_pp @ p - f_p`. All displacement components are interleaved
by cell, as in `x`. No cell unknowns are eliminated, and no auxiliary equations
are solved inside evaluation. A current candidate may have nonzero residuals
in any of the three blocks.

Optional `sources=` supplies the absolute target integrated source vector
`[body force, rotation source, pressure source]`, with shape `(4 * num_cells,)`;
it defaults to zero. Supply cell-integrated quantities, not source densities,
load increments, or the boundary-shifted `rhs`. Momentum subtracts the original
body force because the numerical traction already contains boundary terms.
The auxiliary equations use the target boundary-adjusted right-hand side.

For tractions already evaluated for the same candidate and load, use the
lower-level `assemble_global_residual(operators, x_trial, case.bc_values,
traction, sources=...)` to obtain just the residual vector. The high-level
result also retains candidate displacement/rotation/pressure, sanitized boundary
data, total strain, independent material histories, and numerical traction.
Neither function accepts a load step or mutates committed history. A failed
material update propagates without partially committing any cell.

The residual is raw and unscaled: momentum and auxiliary blocks have different
physical units. The coupled residual tests are in `test/alternatives/original/test_global_residual.py`.

`solve_newton` in `alternatives/newton.py` now solves one target load with a required
Jacobian callback. For an elastic load, the existing TPSA matrix is the reference
Jacobian:

```python
from alternatives.newton import solve_newton

result = solve_newton(
    operators, transfer, case.material_points, state_n, case.bc_values,
    jacobian=lambda x, evaluation: operators.A,
)
candidate = result.trial             # independent converged TpsaState
iterations = result.iterations       # number of Newton corrections
norms = result.residual_norms         # rows: evaluations; columns: momentum, rotation, pressure
from utils.newton_utils import newton_convergence_rates, print_newton_convergence
ratios, orders = newton_convergence_rates(norms)  # same shape as norms, per block
print_newton_convergence(result)      # print this load step's iteration history
# state_n = candidate                # accept explicitly when ready to advance
```

The callback receives the current complete `[u, r, p]` vector and its
`TpsaResidual` evaluation. Return a finite dense or sparse Jacobian of shape
`(4 * num_cells, 4 * num_cells)`; treat both callback arguments as read-only.
The evaluation provides reconstructed strain, returned material histories,
tractions, and stress corrections. Fixed operators or committed history needed
by a consistent Jacobian are captured by the callback. `operators.A`
is exact only while all trial material responses remain elastic; it is not a
consistent plastic Jacobian.

Each iteration evaluates the full residual from the same committed history,
solves `J @ delta_x = -R`, and updates all three fields together. The initial
guess defaults to `state_n.x`; override it with `initial_x=`. Boundary values,
sources, committed state, and the initial guess are copied for the solve.
Operators, transfer weights, and material model parameters must remain fixed.
The optional `sources=` vector uses the same absolute, integrated target-load
convention as `evaluate_global_residual`.

Convergence requires every block to satisfy
`norm(R_block) <= atol_block + rtol_block * norm(R_initial_block)`.
These are Euclidean norms of integrated residual entries, distinct from the
volume-weighted L2 field errors reported by the elastic example. The default
`atol=(1e-5, 1e-12, 1e-12)` uses each block's units and suits the SI example;
adjust it for other scales. The default `rtol=(1e-8, 1e-8, 1e-8)` is
dimensionless. Entries follow `[momentum, rotation, pressure]`, with fixed
initial reference norms. `result.thresholds` stores the actual limits, and
`result.evaluation` retains the final residual and stress correction.

`max_iterations=20` limits corrections; zero permits only an initial convergence
check. An already equilibrated guess returns with `iterations=0` without calling
the Jacobian. Exceeding the limit raises `NewtonConvergenceError`; invalid
Jacobians, failed linear solves, non-finite results, and material-update errors
also abort without accepting a candidate. The supplied committed state and the
case's material-point histories, including existing trials, remain unchanged on
success and failure. A result is returned only after all blocks converge, and
accepting it remains the caller's responsibility.

Tests in `test/alternatives/test_newton.py` cover elastic recovery with dense and sparse
Jacobians, individual block convergence, integrated sources, repeated evaluation
from fixed plastic history during elastic unloading, and failure isolation.
This driver takes full Newton steps. Residual scaling and line search remain
future work. Optional adaptive load increments are provided by
`LoadStepController`, described below. Least-squares reconstruction remains deferred.
`plane_strain.py` continues to run the elastic baseline.

The analytical Jacobian implements the specified residual in
[Plasticity_Dirichlet_marked.pdf](Plasticity_Dirichlet_marked.pdf), especially
Eqs. (12)–(13), (24)–(26), and the full-Dirichlet completion (D1)–(D13).
Use `AnalyticalJacobian` in `alternatives/original/jacobian.py`:

```python
from alternatives.original.jacobian import AnalyticalJacobian

jacobian = AnalyticalJacobian(operators, transfer, case.material_points, state_n)
result = solve_newton(
    operators, transfer, case.material_points, state_n, case.bc_values, jacobian,
)
J = jacobian(result.trial.x, result.evaluation)  # (4*nc, 4*nc), all [u,r,p]
J_face = jacobian.face_jacobian(result.trial.x, result.evaluation)  # (2*nf, 4*nc)
```

The coupled assembly is
`J_face = T - N @ (W kron I_4) @ blockdiag(C_e - C_alg) @ B`, followed by
`J = [D_u @ J_face; A_r; A_p]`. Here `B` differentiates the reconstructed cell
**gradient**, and the material blocks map gradient variations to symmetric
stress variations. It retains the elastic TPSA traction derivative, interior
pressure feedback, one-sided Dirichlet reaction corrections, and every cell
unknown. Normals already include face area; divergence adds only incidence signs.
Fixed boundary displacement has zero variation while its reaction traction varies.
History, prescribed load data, transfer weights, geometry, and elastic moduli
are held fixed during differentiation.

The document presents a condensed displacement solve. This implementation keeps
`u`, `r`, and `p` independent and solves them together. Its displacement Schur
complement is exactly `J_uu + J_ur @ R + J_up @ P`, where
`A_rr @ R = -A_ru` and `A_pp @ P = -A_pu`. Multiplying `J_face` by the stacked
matrix `[I; R; P]` gives the document's condensed face derivative. Production
assembly does not form these dense sensitivity matrices or solve auxiliary
systems inside Newton evaluation. Tests check this equivalence by recomputing
both auxiliary fields for every perturbed displacement.

`vonMisesModel.consistent_tangent()` in `material/J2.py` supplies the full 3D tensor
`C_alg = d sigma_return / d epsilon` for the existing radial return map. Define
`rho = ||dev(sigma_trial) - dev(backstress_n)||`, `n = shifted_trial / rho`, and
`D = 2*mu + (2/3)*(K_prime(alpha) + H_prime(alpha))`. On the plastic branch,

```text
C_alg = C_e - 4*mu^2 * [(gamma/rho)*P_dev + (1/D - gamma/rho)*(n outer n)]
```

`P_dev` is the symmetric deviatoric projector. The code's multiplier convention
is `delta_epsilon_p = gamma*n` and `delta_alpha = sqrt(2/3)*gamma`, so the
multiplier differs from the document's convention using the von Mises yield
normal. The derivatives of both hardening laws are evaluated at the updated
alpha, supporting linear mixed hardening and differentiable nonlinear laws with
a positive consistency denominator. The elastic branch returns `C_e` exactly,
without normalizing a zero trial deviator. At a yield switch the branch is the
one selected by the return map; a unique classical derivative is not claimed.

The plane-strain adapter in `material/tangent.py`, `analytical_material_tangent(point, committed_history,
strain_increment, returned=...)` returns shape `(3,3,2,2)`, mapping the in-plane
gradient to all 3D stress components. Shear columns include the symmetric-gradient
factor of one half. `returned` must match that same increment and history. The
coupled callback reuses the residual's returned histories, performing no extra
return maps or finite differences; omitting `returned` in the standalone helper
performs one isolated update. The existing `MaterialPoint.update()` return
contract `(state, None)` is retained: tangents are evaluated on demand instead
of during every residual evaluation.

Tests in `test/alternatives/original/test_analytical_jacobian.py` compare the full 3D tangent with
independent centered return-map differences, including nonlinear hardening,
perfect plasticity, changed loading direction, zero deviatoric stress, and elastic
unloading. They also check condensed residual/face derivatives, boundary reactions,
interior-face cancellation, reuse of returned histories, and matching load histories
for both Jacobian options. The shared coupled derivative and Newton tests in
`test/alternatives/original/test_jacobian.py` run against both implementations. These checks verify the
derivative of the proposed residual, not spatial convergence of the plastic scheme.

An optional numerical material tangent remains available through
`FiniteDifferenceJacobian` in `alternatives/original/jacobian.py`:

```python
from alternatives.original.jacobian import FiniteDifferenceJacobian

jacobian = FiniteDifferenceJacobian(
    operators, transfer, case.material_points, state_n, step=1e-10,
)
result = solve_newton(
    operators, transfer, case.material_points, state_n, case.bc_values,
    jacobian=jacobian,
)
```

This follows the local forward-difference approach of Mazzanti and Cardiff,
*Performance of a vertex-centred block-coupled finite volume methodology for
small-strain static elastoplasticity*, Section 3.2.2, Algorithm 2 and Eq. (20)
([supplied paper](1-s2.0-S0898122125003098-main.pdf), PDF page 6). Its assembly is
adapted to this project's cell-based TPSA residual and Green–Gauss reconstruction.
For each cell, it evaluates baseline stress and four perturbations of the
in-plane displacement gradient `G`. Perturbing `G_xy` changes both symmetric
strain entries by `step/2`; the return map still uses full 3D tensors and history.
Every evaluation restarts from the supplied committed history. It does not use
or modify the existing committed/trial slots of `case.material_points`.

The standalone helper in `material/tangent.py`, `finite_difference_material_tangent(point, committed_history,
strain_increment, step=...)` returns `d sigma[i,j] / d G[k,l]` with shape
`(3, 3, 2, 2)`. The configurable step is an absolute, dimensionless gradient
perturbation; `1e-10` matches the paper's choice. Smaller steps can increase
roundoff error, while larger steps can cross a yield switch. At such switches,
this one-sided approximation need not yield quadratic Newton convergence.

The callback caches the exact sparse derivative `B = d G / d x`, the configured
cell-to-face weights `W`, normal contraction `N`, and linear TPSA blocks. Its
momentum rows are
`J_u = D_u @ T - D_u @ N @ (W kron I_4) @ blockdiag(C_e - C_FD) @ B`.
Tensor entries use `[xx, xy, yx, yy]` per cell/face. Boundary face displacement
has zero derivative at fixed prescribed values; pressure-dependent interior
reconstruction remains included. Face normals already carry face area, and
orientation signs come from divergence. Rotation and pressure residual rows
retain their original linear derivatives. The full matrix acts on `[u, r, p]`;
finite differences are applied only to local material updates.

Construct a new callback after accepting a load step or changing geometry,
reference elastic coefficients, models, or transfer weights. Use the same
committed state and fixed data for both the callback and the residual evaluator.
Both Jacobians use the same callback API and sparse assembly. Tests in
`test/alternatives/original/test_jacobian.py` check elastic
recovery, tensor shear, plastic loading, mixed elastic/plastic cells, unloading,
custom weights, history isolation, and agreement with independent finite
differences of the full residual. They also exercise elastic and plastic Newton
solves; the plastic test starts near the homogeneous solution, without a line
search or load continuation.

`LoadStepController` in `alternatives/loading.py` advances prescribed load factors
and accepts each complete trial only after the coupled Newton solve converges.
For example, this sequence loads beyond yield and then unloads slightly:

```python
import numpy as np
from alternatives.loading import LoadStepController
from alternatives.plane_strain import PlaneStrainTpsa
from alternatives.original.residual import TpsaOperators
from utils.state import TpsaState
from utils.transfer import CellToFaceTransfer

case = PlaneStrainTpsa(
    cells_per_axis=3, displacement_gradient=np.diag([0.004, 0.0]),
)
case.prepare_simulation()
operators = TpsaOperators(case)
transfer = CellToFaceTransfer(case.grid)
controller = LoadStepController(
    operators, transfer, case.material_points,
    TpsaState.zeros(case.grid), case.bc_values,
)
records = controller.run(np.r_[np.linspace(0.05, 1.0, 20), 0.99, 0.98])
state_n = controller.state
last_factor = controller.load_factor
last_iterations = records[-1].newton.iterations
last_residual_norms = records[-1].newton.residual_norms
```

Each factor is an absolute target: boundary displacement is
`factor * reference_bc_values`, and the optional integrated source vector is
`factor * reference_sources`. Both reference arrays are copied. Interior boundary
entries are ignored. Sources default to zero and follow `[body force, rotation,
pressure]` ordering. Factors are processed in the supplied order, including
unloading, repeated values, and negative factors. An empty schedule is a no-op;
invalid schedules are rejected before any step is taken. `advance(factor)` solves
one target, while `run(factors)` returns records for that invocation and stops
at the first failure.

Every step starts from the previous accepted `[u, r, p]` solution and builds a
fresh `FiniteDifferenceJacobian` by default, using the committed history for
that step. Set `finite_difference_step=` to change the local perturbation, or provide
`jacobian_factory(old_state)` returning your own Newton callback. The factory
receives a copy of the previous state and can capture fixed operators/models.
For the analytical J2 Jacobian, use
`jacobian_factory=lambda old: AnalyticalJacobian(operators, transfer, case.material_points, old)`.
For an elastic-only sequence, use
`jacobian_factory=lambda old: (lambda x, evaluation: operators.A)`.
`max_iterations`, `atol`, and `rtol` are passed to each Newton solve. Relative
residual thresholds restart at each step, so a repeated load can require further
corrections if its previous solve stopped above the absolute tolerance.

Each `LoadStepResult` contains `load_factor`, the target `sources`, and `newton`
with all convergence diagnostics. Its `state` contains accepted strains,
material histories, and numerical tractions as well as `[u, r, p]`. Acceptance
advances this entire state together. It does not call `MaterialPoint.commit()`
on the case's points; the controller's `TpsaState` is the authoritative history.
The original state, reference loads, case solution, and material-point histories
remain unchanged. Returned records, `controller.state`, and `controller.steps`
are independent snapshots, so editing them cannot change the controller.

Failures raise `LoadStepError` with a one-based `step_index` and attempted
`load_factor`; `__cause__` retains the original error. The last accepted state,
load factor, and records remain accessible. Fixed-step calls never accept a failed candidate or retry automatically; the
caller may explicitly submit a new target after handling the error. To restart in a new controller, supply
`committed=previous.state` and `initial_load_factor=previous.load_factor` with the
same reference loads. The initial boundary data must match that factor, and the
caller must supply compatible initial histories/tractions and source loading.
The initial state is not included in the accepted-step records.

For adaptive increments, use `run_adaptive()` with absolute waypoints:

```python
from utils.load_schedule import AdaptiveLoadOptions

options = AdaptiveLoadOptions(
    initial_increment=0.05, min_increment=1e-4, max_increment=0.2,
    cutback_factor=0.5, growth_factor=1.5, target_iterations=4,
    max_retries=8, max_steps=1000,
)
# On a fresh controller, load to 1, then unload to 0.98.
records = controller.run_adaptive([1.0, 0.98], options=options)
attempts = controller.attempts  # Immutable diagnostics for accepted and rejected attempts.
```

Increments are positive magnitudes; the next waypoint determines the direction.
Every waypoint is reached exactly without overshoot. Equal consecutive targets
are skipped. The proposed increment carries across waypoints in the same
direction and resets to `initial_increment` on reversal. A final remainder below
`min_increment` is allowed to land exactly on a waypoint.

Only `NewtonConvergenceError` causes a cutback. The failed trial is discarded,
the increment is multiplied by `cutback_factor` (clamped to `min_increment`),
and a fresh solve/Jacobian starts from the same accepted history. Invalid input,
malformed Jacobians, linear-solve failures and material errors stop immediately.
After an acceptance using at most `target_iterations` corrections, the proposed
increment grows by `growth_factor`, up to `max_increment`; growth is suppressed
immediately after a cutback. Slower converged steps retain their increment.
`growth_factor=1` disables growth. Newton tolerances and full coupled corrections
are unchanged. This controls convergence effort, not a load-discretization error
estimate.

`max_retries` limits retries at one accepted state and resets after a successful
substep. `max_steps` bounds the total accepted substeps in one adaptive invocation.
Failure at the minimum increment, exhaustion of these bounds, or inability to
make floating-point progress raises `AdaptiveLoadStepError` (a `LoadStepError`)
with the waypoint in `target`. Earlier accepted states remain accessible.
Each `LoadAttempt` stores `start_factor`, attempted `load_factor`, accepted
Newton `iterations` (or `None` on failure), and an optional error message;
`accepted` reports success. Attempts are also recorded by fixed-step calls.

Use `iter_adaptive()` with the same arguments to check or report each accepted
substep before the next is attempted. Closing the iterator preserves the latest
accepted state. An entire path is validated before its first solve; empty paths
are no-ops. Do not advance the controller separately while its iterator is active.

Tests in `test/alternatives/test_loading.py` cover absolute load/source scaling, plastic
loading and elastic unloading, fresh Jacobians, restart, snapshot ownership,
and preservation of accepted history after Newton, linear-solve, factory, or
material failures. Adaptive tests cover retry recovery, growth, exact endpoints,
reversals, minimum/retry/step bounds, and floating-point stagnation.

A one-page description of the optional `original` formulation’s boundary
conditions, material, load path and solver settings is available as [PDF](docs/plastic_plane_strain_case.pdf)
and [LaTeX source](docs/plastic_plane_strain_case.tex). Regenerate the supplied
PDF with `python docs/generate_plastic_plane_strain_case.py` (Matplotlib).

The runnable plastic benchmark is `examples/plastic_plane_strain.py`:

```bash
python -m examples.plastic_plane_strain
```

It prepares a 3 by 3 Cartesian grid on the unit square and prescribes
`u_x = factor * 0.004 * x`, `u_y = 0` on every boundary face, with zero body
force. Twenty equal load increments reach factor 1, followed by unloading to
0.99 and 0.98. The material constants and linear mixed J2 hardening are those
of `PlaneStrainProblem`. The runner selects the analytical J2 Jacobian by default,
with a fresh callback at each load step. Use `--jacobian finite-difference` for
the numerical alternative (and `--fd-step` to adjust its perturbation). Both
methods solve all retained increments `[du,dr,dp,d_free]` together: 60 unknowns
on this mesh (36 cell unknowns and 24 interior-face displacement components).
The retained setup does not assemble the alternative elastic TPSA matrices.

Adaptive stepping is optional:

```bash
python -m examples.plastic_plane_strain --adaptive --no-export
```

The runner then uses the turning points `1.0` and `1-unload_fraction`; fixed-mode
`--loading-steps` and `--unloading-steps` counts are unused. Configure
`--initial-increment`, `--min-increment`, `--max-increment`, `--cutback-factor`,
`--growth-factor`, `--target-iterations`, `--max-retries`, and `--max-steps` using
the defaults shown in `AdaptiveLoadOptions` above. Python callers use
`run_example(adaptive=AdaptiveLoadOptions(...))`; `adaptive=None` retains the
fixed schedule. Both Jacobian methods support adaptation.

Every accepted adaptive substep receives the same numerical checks as a fixed
step. The peak state is identified by its factor, so elastic-unloading checks
and the peak Jacobian heatmap use the actual accepted peak and its preceding
history. All accepted substeps are exported in order; rejected attempts appear
in console cutback diagnostics and `result.controller.attempts`, not in VTK/PNG
histories. The console also reports accepted/rejected counts. With the default
adaptive controls, the current benchmark accepts eight steps; this count can
change with material, mesh, tolerances or controls. Fixed mode still uses 22.

Console rows report step number, loading/unloading phase, absolute load factor,
Newton correction count, the three raw residual block norms, and maximum
accumulated equivalent plastic strain. Each step is followed by its complete
Newton history, including iteration 0, all four block norms, `ratio_f` and
`order_f` for retained runs. With `r[k] = ||R_f(z[k])||`, the reduction ratio is `r[k]/r[k-1]`
and the observed residual convergence order is
`log(r[k]/r[k-1]) / log(r[k-1]/r[k-2])`. Orders near 1 indicate linear decay
and near 2 quadratic decay when the iteration is in its asymptotic regime.
The retained report uses the nonlinear face-balance block. The alternatives
report momentum rates (`ratio_u`, `order_u`). In the original formulation,
rotation and pressure equations are linear and usually reach numerical noise
after one correction. In the updated coupled formulation they include plastic
feedback, and the iteration table also reports the strain constraint `||R_eps||`.

`--` marks unavailable rates. Order requires three positive, strictly decreasing
residuals above a heuristic floor of `100*machine_epsilon*max(history)` for that
block, with a resolvable logarithmic denominator. Rate histories restart at every
load step, and neither solver tolerances nor convergence decisions are changed.
These are empirical residual rates; a short history, a yield switch, or numerical
noise can prevent a reliable asymptotic-order estimate. The Python runner remains
quiet unless `verbose=True` or `show_l2_errors=True`.

The L2 error summary is hidden by default, including when `verbose=True`.
Use `python -m examples.plastic_plane_strain --show-l2-errors` (optionally with
`--no-export`) or `run_example(show_l2_errors=True)` to print the largest
**absolute discrete L2 field errors** over all accepted steps.
This option controls printing only: numerical checks always run, and errors
remain available in `result.checks`. These field errors
use cell-volume or PorePy face-dual-volume weights, separately from Newton's
unweighted residual norms. Tensor errors include all nine 3D components, retaining
out-of-plane stress and plastic strain under the plane-strain constraint.

The reference is computed independently of `MaterialPoint.update()` and the
implemented radial return map. For proportional loading from zero history it
uses the closed-form plastic multiplier
`gamma = max(0, ||dev(sigma_elastic)|| - sqrt(2/3)*sigma_y) / (2*mu + 2*H_bar/3)`.
It gives the analytical stress, plastic strain, backstress, and accumulated
plastic strain at every loading factor. During unloading, the reference freezes
peak plastic history and applies the elastic stress increment. The runner
rejects an elastic-only peak or unloading beyond reverse yield before solving;
this reference assumes homogeneous affine deformation and linear mixed hardening.
It does not validate spatial accuracy for nonuniform plastic problems.

Every accepted step must pass freshly assembled equilibrium checks against its
Newton thresholds and analytical comparisons for displacement, rotation stress,
TPSA pressure, strain, material stress, plastic strain, backstress, accumulated
plastic strain, and face force. Plane strain, nondecreasing accumulated plastic
strain, plastic yielding at the peak, and unchanged plastic history during
elastic unloading are checked explicitly. Each field's L2 error must be below
its absolute floor plus `1e-7` times the analytical field's L2 norm. The SI floors
are `1e-12` for displacement, `1e-11` for strain/plastic strain/alpha, `2` for
stress/rotation/pressure, `0.01` for backstress, and `1` for face force, in the
norm units printed by the runner. Every actual error and limit is retained in
the returned step checks.

Successful completion prints `Numerical checks: PASS`. A Newton failure or failed
check stops execution, prints an error, and exits with status 1. The default case
yields near factor 0.4 and reaches peak alpha approximately `1.628e-3`; the two
unloading steps retain that plastic history. After all numerical checks pass,
the command exports accepted states with `pp.Exporter` to `results/`:

- `pvd/plastic_plane_strain_retained_analytical.pvd`: the complete loading/unloading collection. Open this
  file in ParaView to browse all accepted steps.
- `vtu/plastic_plane_strain_retained_analytical_2_000001.vtu`, etc.: one file per accepted step, on the
  undeformed mesh.
- `plastic_plane_strain_retained_analytical_stress_strain.png`: volume-mean Cauchy stress
  `sigma_xx` (MPa) against volume-mean total strain `epsilon_xx`. The curve follows
  accepted-step order, with decreasing load-factor segments marked in red.
- `plastic_plane_strain_retained_analytical_alpha.png`: volume-mean accumulated equivalent plastic
  strain against accepted step number.
- `plastic_plane_strain_retained_analytical_jacobian.png`: the full coupled Jacobian evaluated at the
  converged peak loading state (step 20, factor 1 by default), using the selected
  analytical or finite-difference Jacobian and the preceding committed history.
  Rows are residual blocks `[R_u, R_r, R_p, R_f]`; columns are increments
  `[du, dr, dp, d_free]`.
  Dashed lines separate blocks. Blue/red show negative/positive entries; zero is
  white. A symmetric logarithmic color scale shows the raw entries with their
  different block units, with a linear interval within `+/-1e-12`.

All plastic-run output filenames include the selected formulation and Jacobian type:
`plastic_plane_strain_retained_analytical_*` or `plastic_plane_strain_retained_finite-difference_*`.
This applies to PNGs, VTUs, and both the main and individual-step PVD collections,
so distinct formulation/Jacobian combinations can share an output directory.
Previously generated outputs are left in place.

All VTU files are grouped under `results/vtu/`. PVD collections (including
individual-step PVDs) are grouped under `results/pvd/` and reference VTUs through
`../vtu/`. PNG plots stay directly under `results/`. Custom output directories
use the same layout. Open `results/pvd/plastic_plane_strain_retained_analytical.pvd` in ParaView for
the complete plastic history.

PVD time values are **sequence indices 1, 2, ...**, not physical times or load
factors. Each VTU stores `load_step` and `load_factor` separately, so unloading
retains its chronological order. Only accepted states are exported; no initial
zero state is added. The two history PNGs use cell-volume averages, which equal the local
values for this homogeneous example. A gold diamond labeled **Yield detected**
marks the first recorded state with positive alpha in any cell, showing its step
number and load factor (step 8, factor 0.4 for the default run). This marks the
first sampled plastic state; the exact yield onset can lie between increments.
Entirely elastic histories have no marker. If a supplied history already starts
with accumulated plastic strain, its first recorded state is marked.

VTK cell fields include `displacement` (m, zero z component), its magnitude,
`rotation_stress`, `total_pressure`, full tensors `strain`, `stress`,
`plastic_strain`, `backstress`, and scalar `alpha`. Tensors use nine row-major
components `(xx, xy, xz, yx, yy, yz, zx, zy, zz)` and also expose separate `_xx`,
`_yy`, `_xy`, and `_zz` scalar fields. Stress and backstress are in Pa; strains
and alpha are dimensionless, with tensor shear rather than engineering shear.
In particular, plane strain retains nonzero `stress_zz` and `plastic_strain_zz`.
These are the accepted constitutive fields, distinct from numerical face forces.

Use `--output-dir PATH` to change the destination or `--no-export` to run only
numerical checks. Re-running the same formulation and Jacobian type replaces its matching outputs
and PVD collection;
unrelated files in `results/` remain intact. If a shorter history is exported,
older unreferenced VTUs may remain but are not included in the new collection.

To inspect the accepted fields programmatically:

```python
from examples.plastic_plane_strain import run_example

result = run_example()  # quiet; verbose=True shows Newton, show_l2_errors=True shows L2
state = result.controller.state.tpsa  # Total fields in the default retained state.
records = result.controller.steps
checks = result.checks
stress_cell_0 = state.material_states[0].stress.to_numpy()
peak_stress_error = checks[19].errors["stress"]
```

The Python API exports only when requested: `run_example(output_dir="results")`
returns the generated paths in `result.outputs` (keys `pvd`, `stress_strain`,
`alpha`, `jacobian`). To export the field history of an existing result without
solving again:

```python
from utils.visualization import export_plastic_history

paths = export_plastic_history(
    result.case.grid, result.controller.steps, folder_name="results",
    file_name="plastic_plane_strain_retained_analytical",  # Match the method used for this result.
)
```

A separate heatmap can be exported for any assembled coupled matrix:

```python
from utils.visualization import export_jacobian_png

path = export_jacobian_png(matrix, folder_name="results", file_name="jacobian")
```

The helper accepts dense or sparse matrices in `[u,r,p]` order, or
`[u,r,p,strain]` with `formulation="coupled"`, preserves their values, and
densifies sparse input for plotting. It does not assemble a Jacobian.
Use `linthresh` to change the linear interval of the color scale. The runner's
snapshot is evaluated at the converged peak; it is not a saved matrix from an
earlier Newton correction. `--no-export` also suppresses this PNG.

The result also exposes `case`, `operators`, and `transfer`. Configure
`--cells-per-axis`, `--loading-steps`, `--unloading-steps`, `--peak-strain`,
`--unload-fraction`, `--jacobian`, `--fd-step`, and `--max-iterations` on the command line;
the corresponding keyword arguments are available on `run_example()`
(`finite_difference_step` is the Python name for `--fd-step`). Coarser load
increments may require more Newton iterations; use `--adaptive` to enable cutbacks.
Tests in `test/examples/test_retained_runner.py`, `test/examples/test_formulation_selection.py`, and
`test/examples/test_solver_setup.py` cover the retained default, formulation dispatch,
exports, and optional-import isolation. `test/examples/test_plastic_plane_strain.py`
preserves coverage of the original formulation, an alternate
grid/schedule, CLI failures, analytical limits, and deliberate corruption of
accepted fields and unloading histories. `test/utils/test_plastic_visualization.py`
reads back the VTK series, checks full tensor/component ordering and unloading
chronology, verifies PNG volume averages on unequal cells, and checks input and
figure preservation. Invalid histories are rejected before files are written.

Tests mirror the source packages:

```text
test/
├── core/                 # Retained solver operators, residuals, Jacobians, Newton, loading
├── material/             # J2 return map, material history, tensors
├── utils/                # Shared utilities, transfers, VTK/PNG output
├── alternatives/         # Shared TPSA setup, reconstruction, Newton, loading
│   ├── original/         # Original residual and Jacobian formulation
│   └── coupled/          # Coupled strain formulation
└── examples/             # End-to-end runners, formulation selection, exports
```

Integration tests stay with the main package they exercise. The preserved
`manual_tensor.py` and `manual_tpsa.py` scripts are outside pytest discovery.
Package markers distinguish repeated filenames such as `test_jacobian.py`.

Run the complete suite or a package subset from the repository root:

```bash
python -m pytest -q
python -m pytest test/core -q
python -m pytest test/material test/utils -q
python -m pytest test/alternatives/original -q
```

Type-check the configured solver, example, utility, and test modules with:

```bash
python -m mypy
```

`mypy.ini` enables strict checks for the configured files and preserves PorePy's
implicit exports so its annotations remain visible. Numerical arrays use
`NDArray[np.float64]`. This environment lacks SciPy and meshio type stubs; their imports have scoped
`import-untyped` exceptions (meshio is used to read back VTK files in tests).

The setup and solve helpers follow the names and argument order in PorePy's
`tests/numerics/fv/test_tpsa.py`:

- `_set_uniform_parameters(g, val=1, ...)` creates the parameter dictionary.
  Optional `mu` and `lmbda` supply this case's distinct elastic moduli.
- `_set_uniform_bc(grid, d, bc_type)` assigns the boundary-condition type.
- `_discretize_get_matrices(grid, d)` calls the original `Tpsa.discretize`.
- `_assemble_matrices(matrices, g, d)` returns `face_discretization`,
  `rhs_matrix`, `div`, and `accum`.
- `_solve(face_discretization, rhs_matrix, div, accum, bound_vec)` returns the
  solution vector `x`, ordered as displacement `u`, rotation `r`, and pressure `p`.

The case now uses these explicit sparse-matrix stages directly. Run
`case = PlaneStrainTpsa(); x = case.solve()` in place of the earlier
`ModelRunner(model).run()` call. Every solve prepares a fresh case; it does not
advance or commit material histories. The analytical `reference_*` helpers are
specific to this example; they have no counterparts in the original TPSA files.
Matrix keys, including `stress`, `stress_rotation`, and `stress_total_pressure`,
retain the names from `src/porepy/numerics/fv/tpsa.py`.

A concise description of the current plane-strain problem, its equations,
boundary conditions, TPSA system, and exact elastic solution is available as
[PDF](docs/plane_strain_problem.pdf) and
[LaTeX source](docs/plane_strain_problem.tex). The PDF is compiled directly from
that source. Rebuild it with Tectonic:

```bash
tectonic docs/plane_strain_problem.tex
```

The LaTeX source is self-contained and also works with a standard LaTeX setup:

```bash
pdflatex -halt-on-error -output-directory=docs docs/plane_strain_problem.tex
```
