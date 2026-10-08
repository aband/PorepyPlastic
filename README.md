# PorepyPlastic

An elastoplasticity extension for PorePy.

The first global example is a small-strain, static, **elastic TPSA plane-strain
problem** on an unfractured 1 m by 1 m Cartesian square. Run it from this directory
in an environment with PorePy installed:

```bash
python -m coupling.plane_strain
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
python -m coupling.plane_strain --output-dir /tmp/plane_strain_vtk
```

The reusable visualization module is `coupling/visualization.py`:

```python
from coupling.plane_strain import PlaneStrainTpsa
from coupling.postprocessing import TpsaPostprocessing
from coupling.visualization import export_png, export_strain_png, export_vtk

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

```python
from coupling.plane_strain import PlaneStrainTpsa
from coupling.postprocessing import TpsaPostprocessing

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
use `python -m coupling.plane_strain --show-l2-errors` to print it.
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

`PlaneStrainTpsa` in `coupling/plane_strain.py` allocates one independent
`MaterialPoint` per cell during case preparation. These histories remain at
their initial values during the elastic global solve. Green–Gauss strain
reconstruction is implemented. Work on the global residual starts with the
`TpsaState` container in `coupling/residual.py`. It stores `x`, `bc_values`,
`epsilon`, `material_states`, and integrated numerical face `traction`. The
`reference_*` methods supply analytical elastic values for verification only.

```python
from coupling.residual import TpsaState

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
from coupling.residual import TpsaOperators

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
from coupling.residual import evaluate_material_trial

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
in `coupling/postprocessing.py` also serve the existing solved-case interface.
The numerical and history-isolation tests are in `test/test_material_trial.py`.

`CellToFaceTransfer` in `coupling/transfer.py` supplies a separate, configurable
transfer of cell stress corrections to faces. It stores a sparse matrix `W` of
shape `(num_faces, num_cells)` and applies it to every tensor component:

```python
from coupling.transfer import CellToFaceTransfer

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

`evaluate_face_traction` in `coupling/residual.py` updates the numerical face
traction for the same candidate and material-trial result:

```python
from coupling.residual import evaluate_face_traction

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
loading/unloading in `test/test_face_traction.py`.

The fully coupled global residual is now available through
`evaluate_global_residual` in `coupling/residual.py`. It evaluates the material
response and numerical face tractions for the supplied candidate, then assembles
all three equation blocks:

```python
from coupling.residual import evaluate_global_residual

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
physical units. The coupled residual tests are in `test/test_global_residual.py`.

`solve_newton` in `coupling/newton.py` now solves one target load with a required
Jacobian callback. For an elastic load, the existing TPSA matrix is the reference
Jacobian:

```python
from coupling.newton import solve_newton

result = solve_newton(
    operators, transfer, case.material_points, state_n, case.bc_values,
    jacobian=lambda x, evaluation: operators.A,
)
candidate = result.trial             # independent converged TpsaState
iterations = result.iterations       # number of Newton corrections
norms = result.residual_norms         # rows: evaluations; columns: momentum, rotation, pressure
from coupling.newton import newton_convergence_rates, print_newton_convergence
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

Tests in `test/test_newton.py` cover elastic recovery with dense and sparse
Jacobians, individual block convergence, integrated sources, repeated evaluation
from fixed plastic history during elastic unloading, and failure isolation.
This driver takes full Newton steps. Residual scaling, line search, and adaptive
step reduction remain future work. Least-squares reconstruction also remains deferred.
`plane_strain.py` continues to run the elastic baseline.

The analytical Jacobian implements the specified residual in
[Plasticity_Dirichlet_marked.pdf](Plasticity_Dirichlet_marked.pdf), especially
Eqs. (12)–(13), (24)–(26), and the full-Dirichlet completion (D1)–(D13).
Use `AnalyticalJacobian` in `coupling/jacobian.py`:

```python
from coupling.jacobian import AnalyticalJacobian

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

`vonMisesModel.consistent_tangent()` in `J2.py` supplies the full 3D tensor
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

The plane-strain adapter `analytical_material_tangent(point, committed_history,
strain_increment, returned=...)` returns shape `(3,3,2,2)`, mapping the in-plane
gradient to all 3D stress components. Shear columns include the symmetric-gradient
factor of one half. `returned` must match that same increment and history. The
coupled callback reuses the residual's returned histories, performing no extra
return maps or finite differences; omitting `returned` in the standalone helper
performs one isolated update. The existing `MaterialPoint.update()` return
contract `(state, None)` is retained: tangents are evaluated on demand instead
of during every residual evaluation.

Tests in `test/test_analytical_jacobian.py` compare the full 3D tangent with
independent centered return-map differences, including nonlinear hardening,
perfect plasticity, changed loading direction, zero deviatoric stress, and elastic
unloading. They also check condensed residual/face derivatives, boundary reactions,
interior-face cancellation, reuse of returned histories, and matching load histories
for both Jacobian options. The shared coupled derivative and Newton tests in
`test/test_jacobian.py` run against both implementations. These checks verify the
derivative of the proposed residual, not spatial convergence of the plastic scheme.

An optional numerical material tangent remains available through
`FiniteDifferenceJacobian` in `coupling/jacobian.py`:

```python
from coupling.jacobian import FiniteDifferenceJacobian

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

The standalone `finite_difference_material_tangent(point, committed_history,
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
`test/test_jacobian.py` check elastic
recovery, tensor shear, plastic loading, mixed elastic/plastic cells, unloading,
custom weights, history isolation, and agreement with independent finite
differences of the full residual. They also exercise elastic and plastic Newton
solves; the plastic test starts near the homogeneous solution, without a line
search or load continuation.

`LoadStepController` in `coupling/loading.py` advances prescribed load factors
and accepts each complete trial only after the coupled Newton solve converges.
For example, this sequence loads beyond yield and then unloads slightly:

```python
import numpy as np
from coupling.loading import LoadStepController
from coupling.plane_strain import PlaneStrainTpsa
from coupling.residual import TpsaOperators, TpsaState
from coupling.transfer import CellToFaceTransfer

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
load factor, and records remain accessible. The controller neither accepts a
failed candidate nor retries automatically; the caller may explicitly submit a
new target after handling the error. To restart in a new controller, supply
`committed=previous.state` and `initial_load_factor=previous.load_factor` with the
same reference loads. The initial boundary data must match that factor, and the
caller must supply compatible initial histories/tractions and source loading.
The initial state is not included in the accepted-step records.

Tests in `test/test_loading.py` cover absolute load/source scaling, plastic
loading and elastic unloading, fresh Jacobians, restart, snapshot ownership,
and preservation of accepted history after Newton, linear-solve, factory, or
material failures.

A one-page description of its default boundary conditions, material, load path
and solver settings is available as [PDF](docs/plastic_plane_strain_case.pdf)
and [LaTeX source](docs/plastic_plane_strain_case.tex). Regenerate the supplied
PDF with `python docs/generate_plastic_plane_strain_case.py` (Matplotlib).

The runnable plastic benchmark is `coupling/plastic_plane_strain.py`:

```bash
python -m coupling.plastic_plane_strain
```

It prepares a 3 by 3 Cartesian grid on the unit square and prescribes
`u_x = factor * 0.004 * x`, `u_y = 0` on every boundary face, with zero body
force. Twenty equal load increments reach factor 1, followed by unloading to
0.99 and 0.98. The material constants and linear mixed J2 hardening are those
of `PlaneStrainTpsa`. The runner selects the analytical J2 Jacobian by default,
with a fresh callback at each load step. Use `--jacobian finite-difference` for
the numerical alternative (and `--fd-step` to adjust its perturbation). Both
methods solve `u,r,p` together. The elastic `case.solve()` is never called.

Console rows report step number, loading/unloading phase, absolute load factor,
Newton correction count, the three raw residual block norms, and maximum
accumulated equivalent plastic strain. Each step is followed by its complete
Newton history, including iteration 0, the three block norms, `ratio_u` and
`order_u`. With `r[k] = ||R_u(x[k])||`, the reduction ratio is `r[k]/r[k-1]`
and the observed residual convergence order is
`log(r[k]/r[k-1]) / log(r[k-1]/r[k-2])`. Orders near 1 indicate linear decay
and near 2 quadratic decay when the iteration is in its asymptotic regime.
The report uses the nonlinear momentum block; rotation and pressure equations
are linear and usually reach numerical noise after one correction.

`--` marks unavailable rates. Order requires three positive, strictly decreasing
residuals above a heuristic floor of `100*machine_epsilon*max(history)` for that
block, with a resolvable logarithmic denominator. Rate histories restart at every
load step, and neither solver tolerances nor convergence decisions are changed.
These are empirical residual rates; a short history, a yield switch, or numerical
noise can prevent a reliable asymptotic-order estimate. The Python runner remains
quiet unless `verbose=True` or `show_l2_errors=True`.

The L2 error summary is hidden by default, including when `verbose=True`.
Use `python -m coupling.plastic_plane_strain --show-l2-errors` (optionally with
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

- `pvd/plastic_plane_strain_analytical.pvd`: the complete loading/unloading collection. Open this
  file in ParaView to browse all accepted steps.
- `vtu/plastic_plane_strain_analytical_2_000001.vtu`, etc.: one file per accepted step, on the
  undeformed mesh.
- `plastic_plane_strain_analytical_stress_strain.png`: volume-mean Cauchy stress
  `sigma_xx` (MPa) against volume-mean total strain `epsilon_xx`. The curve follows
  accepted-step order, with decreasing load-factor segments marked in red.
- `plastic_plane_strain_analytical_alpha.png`: volume-mean accumulated equivalent plastic
  strain against accepted step number.
- `plastic_plane_strain_analytical_jacobian.png`: the full coupled Jacobian evaluated at the
  converged peak loading state (step 20, factor 1 by default), using the selected
  analytical or finite-difference Jacobian and the preceding committed history.
  Rows are residual blocks `[R_u, R_r, R_p]`; columns are unknowns `[u, r, p]`.
  Dashed lines separate blocks. Blue/red show negative/positive entries; zero is
  white. A symmetric logarithmic color scale shows the raw entries with their
  different block units, with a linear interval within `+/-1e-12`.

All plastic-run output filenames include the selected Jacobian type:
`plastic_plane_strain_analytical_*` or `plastic_plane_strain_finite-difference_*`.
This applies to PNGs, VTUs, and both the main and individual-step PVD collections,
so runs using different Jacobians can share an output directory.

All VTU files are grouped under `results/vtu/`. PVD collections (including
individual-step PVDs) are grouped under `results/pvd/` and reference VTUs through
`../vtu/`. PNG plots stay directly under `results/`. Custom output directories
use the same layout. Open `results/pvd/plastic_plane_strain_analytical.pvd` in ParaView for
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
numerical checks. Re-running the same Jacobian type replaces its matching outputs
and PVD collection;
unrelated files in `results/` remain intact. If a shorter history is exported,
older unreferenced VTUs may remain but are not included in the new collection.

To inspect the accepted fields programmatically:

```python
from coupling.plastic_plane_strain import run_example

result = run_example()  # quiet; verbose=True shows Newton, show_l2_errors=True shows L2
state = result.controller.state
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
from coupling.visualization import export_plastic_history

paths = export_plastic_history(
    result.case.grid, result.controller.steps, folder_name="results",
    file_name="plastic_plane_strain_analytical",  # Match the method used for this result.
)
```

A separate heatmap can be exported for any assembled coupled matrix:

```python
from coupling.visualization import export_jacobian_png

path = export_jacobian_png(matrix, folder_name="results", file_name="jacobian")
```

The helper accepts dense or sparse matrices in `[u,r,p]` order, preserves their
values, and densifies sparse input for plotting. It does not assemble a Jacobian.
Use `linthresh` to change the linear interval of the color scale. The runner's
snapshot is evaluated at the converged peak; it is not a saved matrix from an
earlier Newton correction. `--no-export` also suppresses this PNG.

The result also exposes `case`, `operators`, and `transfer`. Configure
`--cells-per-axis`, `--loading-steps`, `--unloading-steps`, `--peak-strain`,
`--unload-fraction`, `--jacobian`, `--fd-step`, and `--max-iterations` on the command line;
the corresponding keyword arguments are available on `run_example()`
(`finite_difference_step` is the Python name for `--fd-step`). Coarser load
increments may require more Newton iterations; automatic cutbacks remain deferred.
Tests in `test/test_plastic_plane_strain.py` cover the default path, an alternate
grid/schedule, CLI failures, analytical limits, and deliberate corruption of
accepted fields and unloading histories. `test/test_plastic_visualization.py`
reads back the VTK series, checks full tensor/component ordering and unloading
chronology, verifies PNG volume averages on unequal cells, and checks input and
figure preservation. Invalid histories are rejected before files are written.

Run the baseline checks with:

```bash
python -m pytest -q
```

Type-check the coupling modules and their tests with:

```bash
python -m mypy
```

`mypy.ini` enables strict checks for these twenty-three files and preserves PorePy's
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
