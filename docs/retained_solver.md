# Retained-displacement solver

[Back to README](../README.md) · [Data conventions](architecture.md) ·
[Example settings](examples.md)

The default formulation lives in `prototype/`, while its selection flag remains
`--formulation retained`. It solves cell and free-face increments together,
with a constitutive return map inside every residual evaluation. This is the
nested solver; the Newton–Schur method remains deferred.

## Prepare and advance a problem

This complete example prepares a homogeneous plane-strain case, loads beyond
yield, then unloads slightly:

```python
import numpy as np
from utils.problem import PlaneStrainProblem
from prototype.runner import prepare_retained_solver

case = PlaneStrainProblem(
    cells_per_axis=3, displacement_gradient=np.diag([0.004, 0.0]),
)
case.prepare_simulation()
setup = prepare_retained_solver(case, "analytical")
ops, controller = setup.operators, setup.controller
records = controller.run(np.r_[np.linspace(0.05, 1.0, 20), 0.99, 0.98])

state = controller.state
x = state.tpsa.x                         # total [u, r, p]
epsilon = state.tpsa.epsilon             # total strain, (3, 3, nc)
face_u = state.tpsa.face_displacement     # total displacement, (2, nf)
side_force = state.side_traction         # total outward force, (2, nsides)
z_last = records[-1].newton.z            # increments for the last accepted step
```

Use `"finite-difference"` in `prepare_retained_solver` to select the numerical
local tangent. The setup returns operators, a virgin state, a Jacobian factory
and a controller. For custom controls, construct
[`LoadStepController`](../prototype/loading.py) directly with the same operators,
material points, initial `RetainedState` and reference boundary values.

## Unknowns and reconstruction

For `nc` cells and `nfree` interior faces, the Newton vector is

```text
z = [delta_u, delta_r, delta_p, d_free]     length = 4*nc + 2*nfree
R = [R_u,     R_r,     R_p,     R_f]        same block sizes
```

The first three blocks are increments of the cell fields from the committed
load step. `d_free` contains displacement increments on interior faces.
Boundary data are supplied as **absolute** displacements; the evaluator inserts
target minus committed boundary values. Vector components are interleaved.
Cell `u, r, p` remain independent unknowns. Only the face increments determine
the strain increment through symmetric Green–Gauss reconstruction.

[`RetainedFaceOperators`](../prototype/operators.py) exposes packing, reconstruction,
face/cell-side maps and accumulation coefficients. Prescribed faces have
`face_dofs == -1`; they have no face-balance equations. Global traction/residual
assembly supports orthogonal 2D tensor grids with full component Dirichlet
boundaries, including rotated rectangles. The example preparation uses a
Cartesian unit square.

Elastic coefficients `mu` and `lam` may be positive scalars or cell arrays.
Both are required for the global solve and are checked against the materials.
Only `mu` is needed for traction evaluation; kinematic operators can be built
without either. Rebuild operators after geometry or elastic-coefficient changes.

## Residual and one-sided traction

For a supplied candidate, [`evaluate_global_residual`](../prototype/residual.py):

1. Reconstructs total strain from committed strain and face increments.
2. Calls the shared material evaluator from the same committed history in each cell.
3. Forms `Q = sigma_n + C_e : delta_epsilon - sigma_returned`.
4. Evaluates one-sided integrated traction increments and all four residual blocks.

`evaluation.material_trial` contains total strain, returned material histories
and the full 3D correction `Q`. `evaluation.traction_increment` has shape
`(2, ops.num_sides)`: side `s` belongs to `side_faces[s]` and `side_cells[s]`,
uses that cell's outward normal and already includes face area. Interior faces
have two independent sides; boundary sides provide reaction increments.

Cell momentum sums outward force increments. `R_f` sums both sides of each
interior face. Rotation and pressure use the same face displacement increments
as strain reconstruction, minus `(V/mu)*delta_r` and `(V/lam)*delta_p`.
Corrections `Q` act separately on each side; only the rotation transfer uses
complementary cell-to-face weights. Residuals are integrated and unscaled.

Optional `body_force_increment` contains integrated cell force increments,
shape `(2, nc)`, subtracted once using the convention `div(sigma) = body_force`.
It is not a force density or boundary-adjusted right-hand side. Auxiliary
sources are zero. Supply `committed_residual` to include the old state's
imbalance; omitting it assumes an equilibrated committed state.

Lower-level `evaluate_one_sided_traction` and `assemble_global_residual` support
assembly with supplied corrections or tractions. None of these functions
accepts a load step or changes material-point histories. The controller later
accumulates the incremental traction into the total history.

## Analytical and numerical Jacobians

Both [`RetainedAnalyticalJacobian`](../prototype/jacobian.py) and
`RetainedFiniteDifferenceJacobian` implement the callback `(z, evaluation)`.
Construct one per load step from its committed `TpsaState`, using the same fixed
operators and materials as the residual evaluator.

```text
K = L + P @ D_Q @ B
D_Q = blockdiag(C_e - C_alg)
```

`B` maps increments to cell-major Mandel strain `[xx, yy, sqrt(2)*xy]`;
`L` is the elastic residual derivative, and `P` inserts stress corrections.
These sparse maps are exposed on the callback. Cellwise constant corrections
cancel from cell momentum, so the nonlinear correction occupies only
interior-face balance rows and free-face displacement columns.

`plastic_tangent(z, evaluation)` exposes `D_Q`.
`face_jacobian(z, evaluation)` returns the one-sided force derivative with shape
`(2*nsides, num_dofs)`, including boundary reactions. Its elastic and correction
maps are `T` and `P_side`. Boundary targets, body-force increments and committed
residual are held fixed during differentiation.

The analytical callback reuses the residual's returned histories without extra
return maps. The numerical callback substitutes a forward-difference **local
material tangent**, with `step=1e-10` by default; global assembly stays analytic.
See the [constitutive conventions](architecture.md#shared-constitutive-update-and-tangents).
Both callbacks copy committed history and reject mismatched candidates or strains.

## Newton convergence

[`solve_newton`](../prototype/newton.py) defaults to the analytical Jacobian and zero
load-step increments, including after a nonzero committed state. A sparse linear
solve computes corrections to all four blocks. Each residual evaluation starts
its local updates from a snapshot of the same committed history.

Every block must satisfy `norm <= atol + rtol*initial_norm`. Defaults are:

| Block | Absolute tolerance | Relative tolerance |
| --- | --- | --- |
| Cell momentum `u` | `1e-5` | `1e-8` |
| Rotation `r` | `1e-12` | `1e-8` |
| Pressure `p` | `1e-12` | `1e-8` |
| Interior-face balance `f` | `1e-5` | `1e-8` |

These are Euclidean norms of raw integrated residuals, not discrete field L2
errors. `max_iterations=20` counts corrections; zero checks only the initial
candidate. Supply four-entry tolerances when customizing retained convergence.

Full steps are the default. The Python API supports
`line_search=prototype.newton.BacktrackingOptions()` for Armijo backtracking.
Its merit scales start at the larger of each initial norm and absolute tolerance;
cell momentum and face balance share the larger force scale. Convergence still
uses each block's own tolerance. `step_lengths` records accepted damping,
`residual_evaluations` includes rejected trials, and `residual_norms` records
only the initial and accepted iterates.

Iteration or line-search exhaustion raises `NewtonConvergenceError`. Invalid
inputs, material errors and linear-solve failures propagate immediately. A
successful `RetainedNewtonResult` holds total strain/material histories but
incremental unknowns and traction; Newton itself does not commit the step.

## Acceptance and restart

The controller scales reference loads by an absolute load factor. It passes
Newton absolute boundary targets, the body-force difference from the accepted
state, and the residual assembled from the old total history. After convergence
it accumulates increments exactly once, adopts returned strain/material states,
and checks total equilibrium before accepting the complete state.

`RetainedState` validates stored face displacement, boundary data and total
Green–Gauss strain. Its `tpsa.traction` is a derived signed average for common
reporting; equilibrium uses both `side_traction` histories.
`prototype.state.assemble_state_residual(ops, state)` reassembles total equilibrium.

`advance(factor)` solves one target and `run(factors)` solves a fixed sequence.
Increasing, decreasing, repeated and negative factors are supported. A failed
step leaves accepted state unchanged and raises `LoadStepError` with the original
exception chained. Fixed stepping does not retry automatically. `state`,
`steps` and returned records are independent snapshots; `attempts` records
accepted and failed attempts.

For restart, provide the complete `controller.state` and
`initial_load_factor=controller.load_factor` to a new controller with the same
operators, material parameters and reference loads. Stored boundary values and
body force must match that factor. To select another Jacobian, provide a
`jacobian_factory(old)` receiving the committed `TpsaState`.

## Adaptive load control

Both solver families use [`utils.load_schedule`](../utils/load_schedule.py).
On a fresh controller, the following chooses substeps toward peak load and
then unloading:

```python
from utils.load_schedule import AdaptiveLoadOptions

# Fresh controller: this is a separate run from the fixed schedule above.
adaptive_setup = prepare_retained_solver(case, "analytical")
options = AdaptiveLoadOptions()
adaptive_records = adaptive_setup.controller.run_adaptive([1.0, 0.98], options=options)
```

| Option | Default |
| --- | --- |
| `initial_increment` | `0.05` |
| `min_increment`, `max_increment` | `1e-4`, `0.2` |
| `cutback_factor`, `growth_factor` | `0.5`, `1.5` |
| `target_iterations` | `4` |
| `max_retries`, `max_steps` | `8`, `1000` |

Increments are positive magnitudes; waypoints determine direction. Each
waypoint is reached exactly. Same-direction segments carry the proposed
increment; reversal resets it to `initial_increment`. Repeated adaptive targets
are skipped. A final remainder smaller than `min_increment` is allowed.

Only `NewtonConvergenceError` triggers cutback and a fresh solve/Jacobian from
the last accepted state. Other errors stop immediately. A converged step taking
at most `target_iterations` corrections permits growth up to `max_increment`,
except immediately after a cutback. `growth_factor=1` disables growth. This
adapts to convergence effort, not a load-discretization error estimate.

Failure at minimum increment, retry/step limits or floating-point stagnation
raises `AdaptiveLoadStepError`, preserving earlier accepted states. Each attempt
records start/target factors, success, iterations or an error.
`iter_adaptive(...)` yields each accepted state before attempting the next;
do not advance the controller separately while that iterator is active.
