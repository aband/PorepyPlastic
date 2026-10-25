# Original and coupled formulations

[Back to README](../README.md) · [Data conventions](architecture.md) ·
[Retained solver](retained_solver.md)

The optional solvers in `alternatives/` preserve two formulations for comparison
with the default retained solver. Select them with `--formulation original` or
`--formulation coupled` in either plastic example. Both support analytical and
local finite-difference Jacobians, fixed/adaptive loading and common exports.

## Elastic TPSA setup

[`PlaneStrainTpsa`](../alternatives/plane_strain.py) extends the shared
`PlaneStrainProblem`. After `prepare_simulation()`, the case exposes its grid,
material points, boundary data, PorePy parameter dictionary `d`, discretization
`matrices`, `face_discretization`, `rhs_matrix`, `div` and `accum`.
`case.solve()` prepares a fresh elastic case, solves all `u, r, p` together,
returns the vector and stores it as `case.x`. It does not advance plastic history.

The helpers `_set_uniform_parameters`, `_set_uniform_bc`,
`_discretize_get_matrices`, `_assemble_matrices` and `_solve` follow the names
and argument order in PorePy's `tests/numerics/fv/test_tpsa.py`. Matrix keys such
as `stress`, `stress_rotation` and `stress_total_pressure` follow `Tpsa`.
Run the elastic case with `python -m examples.elastic_plane_strain`.

[`TpsaPostprocessing`](../alternatives/postprocessing.py) provides the solved-case
interface. It retains the case data and offers `strain_green_gauss()` to recover
plane-strain tensors from TPSA face displacements. Reconstruction does not
update material history. See the [export example](visualization.md#export-an-elastic-solution).

## Original formulation

[`TpsaOperators`](../alternatives/original/residual.py) stores the fixed TPSA
matrix blocks, traction maps `T`, `T_g` and force divergence `D_u`.
The Newton candidate `x = [u, r, p]` has `4*nc` entries. Reconstruction uses the
elastic TPSA face-displacement operators to obtain total Green–Gauss strain.
Material evaluation then forms
`Q = sigma_n + C_e:(epsilon - epsilon_n) - sigma_returned` from committed history.

The numerical force is updated as

```text
t = t_n + T @ (x - x_n) + T_g @ (g - g_n) - Q_face @ N_face
Q_face = W Q
```

Here `g` is absolute prescribed displacement, `W` transfers each tensor component
from cells to faces, and `N_face` includes face area. Force divergence supplies
the cell incidence signs. The old numerical traction carries plastic history;
it is not reconstructed from averaged cell stress.

For `f = sources - div @ rhs_matrix @ g`, the residual is

```text
R_u = D_u @ t - sources_u
R_r = A_ru @ u + A_rr @ r - f_r
R_p = A_pu @ u + A_pp @ p - f_p
```

Optional `sources` is the absolute target integrated vector of length `4*nc`,
ordered as body force, rotation source and pressure source. It is not a density,
a load increment or the boundary-adjusted `f`. Momentum subtracts the original
body force because traction already contains the boundary contribution.

`evaluate_material_trial`, `evaluate_face_traction`, `evaluate_global_residual`
and `assemble_global_residual` expose successive stages without committing
history. The high-level evaluation retains an independent candidate `TpsaState`
and stress correction. All three fields are independent during evaluation.
`solve_auxiliary(u, rhs)` is available for reduced-formulation experiments;
production Newton solves `u, r, p` together.

### Transfer weights

[`CellToFaceTransfer`](../utils/transfer.py) defaults to equal interior weights
and weight one on boundary faces. Its sparse `W` has shape `(nf, nc)`, carries
no areas or signs, and stays fixed during Newton. Rules can be supplied without
changing material, residual or Jacobian code:

```python
import numpy as np
from utils.transfer import CellToFaceTransfer


def distance_weights(grid, face, cells):
    distances = np.linalg.norm(
        grid.cell_centers[:, cells] - grid.face_centers[:, face, None], axis=0,
    )
    weights = 1.0 / distances
    return weights / weights.sum()

# With a prepared case: CellToFaceTransfer.from_rule(case.grid, distance_weights)
```

Rules run on interior faces and receive adjacent cell IDs in ascending order.
They must return finite nonnegative weights summing to one. An explicit sparse
matrix can instead be passed as `weights=W`. The `.weights` property returns a
copy; construct a new transfer object to change the rule. Geometry and the
chosen transfer still require spatial verification for the intended problem.

### Original Jacobian

[`AnalyticalJacobian`](../alternatives/original/jacobian.py) uses the consistent
3D return-map tangent. `FiniteDifferenceJacobian` substitutes the local numerical
tangent, with a default absolute gradient perturbation of `1e-10`.
Their sparse assembly is

```text
J_face = T - N @ (W kron I_4) @ blockdiag(C_e - C_alg) @ B
J      = [D_u @ J_face; A_r; A_p]
```

`B` differentiates the reconstructed in-plane gradient. Material blocks map its
four components `[xx, xy, yx, yy]` to symmetric stress. Interior pressure
feedback and boundary reaction derivatives remain included; boundary
displacement has zero variation. Histories, loads, geometry, elastic moduli
and transfer weights are fixed during differentiation.

All cell unknowns remain in the assembled system. For comparison, its
condensed displacement derivative is `J_uu + J_ur @ R + J_up @ P`, where
`A_rr @ R = -A_ru` and `A_pp @ P = -A_pu`. Production assembly does not form
these dense sensitivities or eliminate the auxiliary fields.

### Run with an explicit analytical factory

```python
import numpy as np
from alternatives.plane_strain import PlaneStrainTpsa
from alternatives.original.residual import TpsaOperators
from alternatives.original.jacobian import AnalyticalJacobian
from alternatives.loading import LoadStepController
from utils.state import TpsaState
from utils.transfer import CellToFaceTransfer

case = PlaneStrainTpsa(
    cells_per_axis=3, displacement_gradient=np.diag([0.004, 0.0]),
)
case.prepare_simulation()
ops = TpsaOperators(case)
transfer = CellToFaceTransfer(case.grid)
controller = LoadStepController(
    ops, transfer, case.material_points, TpsaState.zeros(case.grid), case.bc_values,
    jacobian_factory=lambda old: AnalyticalJacobian(ops, transfer, case.material_points, old),
)
records = controller.run(np.r_[np.linspace(0.05, 1.0, 20), 0.99, 0.98])
```

The example runners default to analytical tangents. The lower-level alternative
`LoadStepController` defaults to finite differences when no factory is supplied.

## Coupled strain formulation

[`CoupledOperators`](../alternatives/coupled/residual.py) retains independent
cell strain alongside `u, r, p`: the Newton vector has `7*nc` entries, with
strain packed cell by cell as `[epsilon_xx, epsilon_yy, sqrt(2)*epsilon_xy]`.

For an interior face with `a_i = 2*mu_i/d_i`, displacement weights are
`a_i/(a_i+a_j)` and complementary traction weights are `a_j/(a_i+a_j)`.
Boundary traction weights are one. These use fixed elastic coefficients.
The correction `Q` contributes both to traction and to shared face displacement,
whose plastic feedback includes `(Q_i-Q_j) n/(a_i+a_j)`. Rotation and pressure
balances use that same displacement. The fourth residual enforces symmetric
Green–Gauss reconstruction; strain is solved simultaneously with the cell fields.

With reference reconstruction `B`, strain feedback `H_epsilon`, and mechanical
plastic-source maps `H`, the monolithic Jacobian is

```text
J = [ A          H D_Q             ]
    [ -B         I - H_epsilon D_Q ]
D_Q = blockdiag(C_e - C_alg)          # matching Mandel coordinates
```

[`CoupledAnalyticalJacobian`](../alternatives/coupled/jacobian.py) uses the
consistent tangent, while `CoupledFiniteDifferenceJacobian` uses the same global
chain rule with a numerical material tangent.

For lower-level setup, use `ops = CoupledOperators(case)`,
`transfer = ops.transfer`, and `TpsaState.zeros(case.grid, store_faces=True)`
with the alternative controller. For analytical tangents, provide
`jacobian_factory=lambda old: CoupledAnalyticalJacobian(ops, transfer, case.material_points, old)`.
`state.x` remains length `4*nc`; `ops.unknowns(state)` packs the full vector.
Restart requires stored total `face_displacement`, which elastic interpolation
alone cannot recover after plastic loading.

This formulation supports full displacement Dirichlet conditions on Cartesian
tensor grids with cell/face centroids and positive finite elastic coefficients.
It validates its complementary weights and material/operator consistency.
The geometry restrictions differ from the rotated orthogonal grids supported
by the retained solver.

## Newton, load control and restart

The alternatives share [`newton.py`](../alternatives/newton.py) and
[`loading.py`](../alternatives/loading.py). Direct `solve_newton` requires a
Jacobian callback. Full corrections start from the old total fields by default;
there is no Armijo option in this driver. It checks each raw residual block
against `atol + rtol*initial_norm`, with 20 corrections by default.
Original defaults are `atol=(1e-5, 1e-12, 1e-12)` and relative tolerances `1e-8`.
Coupled solves append strain defaults `atol=1e-12`, `rtol=1e-8`, or accept
explicit four-entry tuples.

The controller scales absolute reference boundary values and optional
`reference_sources` by each target factor. Acceptance advances cell fields,
strain, material histories and numerical tractions together, including coupled
face displacement. Material-point slots and case solution remain unchanged.
Returned records and state properties are independent snapshots.

Fixed/adaptive scheduling, failure handling and attempt diagnostics follow the
[shared load-control rules](retained_solver.md#adaptive-load-control). To restart,
supply the complete accepted state, `initial_load_factor`, and matching reference
loads; retain compatible source and traction histories. Initial state is not
included among accepted-step records.

## One-page case note

The tracked [LaTeX source](plastic_plane_strain_case.tex) describes the optional
**original formulation with the fixed homogeneous schedule**, not the default
retained solver. Generate its ignored PDF using

```bash
python docs/generate_plastic_plane_strain_case.py
```

The [Matplotlib generator](generate_plastic_plane_strain_case.py) writes
`docs/plastic_plane_strain_case.pdf`. Alternatively, compile the LaTeX source
with your LaTeX installation. The historical design PDFs are not required to
run the code; current behavior is described by these guides and source modules.
