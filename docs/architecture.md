# Architecture and data conventions

[Back to README](../README.md)

## Package dependencies

`examples/` selects a formulation and prepares its problem, controller and
exports. The default path uses `prototype/`; selecting `original` or `coupled` loads
`alternatives/` lazily. Both solver families reuse `utils/` and `material/`.
Shared utility modules do not import either solver package, and `prototype/` does
not import `alternatives/`.

| Package | Main modules |
| --- | --- |
| `prototype` | `operators`, `residual`, `jacobian`, `newton`, `state`, `loading`, `runner` |
| `material` | `J2`, `material_state`, `scalar_hardening`, `tensor`, `tangent` |
| `utils` | `problem`, `state`, `material_evaluation`, `reconstruction`, `transfer`, `newton_utils`, `load_schedule`, `visualization`, `formulation` |
| `alternatives` | `original`, `coupled`, shared `newton`/`loading`, elastic `plane_strain`/`postprocessing` |

[`PlaneStrainProblem`](../utils/problem.py) prepares the grid, material points
and reference Dirichlet values. Preparation creates a fresh case; it does not
solve or advance material history.
[`PlaneStrainTpsa`](../alternatives/plane_strain.py) extends this setup with the
elastic TPSA matrices and a linear solve for the optional formulations.

## Fields, units and storage

The examples use SI units: metres, pascals and dimensionless strain. Integrated
2D forces are per unit out-of-plane thickness. Face normals supplied by PorePy
already contain face area; traction assembly includes that area once.

For `nc` cells, `nf` faces and `ns` cell-face sides:

| Data | Shape or ordering | Meaning |
| --- | --- | --- |
| Cell vector `x` | `[u, r, p]`, length `4*nc` | Total cell displacement, rotation stress and TPSA pressure |
| Displacement `u` | `(2, nc)` or interleaved flat vector | `[ux_0, uy_0, ux_1, uy_1, ...]`; reshape with `order="F"` |
| Boundary displacement | `(2, nf)` or interleaved flat vector | Absolute prescribed values; interior entries are unused/zeroed |
| Total strain `epsilon` | `(3, 3, nc)` | Symmetric 3D tensor with `zz`, `xz`, `yz` equal to zero |
| Material history | One `MaterialState` per cell | Full 3D stress, plastic strain, backstress and accumulated equivalent plastic strain `alpha` |
| `TpsaState.traction` | `(2, nf)` | Integrated numerical traction in the fixed grid-normal orientation |
| `RetainedState.side_traction` | `(2, ns)` | Total outward force on each cell-face side |
| `RetainedState.body_force` | `(2, nc)` | Total integrated cell body force |

`r` is the TPSA rotation-stress variable and `p` the TPSA pressure variable.
For the elastic affine reference, `r = mu*(du_x/dy - du_y/dx)` and
`p = lambda*div(u)`; `p` is not the mean Cauchy stress. Out-of-plane **stress and
plastic strain** can be nonzero even though total out-of-plane strain is zero.
Tensor shear is `epsilon_xy`; engineering shear is `2*epsilon_xy`.

## State ownership and acceptance

[`TpsaState`](../utils/state.py) stores total cell fields, boundary values,
strain, material histories, numerical face traction and optional total face
displacement. Construction and `.copy()` give independent arrays and histories.
[`RetainedState`](../prototype/state.py) adds both one-sided traction histories and
body force; these are required to check retained equilibrium and restart it.

Every Newton trial uses the same explicitly supplied committed history. Material
evaluation reuses `MaterialPoint.update()` without changing the material points'
own committed/trial slots. The load controller owns the accepted history and
accepts the complete state only after convergence and its acceptance checks.
Changing a returned snapshot does not modify the controller.

`utils.state.as_tpsa_state(state)` exposes total fields for common reporting.
Use the complete formulation-specific state for equilibrium, acceptance and
restart. In particular, the retained state's averaged face traction cannot
replace its two independent interior-side histories.

## Shared constitutive update and tangents

[`material/J2.py`](../material/J2.py) implements the 3D radial return map and
consistent tangent. The multiplier convention is
`delta_epsilon_p = gamma*n`, `delta_alpha = sqrt(2/3)*gamma`. With
`rho = ||dev(sigma_trial) - dev(backstress_n)||`, unit direction `n`, and
`D = 2*mu + (2/3)*(K_prime(alpha) + H_prime(alpha))`, the plastic tangent is

```text
C_alg = C_e - 4*mu^2 * [(gamma/rho)*P_dev + (1/D - gamma/rho)*(n outer n)]
```

`P_dev` is the symmetric deviatoric projector. Hardening derivatives are
evaluated at the updated `alpha`; the elastic branch returns `C_e` directly.
At yield switches, the selected return-map branch determines the tangent.

[`material/tangent.py`](../material/tangent.py) adapts this response to in-plane
gradient variations. Both `analytical_material_tangent(...)` and
`finite_difference_material_tangent(...)` return `(3, 3, 2, 2)` arrays mapping
the in-plane gradient to all 3D stress components. Shear columns include the
one-half symmetric-gradient factor.

The analytical helper can reuse the residual's matching returned history via
`returned=...`, avoiding extra return maps. `MaterialPoint.update()` retains
its `(state, None)` return contract; tangents are computed on demand.
The finite-difference helper uses a baseline and four forward gradient
perturbations, each starting from the same committed history. Its default
`step=1e-10` is an absolute dimensionless gradient perturbation. Roundoff or
crossing a yield switch can affect the numerical tangent.

## Tests and type checking

`test/prototype`, `test/material`, `test/utils`, `test/alternatives` and
`test/examples` mirror the package structure. Integration tests live with the
main package they exercise. Package markers distinguish repeated test filenames;
`manual_tensor.py` and `manual_tpsa.py` are manual scripts outside discovery.

Run `python -m pytest -q` from the repository root, or pass a package directory.
`python -m mypy` checks the files listed in [`mypy.ini`](../mypy.ini) in strict
mode. Numerical arrays use `NDArray[np.float64]`; PorePy's implicit exports are
enabled so its annotations remain visible.
