# Examples and numerical checks

[Back to README](../README.md) · [Retained solver](retained_solver.md) ·
[Output guide](visualization.md)

Run the commands below from the repository root. Both plastic runners default
to `formulation="retained"`, `jacobian="analytical"`; formulation and Jacobian
selection are independent. The Python `run_example()` functions write no files
unless `output_dir` is supplied. Their CLIs export to `results/` unless passed
`--no-export`.

## Common material and boundary setup

The examples use a unit square with both displacement components prescribed on
all boundary faces, zero body force and initially zero material history.
This is **plane strain**: total `epsilon_zz`, `epsilon_xz` and `epsilon_yz` are
zero, while stress and plastic histories retain all 3D components.

| Material parameter | Default |
| --- | --- |
| Young's modulus `E` | `210 GPa` |
| Poisson ratio `nu` | `0.3` |
| Initial yield stress | `250 MPa` |
| Linear mixed hardening modulus `H_bar` | `1 GPa` |
| Isotropic fraction `theta` | `0.4` |

The scalar laws are `K(alpha) = 250 MPa + 0.4 GPa*alpha` and
`H(alpha) = 0.6 GPa*alpha`. A load factor scales absolute boundary displacement;
it is an amplitude parameter, not physical time. Decreasing factors in the
default histories mean prescribed unloading.

## Homogeneous plastic loading and unloading

[`examples/plastic_plane_strain.py`](../examples/plastic_plane_strain.py)
provides the checked homogeneous case:

```bash
python -m examples.plastic_plane_strain --no-export
python -m examples.plastic_plane_strain --show-l2-errors --no-export
python -m examples.plastic_plane_strain --adaptive --no-export
```

| Setting | Default |
| --- | --- |
| Grid | `3 x 3` cells |
| Boundary displacement | `u_x = factor*0.004*x`, `u_y = 0`, on every side |
| Fixed loading targets | `0.05, 0.10, ..., 1.0` (20 steps) |
| Fixed unloading targets | `0.99, 0.98` (2 steps) |
| Adaptive waypoints | `1.0, 0.98` |
| Newton correction limit | `20` per step |

At factor one the right edge moves 4 mm. Top and bottom are also displaced in
`x` and carry reactions; they are not traction-free surfaces. This constrained
plane-strain case is not a uniaxial-stress or plane-stress test. The retained
system has 60 unknowns (36 cell and 24 free-face components); original has 36,
and coupled has 63.

Use `--cells-per-axis`, `--peak-strain`, `--loading-steps`, `--unloading-steps`
and `--unload-fraction` to adjust the fixed case. Adaptive mode uses peak/final
waypoints instead of fixed step counts. Its increment, growth, cutback and
iteration-target options are listed by `--help` and documented in the
[load-control guide](retained_solver.md#adaptive-load-control).
`--fd-step` changes the numerical material-tangent perturbation only.

### Reference and acceptance checks

The reference is independent of `MaterialPoint.update()`. For proportional
homogeneous loading from zero history with linear mixed hardening, it uses

```text
gamma = max(0, ||dev(sigma_elastic)|| - sqrt(2/3)*sigma_y)
        / (2*mu + 2*H_bar/3)
```

It computes stress and plastic history at each loading factor. During elastic
unloading it freezes peak plastic history and applies the elastic stress
increment. The runner rejects an elastic-only peak or unloading beyond reverse
yield. With default fixed targets, yield is first detected at factor `0.4`,
and peak accumulated equivalent plastic strain is approximately `1.628e-3`.

Every accepted step is checked for freshly assembled equilibrium, plane strain,
nondecreasing `alpha`, plastic yield at peak and unchanged plastic history on
elastic unloading. It also compares displacement, rotation stress, TPSA pressure,
strain, stress, plastic strain, backstress, `alpha` and numerical face force
against the reference. Retained equilibrium uses all four residual blocks and
both side-force histories.

Field errors are **absolute discrete L2 errors**, using cell-volume or PorePy
face-dual-volume weights. Tensor errors include all nine 3D components. Each
limit is an absolute floor plus `1e-7` times the reference field's L2 norm:

| Field | Absolute floor in the corresponding SI norm units |
| --- | --- |
| Displacement | `1e-12` |
| Strain, plastic strain, `alpha` | `1e-11` |
| Stress, rotation stress, pressure | `2` |
| Backstress | `0.01` |
| Face force | `1` |

The checks always run; `--show-l2-errors` controls printing only. Errors and
limits remain available in `result.checks`. Successful CLI completion prints
`Numerical checks: PASS`; solver or check failure exits with status 1. These
checks validate this homogeneous solution, not nonuniform spatial accuracy.

### Newton diagnostics

The CLI reports per-step Newton iterations and block residual histories.
Ratios compare successive residual norms; observed orders use three successive
norms. Retained runs report rates for interior-face balance `R_f`; alternatives
use momentum `R_u`, and coupled runs also display the strain residual `R_eps`.
Short histories, yield switches or roundoff can leave the observed order
undefined. These empirical rates are distinct from the field L2 errors.
The Python runner is quiet unless `verbose=True` or `show_l2_errors=True`.

## Nonuniform Dirichlet example

[`examples/dirichlet_example.py`](../examples/dirichlet_example.py) prescribes

```text
u_x = factor * 0.004 m * (x/L) * [1 + 0.2*sin(pi*y/L)]
u_y = 0,                       L = 1 m
```

Both components are constrained on every boundary. Fixed stepping uses 20
loading targets through factor one and one unload to `0.98`. Adaptive stepping
uses waypoints `[1.0, 0.98]`. The CLI defaults to `4 x 4` cells; the Python
`run_example()` default is `20 x 20`. Set `cells_per_axis` explicitly for a
comparison.

```bash
python -m examples.dirichlet_example --adaptive --no-export
python -m examples.dirichlet_example --cells-per-axis 4 --jacobian finite-difference --adaptive --no-export
```

The runner reports accepted/rejected steps, yielding cells and Newton
convergence. It assumes no exact interior solution and does not report reference
L2 field errors. Its CLI supports mesh size, formulation, Jacobian, adaptive
mode and output location; it does not expose all the homogeneous runner's load
controls. Exported field plots show the final unloaded state, while the Jacobian
heatmap uses the converged peak increment and preceding committed history.

## Supply your own boundary displacement

The shared `PlaneStrainProblem` accepts a callback sampled once during
`prepare_simulation()`. It receives a copy of boundary-face centers of shape
`(3, nb)` in `grid.get_all_boundary_faces()` order. Return finite real values
of shape `(2, nb)` in metres. For example:

```python
import numpy as np
from utils.problem import PlaneStrainProblem
from prototype.runner import prepare_retained_solver


def displacement(centers):
    x, y = centers[:2]  # Unit-square coordinates in metres.
    return 0.004 * np.vstack((x * (1 + 0.2*np.sin(np.pi*y)), np.zeros_like(x)))

case = PlaneStrainProblem(cells_per_axis=4, boundary_displacement=displacement)
case.prepare_simulation()
setup = prepare_retained_solver(case, "analytical")
```

Advance `setup.controller` with a fixed or adaptive schedule as in the
[retained guide](retained_solver.md). The controller scales the entire supplied
pattern by each absolute factor. Boundary types remain full Dirichlet; this
callback changes values only.

`case.bc_values` has shape `(2, nf)` and shares storage with flat `bound_vec`.
`bc_values_displacement(grid)` returns a fresh interleaved vector with zero
interior entries and no face-area scaling. Omitting the callback uses
`displacement_gradient @ coordinates + translation`. Explicit gradient or
translation cannot be combined with a callback. Custom callbacks disable the
affine `reference_*` and homogeneous-reference helpers, even if the callback
happens to describe an affine field.

Changing captured callback parameters requires re-preparing the case and
creating a new controller; existing controllers retain copied reference loads.
Re-preparation also creates fresh geometry and material points.

## Elastic baseline and local return-map animation

```bash
python -m examples.elastic_plane_strain
python -m examples.elastic_plane_strain --show-l2-errors --output-dir results/elastic
python -m examples.visualize_newton_stress --help
```

The elastic baseline uses an `8 x 8` grid (64 cells, 256 unknowns) with
`u_x = 1e-4*x`, `u_y = 0` on every boundary. It solves the elastic TPSA system,
recovers Green–Gauss strain and exports VTK and strain PNGs. Its CLI supports
`--output-dir` and `--show-l2-errors`, not `--no-export` or mesh-size selection.
The displacement PNG helper is available separately in the [output guide](visualization.md).

The animation visualizes the **local** J2 return-map iteration, not global
Newton convergence. Consult `--help` for preview and output options; saving
video requires FFmpeg.
