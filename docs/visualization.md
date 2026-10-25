# Visualization and generated output

[Back to README](../README.md) · [Example settings](examples.md) ·
[State conventions](architecture.md#fields-units-and-storage)

[`utils/visualization.py`](../utils/visualization.py) contains solver-independent
VTK and PNG exporters. Example runners supply a prefix containing both the
formulation and Jacobian type, such as `plastic_plane_strain_retained_analytical`.
Standalone helper calls use the caller's `file_name`; they do not infer a solver.

## File layout and history

With the default output directory, a plastic run produces:

```text
results/
├── pvd/                    # Main history and individual-step collections
├── vtu/                    # Mesh and cell fields for accepted steps
├── <prefix>_stress_strain.png
├── <prefix>_alpha.png
└── <prefix>_jacobian.png
```

PVD references point to `../vtu/`. Open the main collection, for example
`results/pvd/plastic_plane_strain_retained_analytical.pvd`, in ParaView.
It preserves accepted-step order during both loading and unloading. PVD time
values are sequential accepted-step indices, not load factors or physical time;
`load_step` and `load_factor` are stored as cell fields. The initial zero state
is not included in the accepted-step series.

The nonuniform Dirichlet runner also writes final displacement and strain PNGs.
The homogeneous runner writes history curves and a peak Jacobian heatmap;
it does not automatically write separate final field PNGs.

Repeated runs with the same prefix replace the main collection but can leave
unreferenced old VTUs. Formulation/Jacobian suffixes distinguish those choices,
not mesh size, load schedule or every parameter. Use separate `--output-dir`
locations for comparisons. `./clean_results.sh` removes the repository's entire
`results/` directory; custom output directories are unaffected.

## Displacement and strain plots

`export_png(grid, x, ...)` shows the vector-valued cell displacement using
PorePy's `pp.save_img`. Arrow directions show direction and lengths show
magnitude, with the displacement scale indicated. It does not deform the mesh.
The default title and filename refer to displacement.

`export_strain_png(grid, epsilon, ...)` returns paths keyed by `xx`, `yy`, `xy`
and `principal`. With default `file_name="plane_strain"`, the files are
`plane_strain_epsilon_xx.png`, `plane_strain_epsilon_yy.png`,
`plane_strain_epsilon_xy.png` and `plane_strain_principal_strain.png`.

Component maps share a symmetric color range centered at zero. The principal
plot uses centered segments along the in-plane eigenvectors, with lengths
proportional to absolute eigenvalues on one common scale. Red indicates
extension and blue contraction. Equal principal strains appear as circles
because no direction is preferred; zero strain produces no glyph. Plot-level
roundoff suppression does not alter VTK tensor values.

For plastic runs, use stored total strain from the accepted state. Reconstructing
it with the optional elastic TPSA postprocessor would lose the formulation's
plastic face-displacement history.

## Export an elastic solution

This complete example writes displacement, strain components and principal
strain in addition to VTK:

```python
from alternatives.plane_strain import PlaneStrainTpsa
from alternatives.postprocessing import TpsaPostprocessing
from utils.visualization import export_png, export_strain_png, export_vtk

case = PlaneStrainTpsa()
x = case.solve()
epsilon = TpsaPostprocessing(case).strain_green_gauss()
pvd = export_vtk(case.grid, x, epsilon=epsilon, folder_name="results/elastic")
displacement_png = export_png(case.grid, x, folder_name="results/elastic")
strain_pngs = export_strain_png(case.grid, epsilon, folder_name="results/elastic")
```

`export_vtk` exports displacement, rotation and pressure; optional `epsilon`
adds the full `strain` tensor and `strain_xx`, `strain_yy`, `strain_xy`,
`strain_zz`. Tensor data use nine components in row-major order:
`xx, xy, xz, yx, yy, yz, zx, zy, zz`. Shear is tensor strain, so engineering
shear is twice `strain_xy`.

## Plastic history and yield markers

`export_plastic_history(grid, records, ...)` accepts any records exposing
`load_factor` and `state`. The common adapter `as_tpsa_state` exposes total
fields from either solver family. Each VTK snapshot includes displacement,
rotation, pressure, strain, material stress, plastic strain, backstress,
accumulated equivalent plastic strain `alpha` and load metadata. Tensor outputs
retain full 3D components, including out-of-plane stress and plastic strain.
Stored constitutive stress differs from the numerical face force used for
equilibrium.

The stress–strain PNG plots volume-mean `sigma_xx` in MPa against volume-mean
`epsilon_xx`. The alpha PNG plots volume-mean accumulated equivalent plastic
strain against accepted step number. Means use cell volumes:

```text
mean(alpha) = sum(V_cell * alpha_cell) / sum(V_cell)
```

They are not simple cell-count averages on unequal cells. Decreasing-factor
segments are marked as unloading. The yield marker identifies the first
accepted state with positive `alpha` in any cell. It is a sampled onset, not an
interpolated exact yield load; adaptive steps can locate a different first
positive sample.

## Jacobian heatmaps

`export_jacobian_png(matrix, ...)` writes a signed heatmap with a symmetric
logarithmic scale, white at zero and blue/red for negative/positive entries.
The default linear interval is `+/-1e-12`. Entries are raw values in different
block units; their colors are not a nondimensional comparison. Sparse input is
densified for plotting without modifying the matrix.

Retained matrices require `formulation="retained"` and `num_cells=...` to label
the free-face block correctly. The helper's standalone formulation default is
`"original"`; the runners pass their selection explicitly. The following example
exports the peak Jacobian from a checked retained run:

```python
from examples.plastic_plane_strain import run_example
from prototype.jacobian import RetainedAnalyticalJacobian
from utils.state import TpsaState, as_tpsa_state
from utils.visualization import export_jacobian_png

result = run_example(formulation="retained", jacobian="analytical")
records = result.controller.steps
peak_index = next(i for i, record in enumerate(records) if record.load_factor == 1.0)
peak = records[peak_index]
old = (
    TpsaState.zeros(result.case.grid) if peak_index == 0
    else as_tpsa_state(records[peak_index - 1].state)
)
jacobian = RetainedAnalyticalJacobian(result.operators, result.case.material_points, old)
matrix = jacobian(peak.newton.z, peak.newton.evaluation)
png = export_jacobian_png(
    matrix, formulation="retained", num_cells=result.case.grid.num_cells,
    folder_name="results", file_name="plastic_plane_strain_retained_analytical_jacobian",
)
```

The old state must be the **preceding committed history**, not the peak state
itself. Retained rows are `[R_u, R_r, R_p, R_f]` and columns are
`[du, dr, dp, d_free]`; coupled strain matrices append their strain block to
`[u, r, p]`. Dashed separators mark block boundaries. Both plastic runners
export the selected Jacobian at the converged peak loading increment.

## Compare analytical and finite-difference Jacobians

Run the current nonuniform Dirichlet problem once with the analytical Jacobian,
then evaluate both Jacobians at its same converged peak candidate and preceding
committed history:

```bash
python -m examples.compare_jacobians
```

This uses the Dirichlet runner's current Python mesh default. Override it with
`--cells-per-axis`, and change the local perturbation with `--fd-step` (default
`1e-10`). Add `--adaptive` to use the existing adaptive load controller if the
fixed schedule fails to converge. The summary records the actual load path.
Results go to `results/jacobian_comparison/`, or `--output-dir`:
a signed PNG of `J_finite_difference - J_analytical`, all three sparse matrices
in SciPy NPZ format, a JSON summary and a CSV of residual/unknown block errors.
Relative errors use Frobenius norms; the free-face block is reported separately
because the full matrix mixes block units. Blocks with zero reference norm
have a null JSON / empty CSV relative error, avoiding division by zero.

The difference should occupy only the free-face balance/displacement block.
Both evaluations share the same state and load path; this compares tangent
values, not convergence histories or timings of two separate solves. Use
distinct output directories when varying the mesh or perturbation.
