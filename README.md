# PorepyPlastic

Small-strain J2 elastoplasticity for PorePy, using TPSA-based plane-strain
formulations. The main solver in `prototype/` retains cell fields and interior-face
displacements, with constitutive return maps nested inside global Newton.
The original and coupled strain formulations remain available for comparison.

The current plastic examples use full displacement Dirichlet conditions on a
unit-square Cartesian grid. Material updates retain full 3D stress and plastic
history under the plane-strain constraint.

## Setup and quick start

Use a Python environment with PorePy and its dependencies installed. With the
PorePy source checkout alongside this repository, install it with:

```bash
python -m pip install -e ../porepy
python -m pip install pytest mypy
```

Run commands from the PorepyPlastic repository root:

```bash
# Homogeneous loading/unloading with independent numerical checks.
python -m examples.plastic_plane_strain --no-export

# Nonuniform prescribed displacement with adaptive load stepping.
python -m examples.dirichlet_example --adaptive --no-export
```

Both plastic runners default to `--formulation retained --jacobian analytical`.
Omit `--no-export` to write VTK/PVD collections and PNG figures to `results/`.
Use `--output-dir PATH` for a separate run directory.

## Examples and solver options

| Command | Purpose |
| --- | --- |
| `python -m examples.plastic_plane_strain` | Homogeneous plastic loading and elastic unloading, checked against an independent reference |
| `python -m examples.dirichlet_example` | Nonuniform boundary displacement; no exact interior reference is assumed |
| `python -m examples.compare_jacobians --adaptive` | Analytical/finite-difference Jacobian difference at the same Dirichlet peak state |
| `python -m examples.elastic_plane_strain` | Elastic TPSA baseline and strain reconstruction |
| `python -m examples.visualize_newton_stress --help` | Animation of the local constitutive return-map iteration |

The two plastic examples select the formulation independently of the Jacobian:

| `--formulation` | Package | Global Newton unknowns |
| --- | --- | --- |
| `retained` (default) | `prototype/` | Cell increments `du, dr, dp` and free-face displacement increments `d_free` |
| `original` | `alternatives/original/` | Cell displacement, rotation stress and pressure `u, r, p` |
| `coupled` | `alternatives/coupled/` | Cell `u, r, p` and three in-plane Mandel strain components |

`--jacobian analytical` uses the consistent material tangent.
`--jacobian finite-difference` approximates the **local material tangent** and
uses the same sparse global chain rule. It does not finite-difference the whole
global residual. For example:

```bash
python -m examples.plastic_plane_strain --jacobian finite-difference --no-export
python -m examples.plastic_plane_strain --formulation original --no-export
python -m examples.plastic_plane_strain --formulation coupled --no-export
```

Numerical checks always run in the homogeneous example. Their discrete L2 error
summary is hidden by default; add `--show-l2-errors` to print it. Newton block
norms and convergence rates are separate diagnostics. See each command's
`--help` for its supported options.

## Repository layout

| Path | Responsibility |
| --- | --- |
| [`prototype/`](prototype/) | Retained operators, residual, Jacobian, Newton solver, state and load controller |
| [`material/`](material/) | J2 return map, material history, hardening laws, tensors and local tangents |
| [`utils/`](utils/) | Problem setup, common state, reconstruction, transfer, load schedules and visualization |
| [`alternatives/`](alternatives/) | Optional original/coupled formulations and elastic TPSA setup |
| [`examples/`](examples/) | Runnable scripts and formulation dispatch |
| [`test/`](test/) | Tests organized to mirror the source packages |
| [`docs/`](docs/) | Solver, example and output guides |

Import modules from their owning package, for example
`from prototype.operators import RetainedFaceOperators`. The main workflow uses
`utils.problem.PlaneStrainProblem`; it does not assemble the optional elastic
TPSA system. Alternative modules are loaded only when selected.

## Documentation

- [Architecture, state and material conventions](docs/architecture.md)
- [Retained solver: residual, Jacobian, Newton and load stepping](docs/retained_solver.md)
- [Original and coupled formulations](docs/alternative_formulations.md)
- [Example settings, boundary callbacks and numerical checks](docs/examples.md)
- [VTK/PNG output, tensor plots and Jacobian heatmaps](docs/visualization.md)
- [Development priorities and deferred work](TODO.md)

## Verification and generated results

```bash
python -m pytest -q
python -m pytest test/prototype -q
python -m pytest test/material test/utils -q
python -m pytest test/alternatives/original -q
python -m mypy
```

`mypy.ini` selects the modules checked in strict mode. Manual inspection scripts
named `manual_*.py` are outside pytest discovery.

Generated outputs and PDFs are ignored by Git. To remove the repository's
entire `results/` directory, run:

```bash
./clean_results.sh
```

The script leaves custom output directories alone. Exporters recreate
`results/` on the next run.
