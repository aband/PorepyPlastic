# Development plan

Current development focuses on the dimension-aware retained-displacement solver
in `core/`. The existing solver in `prototype/` and the original and coupled
formulations in `alternatives/` remain available for reference and comparisons.
See the [solver guide](docs/retained_solver.md) for the prototype algorithm and
its current scope.

## Current work: dimension-aware core

- [x] Add permanent integration tests for `ResidualAssembler.evaluate()` in 2D
  and 3D, covering Dirichlet, Neumann and mixed boundaries; elastic loading,
  yielding, unloading and reloading; and consistency of the returned material
  trial, corrected tractions and all residual blocks. Verify repeatability,
  preservation of committed history and forwarding of constitutive tolerances.
- [ ] As an optional efficiency improvement, allow
  `RetainedFaceOperators.elastic_one_sided_traction()` to reuse the full face
  increments already computed by `ResidualAssembler.evaluate()`, avoiding a
  second call to `face_displacement_increment()` while preserving existing callers.

## Completed prototype foundation

- [x] Reuse the 3D J2 return map, hardening laws and consistent material tangent.
- [x] Implement retained-face kinematics, one-sided tractions and all four global
  residual blocks.
- [x] Provide analytical and local finite-difference Jacobians.
- [x] Add nested Newton with optional Armijo backtracking, fixed/adaptive load
  control, complete state acceptance, rollback and restart.
- [x] Generalize spatial Dirichlet displacement input through a boundary callback.
- [x] Provide homogeneous numerical checks and a nonuniform Dirichlet example.
- [x] Export VTK histories, displacement/strain PNGs, yield markers and Jacobian
  heatmaps with formulation/Jacobian labels.
- [x] Separate `prototype/`, `material/`, `utils/`, `alternatives/` and `examples/`;
  organize tests to match and document the current entry points.

## Prototype: reproducible Jacobian comparison

- [ ] Add a nonuniform benchmark comparing analytical and finite-difference
  Jacobians for the retained formulation.
- [ ] Use identical meshes, materials, boundary data, fixed load targets and
  convergence tolerances for the baseline comparison.
- [ ] Record solution differences, accepted/rejected steps, Newton iterations,
  residual evaluations and elapsed time; study finite-difference step sensitivity.
- [ ] Compare adaptive runs separately, since their accepted load paths can differ.
- [ ] Store benchmark settings and numerical summaries in a reproducible format.

The homogeneous reference and derivative tests verify specific algebraic and
constitutive behavior. They do not establish spatial convergence of nonuniform
plastic solutions or a performance advantage for either Jacobian.

## Further numerical verification

- [ ] Study mesh refinement and supported mesh variations for nonuniform loading.
- [ ] Verify heterogeneous elastic/plastic properties and the associated
  one-sided traction and transfer weights.
- [ ] Investigate robustness near incompressibility, including pressure-block
  conditioning and the effect of residual scaling.

## Planned extensions

- [ ] Add Neumann and Robin boundary conditions with matching residuals,
  Jacobians, reaction checks and examples.
- [ ] Extend the global plane-strain discretization to three dimensions.

## Deferred

- [ ] Implement least-squares strain reconstruction and compare it with
  Green–Gauss reconstruction.
- [ ] Implement and assess the Newton–Schur solver separately from the current
  nested solver.
