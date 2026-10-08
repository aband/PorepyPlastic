# Development Plan

## Implementation

1. Implement adaptive load incrementation with automatic step-size control.
2. Implement least-squares strain reconstruction and compare it with the
   Green–Gauss approach.
3. Extend boundary-condition support to include Neumann and Robin conditions.
4. Extend 2D planeray case to 3D.

## Numerical Verification

1. Develop a benchmark that distinguishes the convergence behavior and
   computational performance of the analytical and finite-difference Jacobians.
2. Verify the solver on different mesh configurations and spatially heterogeneous
   material properties, with particular attention to cell-to-face transfer weights.
3. Investigate solver robustness near the incompressible limit and address
   potential singularity of the pressure block `A_pp`.
