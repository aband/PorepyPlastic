from copy import deepcopy

import numpy as np
from petsc4py import PETSc

from core.jacobian import AnalyticalJacobian
from core.layout import FloatArray
from core.residual import LoadIncrement, ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory
from utils.newton_utils import NewtonConvergenceError

def solve_newton_petsc(
    assembler: ResidualAssembler,
    loads: LoadIncrement,
    committed: MaterialHistory,
    *,
    initial_z: FloatArray | None = None,
    max_iterations: int = 20,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    committed_residual: FloatArray | None = None,
    jacobian: AnalyticalJacobian | None = None,
) -> tuple[FloatArray, ResidualEvaluation]:
    """Solve one load increment without committing material history."""
    n = assembler.layout.size
    initial = np.zeros(n) if initial_z is None else np.asarray(initial_z)
    assembler._validate_inputs(initial, loads)

    old = committed.copy()
    fixed_loads = deepcopy(loads)

    if jacobian is not None and jacobian.assembler is not assembler:
        raise ValueError("Jacobian must use the supplied residual assembler.")
    tangent = AnalyticalJacobian(assembler)

    cached_z: FloatArray | None = None
    cached_evaluation: ResidualEvaluation | None = None

    size = assembler.layout.size
    offset = (
        np.zeros(size)
        if committed_residual is None
        else np.asarray(committed_residual)
    )

    if (
        offset.shape != (size,)
        or offset.dtype.kind not in ("f", "i", "u")
        or not np.all(np.isfinite(offset))
    ):
        raise ValueError("committed_residual must be finite.")

    fixed_residual = np.array(offset, dtype=np.float64, copy=True)

    def evaluate(x: PETSc.Vec) -> ResidualEvaluation:
        nonlocal cached_z, cached_evaluation

        values = x.getArray(readonly=True)

        if cached_z is None or not np.array_equal(values, cached_z):
#            cached_evaluation = assembler.evaluate(values, fixed_loads, old)
#            cached_z = values.copy()
            candidate = assembler.evaluate(values, fixed_loads, old)
            if candidate.residual.shape != (n,):
                raise RuntimeError("Newton residual has incorrect shape.")

            cached_evaluation = candidate
            cached_z = values.copy()

        assert cached_evaluation is not None
        return cached_evaluation

    def residual_callback(
        snes: PETSc.SNES, x: PETSc.Vec, residual: PETSc.Vec,
    ) -> None:
        total_residual = fixed_residual + evaluate(x).residual
        residual.getArray()[:] = total_residual

    def jacobian_callback(
        snes: PETSc.SNES, x: PETSc.Vec,
        matrix: PETSc.Mat, preconditioner: PETSc.Mat,
    ) -> None:
        csr = tangent.evaluate(old, evaluate(x))

        # Remove values left over from previous plastic states.
        matrix.zeroEntries()
        matrix.setValuesCSR(
            np.asarray(csr.indptr, dtype=PETSc.IntType),
            np.asarray(csr.indices, dtype=PETSc.IntType),
            np.asarray(csr.data, dtype=PETSc.ScalarType),
        )
        matrix.assemble()
        # matrix and preconditioner are registered as the same PETSc object.

    x = PETSc.Vec().createSeq(n, comm=PETSc.COMM_SELF)
    residual = x.duplicate()
    matrix = PETSc.Mat().createAIJ(
        (n, n), nnz=0, comm=PETSc.COMM_SELF,
    )
    snes = PETSc.SNES().create(comm=PETSc.COMM_SELF)

    try:
        x.getArray()[:] = initial

        # Allow the sparsity pattern to grow when cells yield.
        matrix.setOption(
            PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False,
        )

        snes.setFunction(residual_callback, residual)
        snes.setJacobian(jacobian_callback, matrix, matrix)

        snes.setType("newtonls")
        snes.getLineSearch().setType("basic")  # Full Newton steps.
        snes.getKSP().setType("preonly")
        snes.getKSP().getPC().setType("lu")

        snes.setTolerances(
            atol=atol, rtol=rtol, stol=0.0, max_it=max_iterations,
        )

        snes.setOptionsPrefix("global_")
        snes.setFromOptions()
        snes.solve(None, x)

        reason = snes.getConvergedReason()
        if reason <= 0:
            raise NewtonConvergenceError(
                f"PETSc SNES failed: reason={reason}."
            )

        solution = x.getArray(readonly=True).copy()
        return solution, evaluate(x)

    finally:
        snes.destroy()
        matrix.destroy()
        residual.destroy()
        x.destroy()