"""Preconditioned Krylov solvers (PETSc) for the retained-face Newton correction.

The default solves the condensed face system S x_f = b_f of
core.condensation (cell unknowns eliminated exactly), with GMRES and
smoothed-aggregation AMG (GAMG) using rigid-body near-null-space modes.
A full-system path (row/column equilibration + ILU) is kept for comparison.

petsc4py is imported when a solver is constructed, so core works without it.
Every failure (no convergence, PETSc setup error, non-finite solution) raises
NewtonDivergenceError, which the load schedule treats as recoverable, unless
fallback_to_direct is set.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cache
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import scipy.sparse as sps

from core.condensation import CondensedElasticSystem
from core.jacobian import AnalyticalJacobian
from core.layout import FloatArray
from core.linear_solvers import CondensedDirectSolver, LinearSolveResult, _BoundSolver
from core.residual import ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory
from utils.newton_utils import NewtonDivergenceError

KrylovSystem = Literal["condensed", "full"]
KrylovMethod = Literal["gmres", "fgmres", "bcgs"]
KrylovPreconditioner = Literal["gamg", "ilu", "asm", "jacobi"]

_SYSTEMS = ("condensed", "full")
_METHODS = ("gmres", "fgmres", "bcgs")
_PRECONDITIONERS = ("gamg", "ilu", "asm", "jacobi")


@dataclass(frozen=True)
class KrylovOptions:
    """Krylov method, preconditioner and tolerances for one solver instance.

    rtol bounds the true residual, ||A x - b|| <= rtol ||b||, when
    right_preconditioning is set (unpreconditioned norm); otherwise PETSc's
    default left-preconditioned norm is used. petsc_options are passed to
    PETSc under a private prefix, e.g. {"pc_gamg_threshold": "0.02"}; they
    are applied after the named options and may override them.
    """

    system: KrylovSystem = "condensed"
    method: KrylovMethod = "gmres"
    preconditioner: KrylovPreconditioner = "gamg"
    ilu_levels: int = 0
    rtol: float = 1e-6
    max_iterations: int = 500
    restart: int = 200
    right_preconditioning: bool = True
    fallback_to_direct: bool = False
    petsc_options: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("system", self.system, _SYSTEMS),
            ("method", self.method, _METHODS),
            ("preconditioner", self.preconditioner, _PRECONDITIONERS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}; got {value!r}.")
        if not np.isfinite(self.rtol) or not 0.0 < self.rtol < 1.0:
            raise ValueError("rtol must lie strictly between zero and one.")
        for name, number, lower in (
            ("ilu_levels", self.ilu_levels, 0),
            ("max_iterations", self.max_iterations, 1),
            ("restart", self.restart, 1),
        ):
            if isinstance(number, bool) or not isinstance(number, (int, np.integer)) or number < lower:
                raise ValueError(f"{name} must be an integer >= {lower}.")
        if self.method == "fgmres" and not self.right_preconditioning:
            raise ValueError("FGMRES supports right preconditioning only.")
        options = dict(self.petsc_options)
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in options.items()):
            raise ValueError("petsc_options must map option names to string values.")
        if any(k.startswith("-") for k in options):
            raise ValueError("petsc_options names must not start with '-'; the prefix is added.")
        # Freeze a private copy: later changes to the caller's dict are not seen.
        object.__setattr__(self, "petsc_options", MappingProxyType(options))


@dataclass(frozen=True)
class KrylovSolveStats:
    """Diagnostics of one Krylov solve (one Newton correction)."""

    iteration: int          # Newton correction index
    iterations: int         # Krylov iterations
    reason: str             # PETSc convergence reason, e.g. "CONVERGED_RTOL"
    residual_norm: float    # final residual norm reported by PETSc
    setup_time: float       # preconditioner setup
    solve_time: float       # Krylov iterations
    fallback: bool = False  # True if CondensedDirectSolver produced the correction


def _import_petsc() -> Any:
    try:
        from petsc4py import PETSc
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("KrylovSolver requires petsc4py.") from error
    return PETSc


def _equilibrate(matrix: sps.csr_array) -> tuple[FloatArray, FloatArray]:
    """Row then column max-scaling; rows of J differ by about 1e20 in size."""
    rows = np.asarray(abs(matrix).max(axis=1).toarray(), dtype=np.float64).ravel()
    if np.any(rows == 0.0):
        raise NewtonDivergenceError("Jacobian has an empty row; cannot equilibrate.")
    r = 1.0 / rows
    cols = np.asarray(abs(sps.diags_array(r) @ matrix).max(axis=0).toarray(), dtype=np.float64).ravel()
    if np.any(cols == 0.0):
        raise NewtonDivergenceError("Jacobian has an empty column; cannot equilibrate.")
    return r, 1.0 / cols


class KrylovSolver(_BoundSolver):
    """GMRES/BiCGStab with GAMG, ILU, ASM or Jacobi preconditioning (PETSc).

    One KSP object per solver, created in bind and reused for every solve.
    A new PETSc matrix is created per solve because the plastic face block
    changes the sparsity pattern; the preconditioner is rebuilt each time
    (reuse across Newton iterations is a later extension via begin_step).
    Call close() (or use the solver as a context manager) to free PETSc memory.
    """

    def __init__(self, options: KrylovOptions | None = None) -> None:
        if options is not None and not isinstance(options, KrylovOptions):
            raise ValueError("options must be KrylovOptions or None.")
        self.options = KrylovOptions() if options is None else options
        self._petsc = _import_petsc()
        self._prefix = f"core_krylov_{id(self):x}_"
        self._ksp: Any = None
        self._near_null_space: Any = None
        self._system: CondensedElasticSystem | None = None
        self._block_size = 1
        self._fallback: CondensedDirectSolver | None = None
        self._closed = False
        self.stats: list[KrylovSolveStats] = []

    # ------------------------------------------------------------------ setup
    def bind(self, assembler: ResidualAssembler, jacobian: AnalyticalJacobian) -> None:
        if self._closed:
            raise RuntimeError("KrylovSolver has been closed.")
        rebind = self._jacobian is not jacobian
        super().bind(assembler, jacobian)
        if not rebind:
            return
        options = self.options
        self._block_size = assembler.layout.dim
        if options.system == "condensed":
            self._system = jacobian.condensed_elastic()
        if options.fallback_to_direct:
            self._fallback = CondensedDirectSolver()
            self._fallback.bind(assembler, jacobian)
        self._destroy()
        self._ksp = self._create_ksp()
        if options.system == "condensed" and options.preconditioner == "gamg":
            self._near_null_space = self._rigid_body_modes(assembler)

    def _create_ksp(self) -> Any:
        PETSc, options = self._petsc, self.options
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOptionsPrefix(self._prefix)
        ksp.setType(options.method)
        if options.method in ("gmres", "fgmres"):
            ksp.setGMRESRestart(options.restart)
        ksp.setTolerances(rtol=options.rtol, atol=0.0, max_it=options.max_iterations)
        ksp.setInitialGuessNonzero(False)
        if options.right_preconditioning:
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        pc = ksp.getPC()
        pc.setType(options.preconditioner)
        if options.preconditioner == "ilu":
            pc.setFactorLevels(options.ilu_levels)

        # Extra options under a private prefix, read once, then removed so they
        # neither leak into other solvers nor persist in the global database.
        database = PETSc.Options(self._prefix)
        try:
            for name, value in options.petsc_options.items():
                database[name] = value
            ksp.setFromOptions()
        finally:
            for name in options.petsc_options:
                database.delValue(name)
        return ksp

    def _rigid_body_modes(self, assembler: ResidualAssembler) -> Any:
        """Near-null space for GAMG: free-face unknowns are stored face by face."""
        PETSc, layout = self._petsc, assembler.layout
        if layout.free_faces.size == 0:
            return None
        centers = assembler.operators.grid.face_centers[: layout.dim, layout.free_faces]
        coordinates = PETSc.Vec().createWithArray(
            np.ascontiguousarray(centers.ravel(order="F"), dtype=PETSc.ScalarType),
            bsize=layout.dim, comm=PETSc.COMM_SELF,
        )
        try:
            return PETSc.NullSpace().createRigidBody(coordinates)
        finally:
            coordinates.destroy()

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        committed: MaterialHistory,
        evaluation: ResidualEvaluation,
        residual: FloatArray,
        *,
        iteration: int,
    ) -> LinearSolveResult:
        if self._closed:
            raise RuntimeError("KrylovSolver has been closed.")
        jacobian = self._require_jacobian()
        start, recorded = time.perf_counter(), len(self.stats)
        rhs = -np.asarray(residual, dtype=np.float64)
        try:
            if self.options.system == "condensed":
                assert self._system is not None  # set together with the Jacobian in bind
                correction = self._system.solve(
                    rhs,
                    jacobian.plastic_face_block(committed, evaluation),
                    lambda S, b: self._krylov(S, b, iteration=iteration, block_size=self._block_size),
                )
            else:
                matrix = sps.csr_array(jacobian.evaluate(committed, evaluation))
                r, c = _equilibrate(matrix)
                scaled = sps.csr_array(sps.diags_array(r) @ matrix @ sps.diags_array(c))
                correction = c * self._krylov(scaled, r * rhs, iteration=iteration, block_size=None)
        except NewtonDivergenceError:
            if self._fallback is None or len(self.stats) == recorded:
                raise  # no fallback, or failure before any Krylov solve (e.g. equilibration)
            correction = self._fallback.solve(
                committed, evaluation, residual, iteration=iteration,
            ).correction
            self.stats[-1] = replace(self.stats[-1], fallback=True)
        iterations = self.stats[-1].iterations if len(self.stats) > recorded else 0
        return LinearSolveResult(correction, iterations, time.perf_counter() - start)

    def _krylov(
        self,
        matrix: sps.csr_array,
        rhs: FloatArray,
        *,
        iteration: int,
        block_size: int | None,
    ) -> FloatArray:
        """Solve matrix x = rhs with the bound KSP; raise NewtonDivergenceError on failure."""
        PETSc, ksp = self._petsc, self._ksp
        A = sps.csr_array(matrix, copy=True)   # never touch cached read-only storage
        A.sum_duplicates()                     # canonical: sorted, no duplicates
        size = A.shape[0]
        mat = PETSc.Mat().createAIJ(
            size=(size, size),
            bsize=block_size if block_size and size % block_size == 0 else None,
            csr=(A.indptr.astype(PETSc.IntType), A.indices.astype(PETSc.IntType), A.data),
            comm=PETSc.COMM_SELF,
        )
        if self._near_null_space is not None and block_size is not None:
            mat.setNearNullSpace(self._near_null_space)
        b = PETSc.Vec().createWithArray(np.ascontiguousarray(rhs, dtype=PETSc.ScalarType),
                                        comm=PETSc.COMM_SELF)
        x = mat.createVecRight()
        t0 = time.perf_counter()
        try:
            ksp.setOperators(mat)
            ksp.setUp()
            t1 = time.perf_counter()
            ksp.solve(b, x)
            t2 = time.perf_counter()
            solution = np.array(x.getArray(), dtype=np.float64, copy=True)
        except PETSc.Error as error:
            self.stats.append(KrylovSolveStats(
                iteration, ksp.getIterationNumber(), f"PETSC_ERROR_{error.ierr}", float("nan"),
                time.perf_counter() - t0, 0.0,
            ))
            raise NewtonDivergenceError(
                f"Krylov setup/solve failed at correction {iteration + 1} "
                f"(PETSc error {error.ierr})."
            ) from error
        finally:
            for item in (mat, b, x):
                item.destroy()

        reason_code = ksp.getConvergedReason()
        reason = _reason_names().get(reason_code, str(reason_code))
        stats = KrylovSolveStats(
            iteration, ksp.getIterationNumber(), reason, float(ksp.getResidualNorm()),
            t1 - t0, t2 - t1,
        )
        self.stats.append(stats)
        if reason_code <= 0 or not np.all(np.isfinite(solution)):
            raise NewtonDivergenceError(
                f"Krylov solve did not converge at correction {iteration + 1}: {reason} "
                f"after {stats.iterations} iterations, residual norm {stats.residual_norm:.3e}."
            )
        return solution

    # ---------------------------------------------------------------- cleanup
    def _destroy(self) -> None:
        for name in ("_ksp", "_near_null_space"):
            item = getattr(self, name, None)
            if item is not None:
                item.destroy()
                setattr(self, name, None)

    def close(self) -> None:
        """Free PETSc objects; the solver cannot be used afterwards."""
        self._destroy()
        self._closed = True

    def __enter__(self) -> KrylovSolver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._destroy()
        except Exception:  # interpreter shutdown: PETSc may be finalized already
            pass

@cache
def _reason_names() -> dict[int, str]:
    """PETSc convergence-reason names, built on first use (PETSc is optional)."""
    PETSc = _import_petsc()
    return {
        value: name for name, value in vars(PETSc.KSP.ConvergedReason).items()
        if isinstance(value, int) and not name.startswith("_")
    }
