"""Preconditioned Krylov solvers (PETSc) for the retained-face Newton correction.

The default solves the condensed face system S x_f = b_f of
core.condensation (cell unknowns eliminated exactly), with GMRES and
smoothed-aggregation AMG (GAMG) using rigid-body near-null-space modes.
A full-system path (row/column equilibration + ILU) is kept for comparison.

Efficiency options (all off by default except the loose fixed tolerance):
    forcing="eisenstat-walker"  adaptive Krylov tolerance per Newton correction;
    matrix_update="in_place"    one PETSc matrix on a fixed pattern, values updated
                                in place, so GAMG reuses its interpolation;
    solve_elastic()             elastic predictor for LoadStepController(predictor="elastic").

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

from core.condensation import CondensedElasticSystem, FixedFacePattern
from core.jacobian import AnalyticalJacobian
from core.layout import FloatArray
from core.linear_solvers import CondensedDirectSolver, LinearSolveResult, _BoundSolver
from core.residual import ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory
from utils.newton_utils import NewtonDivergenceError

KrylovSystem = Literal["condensed", "full"]
KrylovMethod = Literal["gmres", "fgmres", "bcgs"]
KrylovPreconditioner = Literal["gamg", "ilu", "asm", "jacobi"]
KrylovForcing = Literal["fixed", "eisenstat-walker"]
KrylovMatrixUpdate = Literal["rebuild", "in_place"]
KrylovRefresh = Literal["never", "step"]

_SYSTEMS = ("condensed", "full")
_METHODS = ("gmres", "fgmres", "bcgs")
_PRECONDITIONERS = ("gamg", "ilu", "asm", "jacobi")
_FORCING = ("fixed", "eisenstat-walker")
_MATRIX_UPDATES = ("rebuild", "in_place")
_REFRESH = ("never", "step")

PREDICTOR_ITERATION = -1   # KrylovSolveStats.iteration of elastic-predictor solves


@dataclass(frozen=True)
class KrylovOptions:
    """Krylov method, preconditioner and tolerances for one solver instance.

    rtol bounds the true residual, ||A x - b|| <= rtol ||b||, when
    right_preconditioning is set (unpreconditioned norm); otherwise PETSc's
    default left-preconditioned norm is used. The default 1e-3 is loose on
    purpose: Newton's own residual test decides convergence, and tighter
    solves gave the same Newton iterations at up to 1.7x the cost.

    forcing="eisenstat-walker" chooses the tolerance per correction,
        eta_k = ew_gamma * (||b_k|| / ||b_(k-1)||) ** ew_alpha,
    with the standard safeguard, clamped to [rtol, rtol_max]; b_k is the
    right-hand side of the system GMRES solves. The first correction of a
    load step uses rtol_max.

    matrix_update="in_place" (condensed system only) keeps one PETSc matrix
    on the fixed pattern of core.condensation.FixedFacePattern and updates its
    values; GAMG then reuses its interpolation. refresh_interpolation="step"
    rebuilds that matrix, and so the whole preconditioner, at each load step.

    petsc_options are passed to PETSc under a private prefix, e.g.
    {"pc_gamg_threshold": "0.02"}; they are applied after the named options
    and may override them.
    """

    system: KrylovSystem = "condensed"
    method: KrylovMethod = "gmres"
    preconditioner: KrylovPreconditioner = "gamg"
    ilu_levels: int = 0
    rtol: float = 1e-3
    max_iterations: int = 500
    restart: int = 200
    right_preconditioning: bool = True
    fallback_to_direct: bool = False
    forcing: KrylovForcing = "fixed"
    rtol_max: float = 1e-2
    ew_gamma: float = 0.9
    ew_alpha: float = 0.5 * (1.0 + 5.0 ** 0.5)
    matrix_update: KrylovMatrixUpdate = "rebuild"
    refresh_interpolation: KrylovRefresh = "never"
    petsc_options: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("system", self.system, _SYSTEMS),
            ("method", self.method, _METHODS),
            ("preconditioner", self.preconditioner, _PRECONDITIONERS),
            ("forcing", self.forcing, _FORCING),
            ("matrix_update", self.matrix_update, _MATRIX_UPDATES),
            ("refresh_interpolation", self.refresh_interpolation, _REFRESH),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}; got {value!r}.")
        for name, tolerance in (("rtol", self.rtol), ("rtol_max", self.rtol_max)):
            if not _is_real(tolerance) or not 0.0 < tolerance < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one.")
        if self.forcing == "eisenstat-walker" and self.rtol > self.rtol_max:
            raise ValueError("rtol (lower bound) must not exceed rtol_max with Eisenstat-Walker forcing.")
        if not _is_real(self.ew_gamma) or not 0.0 < self.ew_gamma <= 1.0:
            raise ValueError("ew_gamma must lie in (0, 1].")
        if not _is_real(self.ew_alpha) or not 1.0 < self.ew_alpha <= 2.0:
            raise ValueError("ew_alpha must lie in (1, 2].")
        for name, number, lower in (
            ("ilu_levels", self.ilu_levels, 0),
            ("max_iterations", self.max_iterations, 1),
            ("restart", self.restart, 1),
        ):
            if isinstance(number, bool) or not isinstance(number, (int, np.integer)) or number < lower:
                raise ValueError(f"{name} must be an integer >= {lower}.")
        if self.method == "fgmres" and not self.right_preconditioning:
            raise ValueError("FGMRES supports right preconditioning only.")
        if self.matrix_update == "in_place" and self.system != "condensed":
            raise ValueError("matrix_update='in_place' requires system='condensed'.")
        if self.refresh_interpolation == "step" and self.matrix_update != "in_place":
            raise ValueError("refresh_interpolation='step' requires matrix_update='in_place'.")
        options = dict(self.petsc_options)
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in options.items()):
            raise ValueError("petsc_options must map option names to string values.")
        if any(k.startswith("-") for k in options):
            raise ValueError("petsc_options names must not start with '-'; the prefix is added.")
        # Freeze a private copy: later changes to the caller's dict are not seen.
        object.__setattr__(self, "petsc_options", MappingProxyType(options))


def _is_real(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


@dataclass(frozen=True)
class KrylovSolveStats:
    """Diagnostics of one Krylov solve (one Newton correction or predictor solve)."""

    iteration: int          # Newton correction index; PREDICTOR_ITERATION for solve_elastic
    iterations: int         # Krylov iterations
    reason: str             # PETSc convergence reason, e.g. "CONVERGED_RTOL"
    residual_norm: float    # final residual norm reported by PETSc
    setup_time: float       # preconditioner setup
    solve_time: float       # Krylov iterations
    fallback: bool = False  # True if CondensedDirectSolver produced the correction
    rtol: float = float("nan")   # relative tolerance used for this solve


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
    With matrix_update="rebuild" a new PETSc matrix (and preconditioner) is
    built per correction; with "in_place" one matrix on a fixed pattern is
    kept and only its values change. A second KSP on the elastic face matrix
    serves solve_elastic; its preconditioner is built once per run.
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
        self._pattern: FixedFacePattern | None = None
        self._mat: Any = None                 # in-place face matrix
        self._elastic_ksp: Any = None         # predictor: KSP and matrix on S_el
        self._elastic_mat: Any = None
        self._block_size = 1
        self._fallback: CondensedDirectSolver | None = None
        self._previous_norm: float | None = None
        self._previous_eta: float | None = None
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
        # Full-system solvers build the condensation only if solve_elastic needs it.
        self._system = jacobian.condensed_elastic() if options.system == "condensed" else None
        if options.matrix_update == "in_place":
            self._pattern = jacobian.fixed_face_pattern()
        if options.fallback_to_direct:
            self._fallback = CondensedDirectSolver()
            self._fallback.bind(assembler, jacobian)
        self._destroy()
        self._ksp = self._create_ksp(self._prefix, reuse_interpolation=options.matrix_update == "in_place")
        if options.preconditioner == "gamg" and options.system == "condensed":
            self._near_null_space = self._rigid_body_modes(assembler)

    def begin_step(self) -> None:
        """New load step: restart the forcing sequence; optionally rebuild the matrix."""
        self._previous_norm = self._previous_eta = None
        if self.options.refresh_interpolation == "step" and self._mat is not None:
            # A new matrix object has a new nonzero state: PETSc rebuilds the whole PC.
            self._mat.destroy()
            self._mat = None

    def _create_ksp(self, prefix: str, *, reuse_interpolation: bool = False) -> Any:
        PETSc, options = self._petsc, self.options
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOptionsPrefix(prefix)
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
        # GAMG interpolation reuse is what makes in-place updates pay off; set it
        # explicitly instead of relying on PETSc's default (user options win).
        extra = dict(options.petsc_options)
        if reuse_interpolation and options.preconditioner == "gamg":
            extra = {"pc_gamg_reuse_interpolation": "true", **extra}
        database = PETSc.Options(prefix)
        try:
            for name, value in extra.items():
                database[name] = value
            ksp.setFromOptions()
        finally:
            for name in extra:
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

    def _face_matrix(self, indptr: Any, indices: Any, data: FloatArray, size: int) -> Any:
        """PETSc AIJ face matrix with block size dim and the rigid-body near-null space."""
        PETSc = self._petsc
        bsize = self._block_size if size % self._block_size == 0 else None
        mat = PETSc.Mat().createAIJ(
            size=(size, size), bsize=bsize,
            csr=(indptr, indices, np.ascontiguousarray(data, dtype=np.float64)),
            comm=PETSc.COMM_SELF,
        )
        if self._near_null_space is not None:
            mat.setNearNullSpace(self._near_null_space)
        return mat

    # ---------------------------------------------------------------- forcing
    def _forcing(self, norm: float) -> float:
        """Relative tolerance for the next solve, given its right-hand-side norm."""
        options = self.options
        if options.forcing == "fixed":
            return options.rtol
        previous, previous_eta = self._previous_norm, self._previous_eta
        if previous is None or previous_eta is None:
            eta = options.rtol_max
        elif previous == 0.0:
            eta = options.rtol
        else:
            eta = options.ew_gamma * (norm / previous) ** options.ew_alpha
            guard = options.ew_gamma * previous_eta ** options.ew_alpha
            if guard > 0.1:   # Eisenstat-Walker safeguard against early over-tightening
                eta = max(eta, guard)
        eta = float(min(max(eta, options.rtol), options.rtol_max))
        self._previous_norm, self._previous_eta = norm, eta
        return eta

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
                correction = self._solve_condensed(jacobian, committed, evaluation, rhs, iteration)
            else:
                matrix = sps.csr_array(jacobian.evaluate(committed, evaluation))
                r, c = _equilibrate(matrix)
                scaled = sps.csr_array(sps.diags_array(r) @ matrix @ sps.diags_array(c))
                b = r * rhs
                rtol = self._forcing(float(np.linalg.norm(b)))
                correction = c * self._krylov(scaled, b, iteration=iteration, block_size=None, rtol=rtol)
        except NewtonDivergenceError:
            if self._fallback is None or len(self.stats) == recorded:
                raise  # no fallback, or failure before any Krylov solve (e.g. equilibration)
            correction = self._fallback.solve(
                committed, evaluation, residual, iteration=iteration,
            ).correction
            self.stats[-1] = replace(self.stats[-1], fallback=True)
        iterations = self.stats[-1].iterations if len(self.stats) > recorded else 0
        return LinearSolveResult(correction, iterations, time.perf_counter() - start)

    def _solve_condensed(
        self,
        jacobian: AnalyticalJacobian,
        committed: MaterialHistory,
        evaluation: ResidualEvaluation,
        rhs: FloatArray,
        iteration: int,
    ) -> FloatArray:
        system = self._system
        assert system is not None  # set together with the Jacobian in bind
        b_f, b_c = system.reduce(rhs)
        if system.num_face_dofs == 0:
            return system.recover(np.empty(0), b_c)
        rtol = self._forcing(float(np.linalg.norm(b_f)))
        if self._pattern is None:
            S = system.face_matrix(jacobian.plastic_face_block(committed, evaluation))
            x_f = self._krylov(S, b_f, iteration=iteration, block_size=self._block_size, rtol=rtol)
        else:
            mat = self._update_fixed_matrix(jacobian.fixed_face_values(committed, evaluation))
            x_f = self._ksp_solve(self._ksp, mat, b_f, iteration=iteration, rtol=rtol)
        return system.recover(x_f, b_c)

    def _update_fixed_matrix(self, values: FloatArray) -> Any:
        """Create the in-place matrix once; afterwards only its values change."""
        PETSc, pattern = self._petsc, self._pattern
        assert pattern is not None
        indptr = pattern.indptr.astype(PETSc.IntType)
        indices = pattern.indices.astype(PETSc.IntType)
        if self._mat is None:
            self._mat = self._face_matrix(indptr, indices, values, pattern.num_face_dofs)
            # Any entry outside the fixed pattern is a bug; never reallocate silently.
            self._mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        else:
            self._mat.setValuesCSR(indptr, indices, np.ascontiguousarray(values, dtype=np.float64))
            self._mat.assemble()
        return self._mat

    def _krylov(
        self,
        matrix: sps.csr_array,
        rhs: FloatArray,
        *,
        iteration: int,
        block_size: int | None,
        rtol: float | None = None,
    ) -> FloatArray:
        """Solve matrix x = rhs with a new PETSc matrix and the bound KSP."""
        PETSc = self._petsc
        A = sps.csr_array(matrix, copy=True)   # never touch cached read-only storage
        A.sum_duplicates()                     # canonical: sorted, no duplicates
        size = A.shape[0]
        indptr, indices = A.indptr.astype(PETSc.IntType), A.indices.astype(PETSc.IntType)
        if block_size is None:
            mat = PETSc.Mat().createAIJ(
                size=(size, size), csr=(indptr, indices, A.data), comm=PETSc.COMM_SELF,
            )
        else:
            mat = self._face_matrix(indptr, indices, A.data, size)
        try:
            return self._ksp_solve(self._ksp, mat, rhs, iteration=iteration, rtol=rtol)
        finally:
            mat.destroy()

    def _ksp_solve(
        self, ksp: Any, mat: Any, rhs: FloatArray, *, iteration: int, rtol: float | None,
    ) -> FloatArray:
        """Solve mat x = rhs; record stats; raise NewtonDivergenceError on failure."""
        PETSc, options = self._petsc, self.options
        rtol = options.rtol if rtol is None else rtol
        ksp.setTolerances(rtol=rtol, atol=0.0, max_it=options.max_iterations)
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
                time.perf_counter() - t0, 0.0, rtol=rtol,
            ))
            raise NewtonDivergenceError(
                f"Krylov setup/solve failed at {_where(iteration)} (PETSc error {error.ierr})."
            ) from error
        finally:
            b.destroy()
            x.destroy()

        reason_code = ksp.getConvergedReason()
        reason = _reason_names().get(reason_code, str(reason_code))
        stats = KrylovSolveStats(
            iteration, ksp.getIterationNumber(), reason, float(ksp.getResidualNorm()),
            t1 - t0, t2 - t1, rtol=rtol,
        )
        self.stats.append(stats)
        if reason_code <= 0 or not np.all(np.isfinite(solution)):
            raise NewtonDivergenceError(
                f"Krylov solve did not converge at {_where(iteration)}: {reason} "
                f"after {stats.iterations} iterations, residual norm {stats.residual_norm:.3e}."
            )
        return solution

    # -------------------------------------------------------------- predictor
    def solve_elastic(self, rhs: FloatArray) -> LinearSolveResult:
        """Solve L x = rhs (elastic Jacobian) on the condensed elastic face system.

        Uses a second KSP on S_el whose preconditioner is set up once per run,
        so it never disturbs the Newton KSP's (possibly reused) preconditioner.
        The tolerance is options.rtol: the result only starts Newton.
        """
        if self._closed:
            raise RuntimeError("KrylovSolver has been closed.")
        jacobian = self._require_jacobian()
        if self._system is None:
            self._system = jacobian.condensed_elastic()
        system = self._system
        start, recorded = time.perf_counter(), len(self.stats)
        b_f, b_c = system.reduce(np.asarray(rhs, dtype=np.float64))
        if system.num_face_dofs == 0:
            return LinearSolveResult(system.recover(np.empty(0), b_c), 0, time.perf_counter() - start)
        try:
            if self._elastic_ksp is None:
                self._create_elastic_solver(system)
            x_f = self._ksp_solve(
                self._elastic_ksp, self._elastic_mat, b_f,
                iteration=PREDICTOR_ITERATION, rtol=self.options.rtol,
            )
            correction = system.recover(x_f, b_c)
        except NewtonDivergenceError:
            if self._fallback is None or len(self.stats) == recorded:
                raise
            correction = self._fallback.solve_elastic(rhs).correction
            self.stats[-1] = replace(self.stats[-1], fallback=True)
        iterations = self.stats[-1].iterations if len(self.stats) > recorded else 0
        return LinearSolveResult(correction, iterations, time.perf_counter() - start)

    def _create_elastic_solver(self, system: CondensedElasticSystem) -> None:
        PETSc, assembler = self._petsc, self._assembler
        assert assembler is not None
        if self.options.preconditioner == "gamg" and self._near_null_space is None:
            # Full-system solvers have no face near-null space yet; the predictor needs one.
            self._near_null_space = self._rigid_body_modes(assembler)
        S = system.S_elastic
        self._elastic_mat = self._face_matrix(
            S.indptr.astype(PETSc.IntType), S.indices.astype(PETSc.IntType),
            np.array(S.data, dtype=np.float64), system.num_face_dofs,
        )
        self._elastic_ksp = self._create_ksp(f"{self._prefix}elastic_")

    # ---------------------------------------------------------------- cleanup
    def _destroy(self) -> None:
        for name in ("_ksp", "_mat", "_elastic_ksp", "_elastic_mat", "_near_null_space"):
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


def _where(iteration: int) -> str:
    return "the elastic predictor" if iteration == PREDICTOR_ITERATION else f"correction {iteration + 1}"


@cache
def _reason_names() -> dict[int, str]:
    """PETSc convergence-reason names, built on first use (PETSc is optional)."""
    PETSc = _import_petsc()
    return {
        value: name for name, value in vars(PETSc.KSP.ConvergedReason).items()
        if isinstance(value, int) and not name.startswith("_")
    }
