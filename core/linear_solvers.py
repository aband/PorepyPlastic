"""Pluggable linear solvers for the retained-face Newton correction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from core.condensation import CondensedElasticSystem
from core.jacobian import AnalyticalJacobian
from core.layout import FloatArray
from core.residual import ResidualAssembler, ResidualEvaluation
from core.state import MaterialHistory
from utils.newton_utils import NewtonDivergenceError, _solve_linear_correction


@dataclass(frozen=True, eq=False)
class LinearSolveResult:
    """Correction solving J * correction = -residual, with diagnostics."""

    correction: FloatArray
    iterations: int = 0            # Krylov iterations; 0 for direct solves
    elapsed: float = 0.0           # wall time including assembly and setup


@runtime_checkable
class LinearSolver(Protocol):
    def bind(self, assembler: ResidualAssembler, jacobian: AnalyticalJacobian) -> None:
        """Attach fixed operators once; repeated calls with the same objects are no-ops."""

    def begin_step(self) -> None:
        """Called once per load step, before the first correction."""

    def solve(
        self,
        committed: MaterialHistory,
        evaluation: ResidualEvaluation,
        residual: FloatArray,
        *,
        iteration: int,
    ) -> LinearSolveResult:
        """Return the Newton correction; raise NewtonDivergenceError on failure."""


class _BoundSolver:
    """Shared binding checks: one solver instance serves one assembler."""

    _assembler: ResidualAssembler | None = None
    _jacobian: AnalyticalJacobian | None = None

    def bind(self, assembler: ResidualAssembler, jacobian: AnalyticalJacobian) -> None:
        if jacobian.assembler is not assembler:
            raise ValueError("Jacobian must use the supplied residual assembler.")
        if self._assembler is not None and self._assembler is not assembler:
            raise ValueError("Linear solver is already bound to another assembler.")
        self._assembler, self._jacobian = assembler, jacobian

    def begin_step(self) -> None:
        pass

    def _require_jacobian(self) -> AnalyticalJacobian:
        if self._jacobian is None:
            raise RuntimeError("Linear solver must be bound before solving.")
        return self._jacobian


class DirectSolver(_BoundSolver):
    """Assemble the full Jacobian and factor it (UMFPACK, SuperLU fallback)."""

    def solve(
            self, 
            committed: MaterialHistory, 
            evaluation: ResidualEvaluation, 
            residual: FloatArray, 
            *, 
            iteration: int) -> LinearSolveResult:
        start = time.perf_counter()
        matrix = self._require_jacobian().evaluate(committed, evaluation)
        correction = _solve_linear_correction(matrix, residual, iteration)
        return LinearSolveResult(correction, 0, time.perf_counter() - start)


class CondensedDirectSolver(_BoundSolver):
    """UMFPACK on the condensed face system; reference for condensed Krylov solvers."""

    _system: CondensedElasticSystem | None = None

    def bind(
        self,
        assembler: ResidualAssembler,
        jacobian: AnalyticalJacobian,
    ) -> None:
        super().bind(assembler, jacobian)
        self._system = jacobian.condensed_elastic()  # built once, shared

    def solve(
        self,
        committed: MaterialHistory,
        evaluation: ResidualEvaluation,
        residual: FloatArray,
        *,
        iteration: int,
    ) -> LinearSolveResult:
        start = time.perf_counter()
        jacobian = self._require_jacobian()
        assert self._system is not None  # set together with the Jacobian in bind
        correction = self._system.solve(
            -np.asarray(residual),
            jacobian.plastic_face_block(committed, evaluation),
            # _solve_linear_correction solves S x = -rhs; pass -b to get S x = b.
            lambda S, b: _solve_linear_correction(S, -b, iteration),
        )
        return LinearSolveResult(correction, 0, time.perf_counter() - start)

def _checked(result: LinearSolveResult, size: int, iteration: int) -> FloatArray:
    """Validate a solver's correction; used by solve_newton for every solver."""
    correction = np.asarray(result.correction, dtype=np.float64)
    if correction.shape != (size,):
        raise RuntimeError(f"Linear solver returned shape {correction.shape}; expected ({size},).")
    if not np.all(np.isfinite(correction)):
        raise NewtonDivergenceError(f"Non-finite correction at iteration {iteration + 1}.")
    return correction
