"""Fixed and adaptive load scheduling, independent of the discrete formulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from utils.newton_utils import NewtonConvergenceError


class LoadStepError(RuntimeError):
    """A target failed; inspect the controller for its last accepted state/steps.

    step_index is one-based, counting from the controller's first accepted step.
    The original material, Jacobian, or Newton exception is retained as __cause__.
    """

    def __init__(self, step_index: int, load_factor: float, error: Exception) -> None:
        self.step_index = step_index
        self.load_factor = load_factor
        super().__init__(f"Load step {step_index} (factor={load_factor:g}) failed: {error}")


@dataclass(frozen=True)
class AdaptiveLoadOptions:
    """Positive load-factor increment magnitudes, independent of loading direction.

    Halve failed increments by default; grow easy successful steps by 1.5.
    max_retries counts retries after the first failure at one committed state.
    max_steps bounds accepted substeps in one adaptive invocation. A final
    remainder smaller than min_increment is allowed to reach a target exactly.
    """

    initial_increment: float = 0.05
    min_increment: float = 1e-4
    max_increment: float = 0.2
    cutback_factor: float = 0.5
    growth_factor: float = 1.5
    target_iterations: int = 4
    max_retries: int = 8
    max_steps: int = 1000

    def __post_init__(self) -> None:
        for name in ("initial_increment", "min_increment", "max_increment"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not self.min_increment <= self.initial_increment <= self.max_increment:
            raise ValueError("Require min_increment <= initial_increment <= max_increment.")
        if not np.isfinite(self.cutback_factor) or not 0 < self.cutback_factor < 1:
            raise ValueError("cutback_factor must lie strictly between zero and one.")
        if not np.isfinite(self.growth_factor) or self.growth_factor < 1:
            raise ValueError("growth_factor must be finite and at least one.")
        for name in ("target_iterations", "max_retries", "max_steps"):
            value = getattr(self, name)
            lower = 1 if name == "max_steps" else 0
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}.")


@dataclass(frozen=True)
class LoadAttempt:
    """Lightweight immutable diagnostics; rejected trials contain no accepted state."""

    start_factor: float
    load_factor: float
    iterations: int | None
    error: str | None = None

    @property
    def accepted(self) -> bool:
        return self.iterations is not None


class AdaptiveLoadStepError(LoadStepError):
    """Adaptive bounds exhausted; previous accepted substeps remain available."""

    def __init__(self, step_index: int, load_factor: float, target: float, reason: str) -> None:
        self.target = target
        super().__init__(step_index, load_factor, RuntimeError(
            f"Adaptive stepping toward factor={target:g} stopped: {reason}",
        ))


class _NewtonIterations(Protocol):
    @property
    def iterations(self) -> int: ...


class _StepRecord(Protocol):
    @property
    def newton(self) -> _NewtonIterations: ...


_Step = TypeVar("_Step", bound=_StepRecord)


class _LoadStepSchedule(ABC, Generic[_Step]):
    """Shared fixed/adaptive scheduling; subclasses own solve and acceptance."""

    _load_factor: float
    _steps: list[_Step]

    @abstractmethod
    def advance(self, load_factor: float) -> _Step:
        """Solve and atomically accept one target, or raise LoadStepError."""

    def run(self, load_factors: Sequence[float] | NDArray[np.float64]) -> tuple[_Step, ...]:
        """Advance through finite factors in the given order; stop on the first failure.

        Validate the complete schedule before taking any step. Empty input is a
        no-op. On failure, earlier accepted steps remain available through steps
        and state; callers may explicitly try another target with advance/run.
        The returned tuple contains only steps accepted by this run invocation.
        """
        factors = np.array(load_factors, dtype=np.float64, copy=True)
        if factors.ndim != 1 or not np.all(np.isfinite(factors)):
            raise ValueError("load_factors must be a finite one-dimensional sequence.")
        return tuple(self.advance(float(factor)) for factor in factors)

    def iter_adaptive(
        self,
        load_factors: Sequence[float] | NDArray[np.float64],
        *,
        options: AdaptiveLoadOptions | None = None,
    ) -> Generator[_Step, None, None]:
        """Yield accepted substeps through ordered absolute target factors.

        Validate the complete path before the first solve. Reach every target
        exactly without overshoot; repeated targets are skipped. Increment
        magnitudes carry across targets, resetting to initial_increment when
        direction reverses. Only NewtonConvergenceError triggers cutback/retry.
        Other errors propagate immediately through advance's LoadStepError.

        Grow when iterations <= target_iterations, except immediately after a
        cutback; otherwise keep the increment. A clipped endpoint does not shrink
        the proposed increment. Retry bounds reset after each accepted substep.
        Failed states are never accepted. Use this iterator to check/report each
        accepted state before proceeding; do not advance this controller outside
        the iterator while it is active. Closing it leaves the last accepted state.
        """
        settings = AdaptiveLoadOptions() if options is None else options
        factors = np.array(load_factors, dtype=np.float64, copy=True)
        if factors.ndim != 1 or not np.all(np.isfinite(factors)):
            raise ValueError("load_factors must be a finite one-dimensional sequence.")
        with np.errstate(over="ignore", invalid="ignore"):
            distances = np.diff(np.r_[self._load_factor, factors])
        if not np.all(np.isfinite(distances)):
            raise ValueError("Differences between load factors must be finite.")
        increment = settings.initial_increment
        direction = 0.0
        accepted = 0
        for factor in factors:
            target = float(factor)
            if target == self._load_factor:
                continue
            next_direction = float(np.sign(target - self._load_factor))
            if direction and next_direction != direction:
                increment = settings.initial_increment
            direction = next_direction
            retries = 0
            last_failed: float | None = None
            while self._load_factor != target:
                start = self._load_factor
                remaining = abs(target - start)
                candidate = target if remaining <= increment else start + direction * increment
                candidate = min(candidate, target) if direction > 0 else max(candidate, target)
                if accepted >= settings.max_steps:
                    raise AdaptiveLoadStepError(len(self._steps) + 1, candidate, target,
                                                "accepted-step limit reached")
                if candidate == start or candidate == last_failed:
                    raise AdaptiveLoadStepError(len(self._steps) + 1, candidate, target,
                                                "increment cannot make further floating-point progress")
                try:
                    record = self.advance(candidate)
                except LoadStepError as error:
                    if not isinstance(error.__cause__, NewtonConvergenceError):
                        raise
                    attempted_increment = abs(candidate - start)
                    if attempted_increment <= settings.min_increment:
                        raise AdaptiveLoadStepError(error.step_index, candidate, target,
                                                    "minimum increment reached") from error
                    if retries >= settings.max_retries:
                        raise AdaptiveLoadStepError(error.step_index, candidate, target,
                                                    "retry limit reached") from error
                    increment = max(settings.min_increment, attempted_increment * settings.cutback_factor)
                    retries += 1
                    last_failed = candidate
                    continue
                if retries == 0 and record.newton.iterations <= settings.target_iterations:
                    increment = min(settings.max_increment, increment * settings.growth_factor)
                accepted += 1
                retries = 0
                last_failed = None
                yield record

    def run_adaptive(
        self,
        load_factors: Sequence[float] | NDArray[np.float64],
        *,
        options: AdaptiveLoadOptions | None = None,
    ) -> tuple[_Step, ...]:
        """Collect iter_adaptive's accepted substeps; preserve progress on failure."""
        return tuple(self.iter_adaptive(load_factors, options=options))
