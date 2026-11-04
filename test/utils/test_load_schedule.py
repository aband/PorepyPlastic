"""Adaptive cutback routing by failure type, independent of any formulation."""

from dataclasses import dataclass

import pytest

from material.J2 import ReturnMapConvergenceError
from utils.load_schedule import AdaptiveLoadOptions, LoadStepError, _LoadStepSchedule
from utils.newton_utils import LocalUpdateError, NewtonConvergenceError, NewtonDivergenceError


@dataclass
class _Newton:
    iterations: int


@dataclass
class _Record:
    load_factor: float
    newton: _Newton


class _ThresholdSchedule(_LoadStepSchedule[_Record]):
    """Reject increments above 0.1 with a fixed cause; accept smaller ones."""

    def __init__(self, cause: Exception) -> None:
        self._load_factor = 0.0
        self._steps: list[_Record] = []
        self.cause = cause
        self.attempts: list[float] = []

    def advance(self, load_factor: float) -> _Record:
        self.attempts.append(load_factor)
        if abs(load_factor - self._load_factor) > 0.1 + 1e-12:
            raise LoadStepError(len(self._steps) + 1, load_factor, self.cause) from self.cause
        record = _Record(load_factor, _Newton(1))
        self._steps.append(record)
        self._load_factor = load_factor
        return record


OPTIONS = AdaptiveLoadOptions(initial_increment=0.2, max_increment=0.2, growth_factor=1.0)


@pytest.mark.parametrize("cause", [
    NewtonConvergenceError("iterations"),
    NewtonDivergenceError("non-finite candidate"),
    LocalUpdateError("cell 3"),
    ReturnMapConvergenceError("finite-difference tangent"),
], ids=lambda error: type(error).__name__)
def test_recoverable_failures_cut_back_to_target(cause: Exception) -> None:
    schedule = _ThresholdSchedule(cause)
    steps = schedule.run_adaptive([1.0], options=OPTIONS)
    assert schedule._load_factor == 1.0
    assert all(abs(b.load_factor - a.load_factor) <= 0.1 + 1e-12
               for a, b in zip((_Record(0.0, _Newton(0)), *steps), steps))
    assert len(schedule.attempts) > len(steps)


@pytest.mark.parametrize("cause", [
    ValueError("invalid input"),
    RuntimeError("unclassified failure"),
], ids=lambda error: type(error).__name__)
def test_other_failures_abort_without_retry(cause: Exception) -> None:
    schedule = _ThresholdSchedule(cause)
    with pytest.raises(LoadStepError) as caught:
        schedule.run_adaptive([1.0], options=OPTIONS)
    assert caught.value.__cause__ is cause
    assert schedule.attempts == [0.2] and schedule._load_factor == 0.0
