"""Linear-correction backend selection, numerical agreement and failure handling."""

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import warnings

import numpy as np
import pytest
import scipy.sparse as sps

from utils import newton_utils
from utils.newton_utils import NewtonDivergenceError, _solve_linear_correction


@pytest.fixture(params=["superlu", "umfpack"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Exercise the fallback even when the optional UMFPACK package is installed."""
    umfpack = newton_utils.umfpack
    forbidden = Mock(side_effect=AssertionError("The unselected backend was called"))
    if request.param == "superlu":
        if umfpack is not None:
            monkeypatch.setattr(umfpack, "splu", forbidden)
        monkeypatch.setattr(newton_utils, "umfpack", None)
        return sps.linalg
    if umfpack is None:
        pytest.skip("Optional scikits.umfpack is not installed")
    monkeypatch.setattr(sps.linalg, "splu", forbidden)
    return umfpack


@pytest.mark.parametrize("storage", ["dense", "csr_array", "csc_matrix"])
def test_correction_solves_system_once_without_mutating_inputs(
    backend: ModuleType, storage: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unsorted indices and duplicates represent [[4, 1, 0], [2, 3, 1], [0, 1, 2]].
    sparse = sps.csc_matrix((
        np.array([2., 2., 2., 1., 2., 1., 1., 2., 1.]),
        np.array([1, 0, 0, 2, 1, 0, 1, 2, 1]),
        np.array([0, 3, 7, 9]),
    ), shape=(3, 3))
    matrix = (sparse.toarray() if storage == "dense"
              else sps.csr_array(sparse) if storage == "csr_array" else sparse)
    expected = np.array([1.5, -.5, 2.])
    residual = np.asarray(-matrix @ expected)
    saved_residual = residual.copy()
    arrays = [matrix] if storage == "dense" else [matrix.data, matrix.indices, matrix.indptr]
    saved_arrays = [array.copy() for array in arrays]
    for array in [*arrays, residual]:
        array.setflags(write=False)

    original_factorize = backend.splu
    factors: list[Mock] = []

    def factorize(value: sps.csc_matrix) -> Mock:
        factor = Mock(wraps=original_factorize(value))
        factors.append(factor)
        return factor

    spy = Mock(side_effect=factorize)
    monkeypatch.setattr(backend, "splu", spy)
    correction = _solve_linear_correction(matrix, residual, iteration=0)

    np.testing.assert_allclose(correction, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(matrix @ correction + residual, 0., atol=1e-13)
    assert correction.shape == residual.shape and correction.dtype == np.float64
    spy.assert_called_once()
    factors[0].solve.assert_called_once()
    np.testing.assert_array_equal(factors[0].solve.call_args.args[0], -residual)
    np.testing.assert_array_equal(residual, saved_residual)
    for array, saved in zip(arrays, saved_arrays, strict=True):
        np.testing.assert_array_equal(array, saved)


def test_singular_system_is_a_recoverable_failure(backend: ModuleType) -> None:
    with pytest.raises(NewtonDivergenceError, match="linear solve failed at correction 3") as caught:
        _solve_linear_correction(sps.csr_array((2, 2)), np.ones(2), iteration=2)
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("failure", ["factorization", "solve", "nan", "inf"])
def test_backend_failure_or_nonfinite_correction_is_rejected(
    backend: ModuleType, failure: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("Injected backend failure")
    factor = Mock()
    factorize = Mock(return_value=factor)
    if failure == "factorization":
        factorize.side_effect = error
    elif failure == "solve":
        factor.solve.side_effect = error
    else:
        factor.solve.return_value = np.array([np.nan if failure == "nan" else np.inf, 0.])
    monkeypatch.setattr(backend, "splu", factorize)
    residual = np.array([1., -2.])
    message = "linear solve failed" if failure in ("factorization", "solve") else "non-finite correction"

    with pytest.raises(NewtonDivergenceError, match=message) as caught:
        _solve_linear_correction(np.eye(2), residual, iteration=1)
    if failure in ("factorization", "solve"):
        assert caught.value.__cause__ is error
    np.testing.assert_array_equal(residual, [1., -2.])
    factorize.assert_called_once()
    assert factor.solve.call_count == (0 if failure == "factorization" else 1)


def test_umfpack_singularity_warning_is_an_error_and_filters_are_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A stub keeps warning-to-exception coverage available without UMFPACK installed.
    class UmfpackWarning(UserWarning):
        pass

    warning = UmfpackWarning("Singular matrix")

    def warn(matrix: sps.csc_matrix) -> None:
        warnings.warn(warning, stacklevel=2)
        pytest.fail("Factorization continued after a singularity warning")

    factorize = Mock(side_effect=warn)
    monkeypatch.setattr(newton_utils, "umfpack", SimpleNamespace(
        splu=factorize, UmfpackWarning=UmfpackWarning,
    ))
    monkeypatch.setattr(sps.linalg, "splu", Mock(side_effect=AssertionError("Unexpected fallback")))
    filters = warnings.filters.copy()
    with pytest.raises(NewtonDivergenceError, match="linear solve failed at correction 1") as caught:
        _solve_linear_correction(np.eye(2), np.ones(2), iteration=0)
    assert caught.value.__cause__ is warning
    assert warnings.filters == filters
    factorize.assert_called_once()


@pytest.mark.parametrize("defect", ["shape", "nonfinite"])
def test_invalid_jacobian_is_rejected_before_factorization(
    defect: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.eye(3) if defect == "shape" else np.diag([1., np.nan])
    factorize = Mock(side_effect=AssertionError("Invalid Jacobian reached factorization"))
    monkeypatch.setattr(newton_utils, "umfpack", None)
    monkeypatch.setattr(sps.linalg, "splu", factorize)
    with pytest.raises(ValueError, match="Jacobian must be finite with shape"):
        _solve_linear_correction(matrix, np.ones(2), iteration=0)
    factorize.assert_not_called()
