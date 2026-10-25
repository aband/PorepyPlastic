"""Selection defaults to the retained equations and rejects unknown formulations."""

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pytest

from examples import plastic_plane_strain as homogeneous
from utils.load_schedule import AdaptiveLoadOptions
from prototype.loading import LoadStepController
from utils.problem import PlaneStrainProblem
from alternatives.plane_strain import PlaneStrainTpsa
from examples import dirichlet_example as nonuniform

Runner: TypeAlias = Callable[..., homogeneous.PlasticPlaneStrainResult | nonuniform.DirichletResult]


@pytest.mark.parametrize("runner", [homogeneous.run_example, nonuniform.run_example],
                         ids=["homogeneous", "nonuniform"])
@pytest.mark.parametrize("jacobian", ["analytical", "finite-difference"])
@pytest.mark.parametrize("adaptive", [False, True], ids=["fixed", "adaptive"])
def test_explicit_retained_matches_default(runner: Runner, jacobian: str, adaptive: bool) -> None:
    controls = (adaptive if runner is nonuniform.run_example
                else AdaptiveLoadOptions() if adaptive else None)
    default = runner(cells_per_axis=2, jacobian=jacobian, adaptive=controls)
    explicit = runner(cells_per_axis=2, jacobian=jacobian, adaptive=controls, formulation="retained")
    assert default.formulation == explicit.formulation == "retained"
    assert default.outputs == explicit.outputs == {}
    assert isinstance(default.controller, LoadStepController)
    assert isinstance(explicit.controller, LoadStepController)
    a_steps, b_steps = default.controller.steps, explicit.controller.steps
    assert len(a_steps) == len(b_steps)
    assert default.controller.attempts == explicit.controller.attempts
    for a, b in zip(a_steps, b_steps, strict=True):
        assert a.load_factor == b.load_factor
        assert a.newton.iterations == b.newton.iterations
        np.testing.assert_array_equal(a.newton.residual_norms, b.newton.residual_norms)
        np.testing.assert_array_equal(a.state.side_traction, b.state.side_traction)
        for name in ("x", "epsilon", "traction", "bc_values", "face_displacement"):
            np.testing.assert_array_equal(getattr(a.state.tpsa, name), getattr(b.state.tpsa, name))
        for a_material, b_material in zip(a.state.tpsa.material_states, b.state.tpsa.material_states, strict=True):
            assert a_material.alpha == b_material.alpha
            for name in ("stress", "plastic_strain", "backstress"):
                np.testing.assert_array_equal(getattr(a_material, name).to_numpy(),
                                               getattr(b_material, name).to_numpy())


@pytest.fixture
def forbid_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(case: PlaneStrainProblem) -> None:
        pytest.fail("Unsupported formulation reached simulation preparation.")

    monkeypatch.setattr(PlaneStrainProblem, "prepare_simulation", unexpected)
    monkeypatch.setattr(PlaneStrainTpsa, "prepare_simulation", unexpected)


@pytest.mark.parametrize("runner", [homogeneous.run_example, nonuniform.run_example],
                         ids=["homogeneous", "nonuniform"])
@pytest.mark.parametrize("jacobian", ["analytical", "finite-difference"])
@pytest.mark.parametrize("formulation,error,message", [
    ("misspelled", ValueError, "formulation must be"),
])
def test_invalid_selection_never_prepares_or_exports(
    runner: Runner, jacobian: str, formulation: str, error: type[Exception], message: str,
    forbid_preparation: None, tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    with pytest.raises(error, match=message):
        runner(formulation=formulation, jacobian=jacobian, output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("main", [homogeneous.main, nonuniform.main], ids=["homogeneous", "nonuniform"])
@pytest.mark.parametrize("formulation,code,message", [
    ("misspelled", 2, "invalid choice"),
])
def test_cli_reports_unavailable_or_invalid_formulation(
    main: Callable[[list[str]], None], formulation: str, code: int, message: str,
    forbid_preparation: None, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as caught:
        main(["--formulation", formulation, "--output-dir", str(output)])
    assert caught.value.code == code
    text = capsys.readouterr()
    assert message in text.err and "Traceback" not in text.err
    assert "Converged" not in text.out and "PASS" not in text.out
    assert not output.exists()
