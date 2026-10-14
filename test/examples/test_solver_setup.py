"""Default examples use core without loading the optional formulations."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["plastic_plane_strain", "dirichlet_example"])
@pytest.mark.parametrize("jacobian", ["analytical", "finite-difference"])
def test_default_run_and_export_do_not_import_alternatives(
    module: str, jacobian: str, tmp_path: Path,
) -> None:
    # A fresh interpreter catches imports hidden by the rest of the test suite.
    script = """
import importlib
from pathlib import Path
import sys

runner = importlib.import_module('examples.' + sys.argv[1])
result = runner.run_example(cells_per_axis=2, jacobian=sys.argv[2], output_dir=Path(sys.argv[3]))
assert result.formulation == 'retained'
assert result.controller.__class__.__module__ == 'core.loading'
assert result.case.__class__.__module__ == 'utils.problem'
assert result.controller.load_factor == 0.98
assert result.outputs and all(path.is_file() for path in result.outputs.values())
assert all('_retained_' in path.name for path in result.outputs.values())
assert not hasattr(result.case, 'matrices')
assert not any(name.split('.')[0] == 'alternatives' for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, module, jacobian, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
        timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module", [
    "examples.plastic_plane_strain",
    "examples.dirichlet_example",
])
def test_default_cli_runs_retained_without_export(module: str, tmp_path: Path) -> None:
    output = tmp_path / "unused"
    result = subprocess.run(
        [sys.executable, "-m", module, "--cells-per-axis", "2", "--no-export",
         "--output-dir", str(output)], cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Formulation: retained" in result.stdout
    assert "ratio_f" in result.stdout
    assert not output.exists()
