"""Alternative solver isolation and the canonical elastic example command."""

from pathlib import Path
import subprocess
import sys


def test_alternatives_import_without_prototype() -> None:
    script = """
import importlib
import sys

import alternatives
import alternatives.original
import alternatives.coupled
assert 'alternatives.newton' not in sys.modules
assert 'alternatives.original.residual' not in sys.modules
assert 'alternatives.coupled.residual' not in sys.modules
for module in (
    'alternatives.original.residual', 'alternatives.original.jacobian',
    'alternatives.coupled.residual', 'alternatives.coupled.jacobian',
    'alternatives.newton', 'alternatives.loading',
    'alternatives.plane_strain', 'alternatives.postprocessing',
):
    importlib.import_module(module)
assert not any(name.split('.')[0] == 'prototype' for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_elastic_example_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.elastic_plane_strain", "--help"], cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--show-l2-errors" in result.stdout
    assert "--output-dir" in result.stdout
