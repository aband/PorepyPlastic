"""Compare retained Jacobians at the same peak state of the Dirichlet example.

Run with python -m examples.compare_jacobians. With no mesh override, reuse
dirichlet_example.run_example's current Python default. Solve its
loading/unloading path once with the analytical Jacobian, then evaluate
both tangents at its peak candidate and PRECEDING committed history.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]

from prototype.jacobian import RetainedAnalyticalJacobian, RetainedFiniteDifferenceJacobian
from prototype.loading import LoadStepController
from prototype.operators import RetainedFaceOperators
from examples.dirichlet_example import run_example
from utils.state import TpsaState
from utils.visualization import export_jacobian_png


def _errors(reference: sps.csr_array, difference: sps.csr_array) -> dict[str, float | None]:
    """Sparse Frobenius/max norms; undefined relative norms remain explicit."""
    reference_norm = float(np.linalg.norm(reference.data))
    difference_norm = float(np.linalg.norm(difference.data))
    return {
        "analytical_frobenius": reference_norm,
        "difference_frobenius": difference_norm,
        "relative_frobenius": difference_norm / reference_norm if reference_norm else None,
        "max_abs_difference": float(np.max(np.abs(difference.data), initial=0.0)),
    }


def run_comparison(
    *, cells_per_axis: int | None = None, fd_step: float = 1e-10,
    adaptive: bool = False, output_dir: str | Path = "results/jacobian_comparison",
    verbose: bool = False,
) -> dict[str, Path]:
    """Save J_FD - J_analytical, its signed heatmap, and full/blockwise norms.

    Both matrices differentiate exactly the same candidate, history and load.
    Relative errors use Frobenius norms, not entrywise division by possible
    zeros. Global norms mix block units; the free-face block is reported
    separately. Generated files use a fixed prefix: choose separate directories
    when comparing different meshes or finite-difference perturbations.
    """
    if not np.isfinite(fd_step) or fd_step <= 0:
        raise ValueError("fd_step must be finite and positive.")
    if cells_per_axis is None:
        result = run_example(formulation="retained", jacobian="analytical",
                             adaptive=adaptive, verbose=verbose)
    else:
        result = run_example(cells_per_axis=cells_per_axis, formulation="retained",
                             jacobian="analytical", adaptive=adaptive, verbose=verbose)
    if not isinstance(result.controller, LoadStepController):
        raise TypeError("Expected the retained load controller.")
    case = result.case
    ops = RetainedFaceOperators(
        case.grid, case.bc, mu=case.material.isotropic_shear_modulus,
        lam=case.material.lame_parameter,
    )
    records = result.controller.steps
    peak_index = next(i for i, record in enumerate(records) if record.load_factor == 1.0)
    peak = records[peak_index]
    old = TpsaState.zeros(case.grid) if peak_index == 0 else records[peak_index - 1].state.tpsa
    analytical = RetainedAnalyticalJacobian(ops, case.material_points, old)(
        peak.newton.z, peak.newton.evaluation,
    )
    numerical = RetainedFiniteDifferenceJacobian(ops, case.material_points, old, step=fd_step)(
        peak.newton.z, peak.newton.evaluation,
    )
    difference = sps.csr_array(numerical - analytical)
    difference.eliminate_zeros()
    # Fixed elastic maps match exactly; only the face/face block can differ.
    if difference[:4 * ops.num_cells].nnz or difference[:, :4 * ops.num_cells].nnz:
        raise RuntimeError("Jacobian difference extends outside the free-face block.")

    names = ("u", "r", "p", "face")
    slices = (ops.u_slice, ops.r_slice, ops.p_slice, ops.face_slice)
    blocks = [
        {"residual_block": row_name, "unknown_block": col_name,
         **_errors(analytical[row, col], difference[row, col])}
        for row_name, row in zip(names, slices, strict=True)
        for col_name, col in zip(names, slices, strict=True)
    ]
    errors = _errors(analytical, difference)
    report = {
        "formulation": "retained", "trajectory_jacobian": "analytical",
        "load_control": "adaptive" if adaptive else "fixed",
        "rejected_load_attempts": sum(not attempt.accepted for attempt in result.controller.attempts),
        "difference": "finite-difference minus analytical",
        "finite_difference_step": fd_step, "num_cells": ops.num_cells,
        "num_free_faces": ops.num_free_faces, "matrix_shape": list(analytical.shape),
        "accepted_load_factors": [record.load_factor for record in records],
        "peak_step": peak_index + 1, "load_factor": peak.load_factor,
        "committed_load_factor": 0.0 if peak_index == 0 else records[peak_index - 1].load_factor,
        "plastic_cells_in_peak_increment": int(sum(
            new.alpha > previous.alpha for new, previous in
            zip(peak.state.tpsa.material_states, old.material_states, strict=True)
        )),
        "norm_convention": "Raw entries; full Frobenius norm mixes block units.",
        "full_matrix": errors, "blocks": blocks,
    }
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    prefix = "dirichlet_example_retained_finite-difference_minus_analytical"
    outputs = {}
    for name, matrix in (("analytical", analytical), ("finite-difference", numerical),
                         ("difference", difference)):
        matrix_prefix = prefix if name == "difference" else f"dirichlet_example_retained_{name}"
        outputs[name] = folder / f"{matrix_prefix}_peak.npz"
        sps.save_npz(outputs[name], matrix)
    outputs["summary"] = folder / f"{prefix}_summary.json"
    outputs["summary"].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    outputs["blocks"] = folder / f"{prefix}_blocks.csv"
    with outputs["blocks"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(blocks[0]))
        writer.writeheader()
        writer.writerows(blocks)
    relative = errors["relative_frobenius"]
    assert relative is not None  # The retained elastic matrix is nonzero.
    outputs["heatmap"] = export_jacobian_png(
        difference, formulation="retained", num_cells=ops.num_cells,
        folder_name=folder, file_name=f"{prefix}_peak",
        title=("Jacobian difference: finite-difference − analytical\n"
               f"Same peak state, factor 1; h={fd_step:g}; relative Frobenius={relative:.3e}"),
    )
    if verbose:
        print(f"Full relative Frobenius difference: {relative:.6e}")
        print(f"Max absolute entry difference: {errors['max_abs_difference']:.6e}")
        print(f"Free-face relative Frobenius difference: {blocks[-1]['relative_frobenius']}")
        for name, path in outputs.items():
            print(f"Output ({name}): {path}")
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-per-axis", type=int, default=None,
                        help="Override the Dirichlet example's current Python mesh default.")
    parser.add_argument("--fd-step", type=float, default=1e-10)
    parser.add_argument("--adaptive", action="store_true",
                        help="Use the existing adaptive Dirichlet load controller.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/jacobian_comparison"))
    args = parser.parse_args(argv)
    try:
        run_comparison(cells_per_axis=args.cells_per_axis, fd_step=args.fd_step,
                       adaptive=args.adaptive, output_dir=args.output_dir, verbose=True)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")


if __name__ == "__main__":
    main()
