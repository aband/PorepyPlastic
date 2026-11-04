"""Textbook strip footing with the prototype solver; no core imports.

Run from the repository root: python -B -m examples.strip_footing_example
Outputs are labelled prototype and use m, kPa, and kN/m units.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prototype.strip_footing import make_case, run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-under-footing", type=int, default=6, help="Cells across the half-footing (0.5 m)")
    parser.add_argument("--growth", type=float, default=1.25, help="Geometric cell growth away from the footing")
    parser.add_argument("--steps", type=int, default=14, help="Uniform target increments; cutbacks may add substeps")
    parser.add_argument("--settlement", type=float, default=0.002, help="Final settlement in m (textbook: 0.002)")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--fixed-steps", action="store_true", help="Disable adaptive cutbacks")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/prototype_strip_footing"))
    args = parser.parse_args()
    case = make_case(args.cells_under_footing, growth=args.growth, settlement=args.settlement)
    print(f"Prototype strip footing: {case.grid.num_cells} cells, {case.operators.num_dofs} DOFs", flush=True)
    result = run_case(case, steps=args.steps, adaptive=not args.fixed_steps,
                      max_iterations=args.max_iterations, progress=True)
    if not args.no_export:
        from prototype.strip_footing_plots import export_results
        paths = export_results(result, args.output_dir)
        print(f"Results and numerical checks: {paths['summary']}")


if __name__ == "__main__":
    main()
