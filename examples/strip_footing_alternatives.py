"""Run the same strip footing with alternatives/coupled; no core imports.

python -B -m examples.strip_footing_alternatives
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alternatives.strip_footing import StripFootingCase, run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-under-footing", type=int, default=6)
    parser.add_argument("--growth", type=float, default=1.25)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--settlement", type=float, default=0.002)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--fixed-steps", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/alternatives_strip_footing"))
    args = parser.parse_args()
    case = StripFootingCase(args.cells_under_footing, growth=args.growth, settlement=args.settlement)
    print(f"Alternatives coupled strip footing: {case.grid.num_cells} cells, "
          f"{case.operators.num_dofs} DOFs", flush=True)
    result = run_case(case, steps=args.steps, adaptive=not args.fixed_steps,
                      max_iterations=args.max_iterations, progress=True)
    if not args.no_export:
        from alternatives.strip_footing_plots import export_results
        paths = export_results(result, args.output_dir)
        print(f"Results and numerical checks: {paths['summary']}")


if __name__ == "__main__":
    main()
