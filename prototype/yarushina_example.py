"""Run the prototype circular-hole benchmark with a staircase boundary.

    python -B -m prototype.yarushina_example
    python -B -m prototype.yarushina_example --levels 0 1 --steps 40

All lengths/stresses use R=k=1. Default load path reaches P/k=2, tau/k=-0.3.
The finite-nu J2 approximation and exact outer-displacement truncation are
recorded in the output JSON. Source equations are documented in yarushina.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prototype.yarushina import make_case, run_case
from prototype.yarushina_plots import export_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--nu", type=float, default=.4999)
    parser.add_argument("--half-width", type=float, default=3.)
    parser.add_argument("--final-pressure", type=float, default=2.)
    parser.add_argument("--output-dir", type=Path, default=Path("results/prototype_yarushina"))
    args = parser.parse_args()
    results = []
    for level in sorted(set(args.levels)):
        case = make_case(level, half_width=args.half_width, nu=args.nu)
        results.append(run_case(case, steps=args.steps, final_pressure=args.final_pressure, progress=True))
        # Preserve completed meshes even if a subsequent finer solve fails.
        paths = export_results(results, args.output_dir)
        print(f"Prototype analytical comparison: {paths['summary']}", flush=True)


if __name__ == "__main__":
    main()
