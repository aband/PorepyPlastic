"""Run the paper's 3D cantilever and write PNGs/CSV/JSON under results/cantilever_3d.

    python -B -m prototype.cantilever_example --study

The default graded grid is 10 x 2 x 6. --study also runs uniform and graded
2/4/8/16 x 2 x 2 grids. --grading both-ends clusters cells at both the root
and tip (its study uses 4/8/16/32 axial cells). For example:

    python -B -m prototype.cantilever_example --nx 20 --ny 4 --nz 12 --steps 80 --grading both-ends --output-dir results/cantilever_3d/both_ends_20x4x12_80steps

Reference curves are explicitly labelled as digitized
from the provided paper, not raw Abaqus output. See prototype.cantilever.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prototype.cantilever import make_case, run_case
from prototype.cantilever_plots import export_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=10)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--grading-ratio", type=float, default=8., help="Largest/smallest axial cell width.")
    parser.add_argument("--grading", choices=("clamp", "both-ends"), default="clamp")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--jacobian", choices=("analytical", "finite-difference"), default="analytical")
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cantilever_3d"))
    args = parser.parse_args()
    result = run_case(make_case(args.nx, args.ny, args.nz, grading_ratio=args.grading_ratio, grading=args.grading),
                      steps=args.steps, jacobian=args.jacobian, progress=True)
    study = []
    if args.study:
        for ratio in sorted({1., args.grading_ratio}):
            for nx in ((4, 8, 16, 32) if args.grading == "both-ends" else (2, 4, 8, 16)):
                study.append(run_case(make_case(nx, args.ny, 2, grading_ratio=ratio, grading=args.grading),
                                      steps=args.steps, jacobian=args.jacobian, progress=True))
    paths = export_results(result, study, args.output_dir)
    print(f"Wrote {sum(p.suffix == '.png' for p in paths.values())} PNG plots to {args.output_dir}")
    print(f"Case settings, deviations and numerical checks: {paths['summary']}")


if __name__ == "__main__":
    main()
