"""Run elastic bending and reconstructed-stress convergence in the prototype.

    python -B -m prototype.elastic_bending_example
    python -B -m prototype.elastic_bending_example --quick
    python -B -m prototype.elastic_bending_example --only pure --pure-levels 0 1 2 3
    python -B -m prototype.elastic_bending_example --only cantilever --axial-cells 256 512 1024 2048 --ny 4 --nz 8

The default study includes exact 3D bending, axial cantilever refinement, and
a separate cross-section sensitivity check. All files carry a prototype prefix.
No plastic return map or core solver is used. Completed levels are checkpointed.

Verification problems (N, mm, MPa):
  * Exact bending: a 4 x 2 x 1 block, E=210000, nu=0.3, k=1e-4/mm.
    u=k*(-xz, nu*yz, (x^2+nu*(z^2-y^2))/2), sigma_xx=-E*k*z.
    Prescribe exact face-average displacement at x=0 and exact integrated
    traction elsewhere. This includes Poisson warping at x=0, not a rigid clamp.
    Refine all three axes together and report volume-weighted relative L2
    errors and observed orders against the exact solution.
  * Paper cantilever: the existing 100 x 10 x 1 geometry, rigid left clamp,
    tip traction 0.06 in +z, all other surfaces free. E and nu as above.
    Compare mean tip displacement with the digitized paper elastic point and
    PL^3/(3EI); compare interior axial stress with -P(L-x)z/I and section
    moments with P(L-x). Beam stress is approximate, particularly near ends.

Stress is reconstructed from the prototype Green-Gauss face-displacement
strain using sigma=2*mu*epsilon+lambda*tr(epsilon)*I. It is different from the
numerical face traction used for equilibrium. Both moment measures are saved;
the cell stress moment also has a midpoint-sampled analytical reference to
separate quadrature error (1/nz^2 for linear sigma_xx) from stress error.

PNG plots, convergence CSV/JSON and finest strain/stress NPZ fields are written
under results/prototype_elastic_bending. Axial and cross-section studies have
separate field exports. Axial refinement alone cannot establish full 3D
convergence, and these elastic checks do not validate the plastic load cycle.
The default cross-section check uses 2 x 4, 4 x 8 and 8 x 16 cells at nx=512.
The largest case uses an equivalent banded direct solve and can require over
10 GB of memory and several minutes. Use --quick for an execution check, or
--no-cross-section-study to omit the separate cross-section series.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from prototype.elastic_bending import make_elastic_cantilever, make_pure_bending_case, solve_elastic
from prototype.elastic_bending_plots import Record, export_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("both", "pure", "cantilever"), default="both")
    parser.add_argument("--pure-levels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--axial-cells", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--traction", type=float, default=.06)
    parser.add_argument("--cross-section-study", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quick", action="store_true", help="Small smoke run; not a converged benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/prototype_elastic_bending"))
    args = parser.parse_args()
    if any(n < 0 for n in args.pure_levels):
        parser.error("pure levels must be nonnegative")
    if args.quick:
        args.pure_levels, args.axial_cells = [0, 1], [16, 32]
        args.ny, args.nz, args.cross_section_study = 2, 4, False
    jobs = []
    if args.only in ("both", "pure"):
        jobs.extend(("exact_3d", "pure", (4*2**n, 2*2**n, 2*2**n)) for n in sorted(set(args.pure_levels)))
    if args.only in ("both", "cantilever"):
        jobs.extend((f"axial_{args.ny}x{args.nz}", "cantilever", (nx, args.ny, args.nz)) for nx in sorted(set(args.axial_cells)))
        if args.cross_section_study:
            jobs.extend(("cross_section_at_512", "cantilever", (512, ny, nz)) for ny, nz in ((2, 4), (4, 8), (8, 16)))
    records: list[Record] = []
    completed = {}
    for series, kind, shape in jobs:
        start = time.monotonic()
        key = (kind, shape)
        if key not in completed:
            case = make_pure_bending_case(shape) if kind == "pure" else make_elastic_cantilever(shape, traction=args.traction)
            try:
                completed[key] = solve_elastic(case)
            except Exception:
                # Preserve the usable study, including figures, if a later
                # factorization fails (for example on a memory-limited host).
                if records:
                    export_results(records, args.output_dir)
                raise
        result = completed[key]
        records.append((series, result))
        m = result.metrics
        print(f"{series}: {shape}, tip={m['tip_mm']:.8g} mm, "
              f"stress L2={m['stress_relative_l2']:.3%}, "
              f"interior xx={m['interior_sigma_xx_relative_l2']:.3%}, "
              f"moment={m['interior_face_moment_relative_l2']:.3%}, "
              f"{time.monotonic()-start:.1f} s", flush=True)
        export_results(records, args.output_dir, plots=False)
    paths = export_results(records, args.output_dir)
    print(f"Results: {paths['summary']}", flush=True)
    if args.quick:
        print("Quick run verifies execution and output only; use the default study for refinement.")


if __name__ == "__main__":
    main()
