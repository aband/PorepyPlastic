"""Elastic TPSA baseline: python -m examples.elastic_plane_strain."""

import argparse
from pathlib import Path

import numpy as np
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.numerics.fv.tpsa import Tpsa

from alternatives.plane_strain import KEYWORD, PlaneStrainTpsa
from alternatives.postprocessing import TpsaPostprocessing
from utils.visualization import export_strain_png, export_vtk


def main() -> None:
    """Solve, compare with the affine reference, and export the numerical fields."""
    parser = argparse.ArgumentParser(
        description="Solve the elastic TPSA plane-strain example and export VTK and strain PNG files."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="VTK and strain PNG output directory (default: results).",
    )
    parser.add_argument(
        "--show-l2-errors", action="store_true",
        help="Print the L2 error report (hidden by default).",
    )
    args = parser.parse_args()
    case = PlaneStrainTpsa()
    case.solve()
    post = TpsaPostprocessing(case)
    g, x = post.grid, post.x
    u, r, p = post.u_cell, post.r, post.p
    epsilon = post.strain_green_gauss()
    epsilon_ex = np.repeat(
        case.reference_strain().to_numpy()[:, :, None], g.num_cells, axis=2,
    )

    u_ex = case.reference_displacement(g.cell_centers)
    r_ex = np.full(g.num_cells, case.reference_rotation_stress())
    p_ex = np.full(g.num_cells, case.reference_total_pressure())
    generalized_flux = (
        post.face_discretization @ x + post.rhs_matrix @ post.bound_vec
    )
    # TPSA's stress block contains face-integrated tractions; normals already
    # include face measures. Forces are per unit out-of-plane thickness in 2D.
    stress = generalized_flux[:g.dim * g.num_faces].reshape((g.dim, -1), order="F")
    stress_ex = case.reference_stress()[:2, :2] @ g.face_normals[:2]
    # Absolute discrete L2 norms use cell volumes for cell fields and PorePy's
    # face dual volumes for integrated forces. In 2D the square root of these
    # measures contributes one metre to the norm's units. Vector components
    # must be interleaved by cell/face for the integration weights to align.
    comparisons = {
        "displacement [m^2]": (u_ex.ravel(order="F"), u.ravel(order="F"), True),
        "total pressure [Pa m]": (p_ex, p, True),
        "rotation stress [Pa m]": (r_ex, r, True),
        "Green-Gauss strain [m]": (
            epsilon_ex.ravel(order="F"), epsilon.ravel(order="F"), True,
        ),
        "face force per thickness [N]": (
            stress_ex.ravel(order="F"), stress.ravel(order="F"), False,
        ),
    }
    print(
        f"Cartesian plane strain: {g.num_cells} cells, "
        f"{Tpsa(KEYWORD).ndof(g)} global unknowns"
    )
    print(f"Material points: {len(case.material_points)} (initialized 3D histories)")
    if args.show_l2_errors:
        print("Absolute discrete L2 errors against the affine elastic solution (norm units):")
        for name, (exact, numerical, is_cc) in comparisons.items():
            error = ConvergenceAnalysis.lp_error(
                grid=g, true_array=exact, approx_array=numerical,
                is_cc=is_cc, p=2, relative=False,
            )
            print(f"  {name}: {error:.3e}")
    print(f"Reference sigma_zz: {case.reference_stress()[2, 2]:.6e} Pa")
    pvd = export_vtk(g, x, epsilon=epsilon, folder_name=args.output_dir)
    pngs = export_strain_png(g, epsilon, folder_name=args.output_dir)
    print(f"VTK output: {pvd} (open in ParaView)")
    for name, path in pngs.items():
        print(f"Strain PNG ({name}): {path}")


if __name__ == "__main__":
    main()
