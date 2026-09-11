"""Animate the Newton stress updates in a local J2 radial return.

Place this file in the PorepyPlastic repository root and run

    python visualize_newton_stress.py

The script leaves the constitutive implementation unchanged. It records the
same scalar Newton equation used by ``vonMisesModel.consistency_parameter``,
reconstructs the corresponding stress point at every iterate, verifies the
final state, and writes an MP4 animation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle
import numpy as np

from J2 import vonMisesModel
from scalar_hardening import (
    HardeningParameters,
    ScalarHardeningLaw,
    isotropic_hardening_K,
    kinematic_hardening_H,
)


@dataclass(frozen=True)
class NewtonState:
    iteration: int
    dgamma: float
    residual: float
    alpha: float
    isotropic_strength: float
    stress: np.ndarray
    backstress: np.ndarray
    shifted_stress: np.ndarray
    yield_radius: float


def deviatoric(value: np.ndarray) -> np.ndarray:
    """Return the three-dimensional deviatoric part."""
    return value - np.trace(value) * np.eye(3) / 3.0


def frobenius_norm(value: np.ndarray) -> float:
    return float(np.sqrt(np.einsum("ij,ij->", value, value)))


def compute_newton_history(
    sigma_trial: np.ndarray,
    backstress_n: np.ndarray,
    alpha_n: float,
    young_modulus: float,
    poisson_ratio: float,
    parameters: HardeningParameters,
    K_law: ScalarHardeningLaw = isotropic_hardening_K,
    H_law: ScalarHardeningLaw = kinematic_hardening_H,
    tolerance: float = 1.0e-10,
    max_iterations: int = 20,
) -> tuple[list[NewtonState], np.ndarray, float]:
    """Record the stress point associated with every Newton iterate."""
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    c = np.sqrt(2.0 / 3.0)

    s_trial = deviatoric(sigma_trial)
    beta_n = deviatoric(backstress_n)
    xi_trial = s_trial - beta_n
    xi_trial_norm = frobenius_norm(xi_trial)
    direction = xi_trial / xi_trial_norm

    K_n, _ = K_law(alpha_n, parameters)
    H_n, _ = H_law(alpha_n, parameters)
    trial_residual = xi_trial_norm - c * K_n

    dgamma = 0.0
    history: list[NewtonState] = []

    for iteration in range(max_iterations + 1):
        alpha = alpha_n + c * dgamma
        K, K_prime = K_law(alpha, parameters)
        H, H_prime = H_law(alpha, parameters)

        residual = (
            xi_trial_norm
            - 2.0 * mu * dgamma
            - c * (K + H - H_n)
        )

        sigma = sigma_trial - 2.0 * mu * dgamma * direction
        backstress = backstress_n + c * (H - H_n) * direction
        shifted_stress = deviatoric(sigma) - deviatoric(backstress)

        history.append(
            NewtonState(
                iteration=iteration,
                dgamma=dgamma,
                residual=float(residual),
                alpha=float(alpha),
                isotropic_strength=float(K),
                stress=sigma,
                backstress=backstress,
                shifted_stress=shifted_stress,
                yield_radius=float(c * K),
            )
        )

        if trial_residual <= 0.0 or abs(residual) <= tolerance:
            break

        derivative = (
            -2.0 * mu
            - (2.0 / 3.0) * (K_prime + H_prime)
        )
        dgamma = max(0.0, dgamma - residual / derivative)
    else:
        raise RuntimeError("Local Newton iteration did not converge.")

    model_dgamma = vonMisesModel("J2").consistency_parameter(
        xi_trial_norm=xi_trial_norm,
        alpha_n=alpha_n,
        mu=mu,
        parameters=parameters,
        K_law=K_law,
        H_law=H_law,
        tolerance=tolerance,
        max_iteration=max_iterations,
    )
    if not np.isclose(history[-1].dgamma, model_dgamma, atol=tolerance, rtol=0.0):
        raise AssertionError("Recorded Newton solution differs from the J2 model.")

    return history, direction, mu


def verify_history(
    history: list[NewtonState],
    sigma_trial: np.ndarray,
    direction: np.ndarray,
    mu: float,
    tolerance: float,
) -> None:
    """Numerically verify the animated stress updates."""
    final = history[-1]
    scale = max(final.yield_radius, 1.0)

    if abs(final.residual) > tolerance:
        raise AssertionError("The consistency residual did not converge.")
    if abs(frobenius_norm(final.shifted_stress) - final.yield_radius) > tolerance:
        raise AssertionError("The final stress point is not on the yield surface.")

    for state in history:
        expected = sigma_trial - 2.0 * mu * state.dgamma * direction
        if not np.allclose(state.stress, expected, atol=tolerance * scale, rtol=0.0):
            raise AssertionError("A Newton stress update is inconsistent with Δγ.")
        if not np.isclose(
            np.trace(state.stress),
            np.trace(sigma_trial),
            atol=tolerance * scale,
            rtol=0.0,
        ):
            raise AssertionError("The radial correction changed hydrostatic stress.")


def projection_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct an orthonormal two-dimensional deviatoric slice."""
    radial = direction / frobenius_norm(direction)
    transverse = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    transverse = transverse - np.einsum("ij,ij->", transverse, radial) * radial

    if frobenius_norm(transverse) < 1.0e-12:
        transverse = np.zeros((3, 3))
        transverse[0, 1] = transverse[1, 0] = 1.0 / np.sqrt(2.0)
        transverse = transverse - np.einsum("ij,ij->", transverse, radial) * radial

    transverse /= frobenius_norm(transverse)
    return radial, transverse


def project(
    value: np.ndarray,
    radial: np.ndarray,
    transverse: np.ndarray,
) -> np.ndarray:
    value_dev = deviatoric(value)
    return np.array(
        [
            np.einsum("ij,ij->", value_dev, radial),
            np.einsum("ij,ij->", value_dev, transverse),
        ]
    )


def interpolate_state(
    left: NewtonState,
    right: NewtonState,
    fraction: float,
) -> dict[str, float | np.ndarray]:
    one_minus = 1.0 - fraction
    return {
        "iteration": left.iteration + fraction,
        "dgamma": one_minus * left.dgamma + fraction * right.dgamma,
        "residual": one_minus * left.residual + fraction * right.residual,
        "stress": one_minus * left.stress + fraction * right.stress,
        "backstress": one_minus * left.backstress + fraction * right.backstress,
        "xi_norm": one_minus * frobenius_norm(left.shifted_stress)
        + fraction * frobenius_norm(right.shifted_stress),
        "yield_radius": one_minus * left.yield_radius + fraction * right.yield_radius,
    }


def animation_frames(
    number_of_states: int,
    transition_frames: int = 22,
    hold_frames: int = 10,
) -> list[tuple[int, float]]:
    frames: list[tuple[int, float]] = [(0, 0.0)] * (2 * hold_frames)
    for index in range(number_of_states - 1):
        frames.extend(
            (index, step / transition_frames)
            for step in range(1, transition_frames + 1)
        )
        frames.extend([(index + 1, 0.0)] * hold_frames)
    frames.extend([(number_of_states - 1, 0.0)] * (3 * hold_frames))
    return frames


def save_animation(
    history: list[NewtonState],
    direction: np.ndarray,
    output: Path,
    preview: Path | None = None,
) -> None:
    """Write the Newton stress-point animation."""
    radial, transverse = projection_basis(direction)
    stress_points = np.array(
        [project(state.stress, radial, transverse) for state in history]
    )
    beta_points = np.array(
        [project(state.backstress, radial, transverse) for state in history]
    )
    xi_norms = np.array([frobenius_norm(state.shifted_stress) for state in history])
    radii = np.array([state.yield_radius for state in history])
    residuals = np.maximum(
        np.abs([state.residual for state in history]),
        1.0e-14,
    )
    iterations = np.arange(len(history))

    circle_left = beta_points[:, 0] - radii
    circle_right = beta_points[:, 0] + radii
    circle_bottom = beta_points[:, 1] - radii
    circle_top = beta_points[:, 1] + radii
    x_min = min(circle_left.min(), stress_points[:, 0].min())
    x_max = max(circle_right.max(), stress_points[:, 0].max())
    y_min = min(circle_bottom.min(), stress_points[:, 1].min())
    y_max = max(circle_top.max(), stress_points[:, 1].max())
    x_pad = 0.10 * (x_max - x_min)
    y_pad = 0.10 * (y_max - y_min)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(12.8, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.55, 1.0))
    stress_axis = figure.add_subplot(grid[:, 0])
    radius_axis = figure.add_subplot(grid[0, 1])
    residual_axis = figure.add_subplot(grid[1, 1])

    frames = animation_frames(len(history))

    def current_frame(index: int, fraction: float) -> dict[str, float | np.ndarray]:
        if fraction == 0.0 or index == len(history) - 1:
            state = history[index]
            return {
                "iteration": float(index),
                "dgamma": state.dgamma,
                "residual": state.residual,
                "stress": state.stress,
                "backstress": state.backstress,
                "xi_norm": frobenius_norm(state.shifted_stress),
                "yield_radius": state.yield_radius,
            }
        return interpolate_state(history[index], history[index + 1], fraction)

    def update(frame_number: int):
        index, fraction = frames[frame_number]
        current = current_frame(index, fraction)
        current_iteration = float(current["iteration"])
        current_stress = project(
            np.asarray(current["stress"]), radial, transverse
        )
        current_beta = project(
            np.asarray(current["backstress"]), radial, transverse
        )
        current_radius = float(current["yield_radius"])
        current_xi_norm = float(current["xi_norm"])

        stress_axis.clear()
        radius_axis.clear()
        residual_axis.clear()

        initial_circle = Circle(
            beta_points[0],
            radii[0],
            fill=False,
            color="0.55",
            linestyle=":",
            linewidth=1.5,
            label="Initial yield surface",
        )
        final_circle = Circle(
            beta_points[-1],
            radii[-1],
            fill=False,
            color="#009E73",
            linestyle="--",
            linewidth=1.8,
            label="Converged yield surface",
        )
        active_circle = Circle(
            current_beta,
            current_radius,
            fill=False,
            color="#E69F00",
            linewidth=2.4,
            label="Current yield surface",
        )
        stress_axis.add_patch(initial_circle)
        stress_axis.add_patch(final_circle)
        stress_axis.add_patch(active_circle)

        completed = index + 1
        path = stress_points[:completed]
        if fraction > 0.0:
            path = np.vstack((path, current_stress))
        stress_axis.plot(
            path[:, 0],
            path[:, 1],
            color="#0072B2",
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="Newton stress path",
        )
        stress_axis.scatter(
            current_stress[0],
            current_stress[1],
            s=90,
            color="#D55E00",
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
            label="Current stress point",
        )
        stress_axis.scatter(
            current_beta[0],
            current_beta[1],
            s=80,
            color="#CC79A7",
            marker="x",
            linewidth=2.0,
            zorder=5,
            label="Backstress center",
        )
        stress_axis.plot(
            [current_beta[0], current_stress[0]],
            [current_beta[1], current_stress[1]],
            color="#D55E00",
            linewidth=1.4,
            alpha=0.8,
        )

        label_offsets = (
            (8, 10),
            (8, 20),
            (8, -20),
            (8, 36),
            (8, -36),
        )
        for point_index in range(completed):
            stress_axis.annotate(
                f"k={point_index}",
                stress_points[point_index],
                xytext=label_offsets[point_index % len(label_offsets)],
                textcoords="offset points",
                fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.7},
            )

        stress_axis.axhline(0.0, color="0.8", linewidth=0.8)
        stress_axis.axvline(0.0, color="0.8", linewidth=0.8)
        stress_axis.set_xlim(x_min - x_pad, x_max + x_pad)
        stress_axis.set_ylim(y_min - y_pad, y_max + y_pad)
        stress_axis.set_aspect("equal", adjustable="box")
        stress_axis.set_title("Stress point in a deviatoric plane")
        stress_axis.set_xlabel(r"Radial coordinate $\mathbf{s}:\mathbf{n}$ (MPa)")
        stress_axis.set_ylabel("Orthogonal deviatoric coordinate (MPa)")
        stress_axis.legend(loc="upper left", fontsize=8, frameon=True)

        progress_iterations = iterations[:completed].astype(float)
        progress_norms = xi_norms[:completed]
        progress_radii = radii[:completed]
        if fraction > 0.0:
            progress_iterations = np.append(progress_iterations, current_iteration)
            progress_norms = np.append(progress_norms, current_xi_norm)
            progress_radii = np.append(progress_radii, current_radius)

        radius_axis.plot(
            progress_iterations,
            progress_norms,
            marker="o",
            color="#0072B2",
            label=r"$\|\mathbf{\xi}_k\|$",
        )
        radius_axis.plot(
            progress_iterations,
            progress_radii,
            marker="s",
            color="#E69F00",
            label=r"$\sqrt{2/3}\,K(\alpha_k)$",
        )
        radius_axis.set_xlim(-0.15, max(len(history) - 0.85, 1.0))
        radius_axis.set_ylim(
            0.9 * min(radii.min(), xi_norms.min()),
            1.08 * max(radii.max(), xi_norms.max()),
        )
        radius_axis.set_title("Yield-surface approach")
        radius_axis.set_xlabel("Newton iteration")
        radius_axis.set_ylabel("Shifted stress norm (MPa)")
        radius_axis.legend(fontsize=9)

        progress_residuals = residuals[:completed]
        progress_residual_iterations = iterations[:completed].astype(float)
        if fraction > 0.0:
            left_log = np.log10(residuals[index])
            right_log = np.log10(residuals[index + 1])
            current_log_residual = 10.0 ** (
                (1.0 - fraction) * left_log + fraction * right_log
            )
            progress_residuals = np.append(
                progress_residuals, current_log_residual
            )
            progress_residual_iterations = np.append(
                progress_residual_iterations, current_iteration
            )

        residual_axis.semilogy(
            progress_residual_iterations,
            progress_residuals,
            marker="o",
            color="#D55E00",
        )
        residual_axis.axhline(
            1.0e-10,
            color="#009E73",
            linestyle="--",
            linewidth=1.5,
            label="Tolerance",
        )
        residual_axis.set_xlim(-0.15, max(len(history) - 0.85, 1.0))
        residual_axis.set_ylim(1.0e-14, 10.0 * residuals.max())
        residual_axis.set_title("Consistency residual")
        residual_axis.set_xlabel("Newton iteration")
        residual_axis.set_ylabel(r"$|R_k|$ (MPa)")
        residual_axis.legend(fontsize=9)

        converged = index == len(history) - 1 and fraction == 0.0
        status = "Converged" if converged else "Newton update"
        figure.suptitle(
            f"J2 radial return — {status}\n"
            f"iteration = {current_iteration:.2f},  "
            f"Δγ = {float(current['dgamma']):.7f},  "
            f"|R| = {abs(float(current['residual'])):.3e} MPa",
            fontsize=15,
        )

        return ()

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=50,
        blit=False,
        repeat=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        output,
        writer=FFMpegWriter(fps=20, bitrate=2_800),
        dpi=100,
    )

    if preview is not None:
        update(len(frames) - 1)
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview, dpi=150)

    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate stress updates during a J2 local Newton solve."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("radial_return_newton.mp4"),
    )
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    parameters = HardeningParameters(
        sigma_y=250.0,
        sigma_u=450.0,
        H_bar=10_000.0,
        theta=0.4,
        delta=250.0,
    )
    sigma_trial = np.diag([900.0, 0.0, 0.0])
    backstress_n = np.zeros((3, 3))
    tolerance = 1.0e-10

    history, direction, mu = compute_newton_history(
        sigma_trial=sigma_trial,
        backstress_n=backstress_n,
        alpha_n=0.0,
        young_modulus=210_000.0,
        poisson_ratio=0.3,
        parameters=parameters,
        tolerance=tolerance,
    )
    verify_history(
        history=history,
        sigma_trial=sigma_trial,
        direction=direction,
        mu=mu,
        tolerance=tolerance,
    )
    save_animation(history, direction, args.output, args.preview)

    print(f"Newton states: {len(history)}")
    print(f"Final residual: {history[-1].residual:.3e} MPa")
    print(f"Video: {args.output.resolve()}")


if __name__ == "__main__":
    main()
