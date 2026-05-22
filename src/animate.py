"""Create an animated GIF of the rocket trajectory."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("output")
MPL_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from simulator import simulate


TARGET_FRAME_COUNT = 100
GIF_FPS = 20


def sample_frame_indices(point_count: int) -> np.ndarray:
    """Select roughly 100 evenly spaced trajectory samples for a small GIF."""
    frame_count = min(TARGET_FRAME_COUNT, point_count)
    return np.linspace(0, point_count - 1, frame_count, dtype=int)


def axis_limits(values: np.ndarray, lower_bound: float = 0.0) -> tuple[float, float]:
    """Return padded fixed limits so the axes cover the whole trajectory."""
    max_value = float(np.max(values))
    padding = max(max_value * 0.05, 1.0)
    return lower_bound, max_value + padding


def create_animation() -> None:
    """Run the simulation and save a compact animated flight-path GIF."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    results = simulate()
    time_s = results["time_s"]
    x_m = results["x_m"]
    altitude_m = results["altitude_m"]
    frame_indices = sample_frame_indices(len(time_s))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Rocket Flight Animation")
    ax.set_xlabel("Downrange Distance (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_xlim(axis_limits(x_m))
    ax.set_ylim(axis_limits(altitude_m))
    ax.grid(True)

    # The trailing line shows the path flown up to the current animation frame.
    (trail_line,) = ax.plot([], [], color="tab:blue", linewidth=2)

    # The marker represents the rocket's current position along the trajectory.
    (rocket_marker,) = ax.plot([], [], marker="o", color="tab:red", markersize=6)

    readout = ax.text(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    def update(frame_number: int):
        index = frame_indices[frame_number]

        trail_line.set_data(x_m[: index + 1], altitude_m[: index + 1])
        rocket_marker.set_data([x_m[index]], [altitude_m[index]])
        readout.set_text(
            f"Time: {time_s[index]:.1f} s\nAltitude: {altitude_m[index]:.0f} m"
        )

        return trail_line, rocket_marker, readout

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / GIF_FPS,
        blit=True,
    )

    fig.tight_layout()
    animation.save(OUTPUT_DIR / "flight.gif", writer=PillowWriter(fps=GIF_FPS))
    plt.close(fig)


if __name__ == "__main__":
    create_animation()
