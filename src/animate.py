"""Design notebook: create an animated GIF of the rocket trajectory."""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 2. User controls / given values
# ---------------------------------------------------------------------------

# Output files are collected here so generated media are easy to find.
OUTPUT_DIR = Path("output")

# Matplotlib needs a writable cache directory when running in this workspace.
MPL_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from simulator import simulate


# ---------------------------------------------------------------------------
# 3. Constants
# ---------------------------------------------------------------------------

GIF_PATH = OUTPUT_DIR / "flight.gif"
TARGET_FRAME_COUNT = 100
GIF_FPS = 20
ROCKET_SIZE_AXES = 0.045

# The rocket is drawn in local display-style coordinates before being scaled
# into data units. The positive x direction is the nose-forward direction.
ROCKET_SHAPE = np.array(
    [
        [1.00, 0.00],  # nose cone
        [0.45, 0.18],  # upper body shoulder
        [-0.45, 0.18],  # upper body tail
        [-0.70, 0.42],  # upper fin tip
        [-0.62, 0.14],  # upper fin root
        [-0.95, 0.08],  # engine end
        [-0.95, -0.08],  # engine end
        [-0.62, -0.14],  # lower fin root
        [-0.70, -0.42],  # lower fin tip
        [-0.45, -0.18],  # lower body tail
        [0.45, -0.18],  # lower body shoulder
    ]
)


# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------

def sample_frame_indices(point_count: int) -> np.ndarray:
    """Select roughly 100 evenly spaced trajectory samples for a small GIF."""
    # The simulator may produce many more samples than the animation needs.
    frame_count = min(TARGET_FRAME_COUNT, point_count)
    return np.linspace(0, point_count - 1, frame_count, dtype=int)


def axis_limits(values: np.ndarray, lower_bound: float = 0.0) -> tuple[float, float]:
    """Return padded fixed limits so the axes cover the whole trajectory."""
    # Fixed axes make the animation easier to read because the view does not pan.
    max_value = float(np.max(values))
    padding = max(max_value * 0.05, 1.0)
    return lower_bound, max_value + padding


def rocket_vertices(
    x_m: float,
    altitude_m: float,
    angle_rad: float,
    x_range_m: float,
    y_range_m: float,
) -> np.ndarray:
    """Return rocket polygon vertices centered on the current flight position."""
    # Start with a small rocket in normalized display proportions.
    scaled_shape = ROCKET_SHAPE * ROCKET_SIZE_AXES

    # Rotate that shape so the nose points along the velocity vector.
    rotation = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    rotated_shape = scaled_shape @ rotation.T

    # Convert normalized offsets into data units. Scaling x and y by their own
    # axis ranges prevents the rocket from looking stretched on non-square axes.
    data_offsets = rotated_shape * np.array([x_range_m, y_range_m])

    # Move the rotated rocket so its center sits at the current trajectory point.
    return data_offsets + np.array([x_m, altitude_m])


# ---------------------------------------------------------------------------
# 5. Main calculation
# ---------------------------------------------------------------------------

def create_animation() -> None:
    """Run the simulation and save a compact animated flight-path GIF."""
    # Ensure the output folder exists before the GIF writer saves anything.
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Run the same physics calculation used by the static plots.
    results = simulate()

    # -----------------------------------------------------------------------
    # 6. Results
    # -----------------------------------------------------------------------

    time_s = results["time_s"]
    x_m = results["x_m"]
    altitude_m = results["altitude_m"]
    vx_mps = results["vx_mps"]
    vy_mps = results["vy_mps"]
    frame_indices = sample_frame_indices(len(time_s))

    # -----------------------------------------------------------------------
    # 7. Animation
    # -----------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Rocket Flight Animation")
    ax.set_xlabel("Downrange Distance (m)")
    ax.set_ylabel("Altitude (m)")

    # These fixed limits cover the full trajectory for every animation frame.
    x_limits = axis_limits(x_m)
    y_limits = axis_limits(altitude_m)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.grid(True)

    # Store ranges once because the rocket drawing uses them every frame.
    x_range_m = x_limits[1] - x_limits[0]
    y_range_m = y_limits[1] - y_limits[0]

    # The trailing line shows the path flown up to the current animation frame.
    (trail_line,) = ax.plot([], [], color="tab:blue", linewidth=2)

    # The polygon represents the rocket body, nose cone, and fins.
    rocket_patch = Polygon(
        rocket_vertices(0.0, 0.0, 0.0, x_range_m, y_range_m),
        closed=True,
        facecolor="tab:red",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax.add_patch(rocket_patch)

    # The readout gives a quick numerical check as the animation advances.
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
        """Update the trail, rocket body, and readout for one GIF frame."""
        index = frame_indices[frame_number]

        # Draw all trajectory points up to the current frame.
        trail_line.set_data(x_m[: index + 1], altitude_m[: index + 1])

        # Pitch the rocket along the current velocity vector as it arcs over.
        flight_angle_rad = np.atan2(vy_mps[index], vx_mps[index])
        rocket_patch.set_xy(
            rocket_vertices(
                x_m[index],
                altitude_m[index],
                flight_angle_rad,
                x_range_m,
                y_range_m,
            )
        )

        # Update the text readout in the upper-left corner of the axes.
        readout.set_text(
            f"Time: {time_s[index]:.1f} s\nAltitude: {altitude_m[index]:.0f} m"
        )

        return trail_line, rocket_patch, readout

    # FuncAnimation asks update() to draw each selected frame.
    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / GIF_FPS,
        blit=True,
    )

    # PillowWriter produces a compact animated GIF that can be embedded in README.
    fig.tight_layout()
    animation.save(GIF_PATH, writer=PillowWriter(fps=GIF_FPS))
    plt.close(fig)


if __name__ == "__main__":
    create_animation()
