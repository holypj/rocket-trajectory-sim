"""Design notebook: plot rocket trajectory simulation results."""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# 2. User controls / given values
# ---------------------------------------------------------------------------

# Output files are collected here so generated plots are easy to find.
OUTPUT_DIR = Path("output")

# Matplotlib needs a writable cache directory when running in this workspace.
MPL_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import simulate


# ---------------------------------------------------------------------------
# 3. Constants
# ---------------------------------------------------------------------------

ALTITUDE_PLOT_PATH = OUTPUT_DIR / "altitude_vs_time.png"
FLIGHT_PATH_PLOT_PATH = OUTPUT_DIR / "flight_path.png"
PLOT_DPI = 150


# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------

def plot_altitude(time_s, altitude_m) -> None:
    """Create and save the altitude-over-time plot."""
    # This figure shows vertical motion through the whole flight.
    fig, ax = plt.subplots()
    ax.plot(time_s, altitude_m)
    ax.set_title("Rocket Altitude vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True)

    # Tight layout reduces clipped labels before saving the PNG.
    fig.tight_layout()
    fig.savefig(ALTITUDE_PLOT_PATH, dpi=PLOT_DPI)
    plt.close(fig)


def plot_flight_path(x_m, altitude_m) -> None:
    """Create and save the 2D flight path plot."""
    # This figure shows the rocket's horizontal and vertical position together.
    fig, ax = plt.subplots()
    ax.plot(x_m, altitude_m)
    ax.set_title("2D Rocket Flight Path")
    ax.set_xlabel("Downrange Distance (m)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True)

    # Equal aspect ratio keeps distances visually meaningful on both axes.
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(FLIGHT_PATH_PLOT_PATH, dpi=PLOT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Main calculation
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the simulation and save trajectory plots."""
    # Ensure the output folder exists before Matplotlib writes image files.
    OUTPUT_DIR.mkdir(exist_ok=True)

    # The simulator returns one NumPy array per quantity we want to plot.
    results = simulate()

    # -----------------------------------------------------------------------
    # 6. Results
    # -----------------------------------------------------------------------

    time_s = results["time_s"]
    x_m = results["x_m"]
    altitude_m = results["altitude_m"]

    # -----------------------------------------------------------------------
    # 7. Plotting
    # -----------------------------------------------------------------------

    plot_altitude(time_s, altitude_m)
    plot_flight_path(x_m, altitude_m)


if __name__ == "__main__":
    main()
