"""Plot rocket trajectory simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from simulator import simulate


OUTPUT_DIR = Path("output")


def plot_altitude(time_s, altitude_m) -> None:
    """Create and save the altitude-over-time plot."""
    fig, ax = plt.subplots()
    ax.plot(time_s, altitude_m)
    ax.set_title("Rocket Altitude vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "altitude_vs_time.png", dpi=150)
    plt.close(fig)


def plot_flight_path(x_m, altitude_m) -> None:
    """Create and save the 2D flight path plot."""
    fig, ax = plt.subplots()
    ax.plot(x_m, altitude_m)
    ax.set_title("2D Rocket Flight Path")
    ax.set_xlabel("Downrange Distance (m)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "flight_path.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the simulation and save trajectory plots."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    results = simulate()
    plot_altitude(results["time_s"], results["altitude_m"])
    plot_flight_path(results["x_m"], results["altitude_m"])


if __name__ == "__main__":
    main()
