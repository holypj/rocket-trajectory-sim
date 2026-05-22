"""Design notebook: 2D point-mass rocket trajectory simulator."""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# 2. User controls / given values
# ---------------------------------------------------------------------------

# Rocket masses, in kilograms.
# Dry mass is the empty rocket. Propellant mass is the fuel available to burn.
DRY_MASS_KG = 50.0
PROPELLANT_MASS_KG = 100.0

# Engine settings.
# Thrust is treated as constant until the burn time ends.
THRUST_N = 2_000.0
BURN_TIME_S = 8.0

# Launch and solver settings.
# The rocket points at this fixed launch angle while thrust is active.
LAUNCH_ANGLE_DEG = 75.0
TIME_STEP_S = 0.02


# ---------------------------------------------------------------------------
# 3. Constants
# ---------------------------------------------------------------------------

# Gravity is constant and points downward. This version does not include drag.
GRAVITY_MPS2 = 9.81


# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------

def rocket_mass(time_s: float) -> float:
    """Return rocket mass at a given time as propellant burns linearly."""
    # After burnout, all propellant is gone and only dry mass remains.
    if time_s >= BURN_TIME_S:
        return DRY_MASS_KG

    # Before burnout, propellant decreases from full to zero at a steady rate.
    propellant_remaining = PROPELLANT_MASS_KG * (1.0 - time_s / BURN_TIME_S)
    return DRY_MASS_KG + propellant_remaining


def thrust_acceleration(time_s: float) -> np.ndarray:
    """Return thrust acceleration components at a given time."""
    # No engine force is produced after the burn phase ends.
    if time_s >= BURN_TIME_S:
        return np.array([0.0, 0.0])

    # Convert the launch angle into a unit vector for the thrust direction.
    launch_angle_rad = math.radians(LAUNCH_ANGLE_DEG)
    thrust_direction = np.array(
        [math.cos(launch_angle_rad), math.sin(launch_angle_rad)]
    )

    # Newton's second law: acceleration = force / mass.
    # As fuel burns away, mass falls and the same thrust produces more acceleration.
    return (THRUST_N / rocket_mass(time_s)) * thrust_direction


def derivatives(time_s: float, state: np.ndarray) -> np.ndarray:
    """Return derivatives for position and velocity at the current state."""
    # State vector layout: x position, y position, x velocity, y velocity.
    velocity = state[2:4]

    # Gravity contributes only to vertical acceleration.
    gravity_acceleration = np.array([0.0, -GRAVITY_MPS2])

    # The total acceleration is the sum of thrust acceleration and gravity.
    acceleration = thrust_acceleration(time_s) + gravity_acceleration

    # The derivative of position is velocity; the derivative of velocity is acceleration.
    return np.array([velocity[0], velocity[1], acceleration[0], acceleration[1]])


def rk4_step(time_s: float, state: np.ndarray, dt_s: float) -> np.ndarray:
    """Advance the simulation by one fixed time step using RK4 integration."""
    # RK4 estimates the slope at the start of the interval.
    k1 = derivatives(time_s, state)

    # Then it samples two midpoint slopes using half-step predictions.
    k2 = derivatives(time_s + 0.5 * dt_s, state + 0.5 * dt_s * k1)
    k3 = derivatives(time_s + 0.5 * dt_s, state + 0.5 * dt_s * k2)

    # Finally it samples the slope at the end of the interval.
    k4 = derivatives(time_s + dt_s, state + dt_s * k3)

    # The weighted average of these four slopes gives the next state.
    return state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# 5. Main calculation
# ---------------------------------------------------------------------------

def simulate() -> dict[str, np.ndarray]:
    """Run the rocket trajectory simulation until the rocket returns to ground."""
    # Start clock at launch.
    time_s = 0.0

    # The rocket starts on the ground at rest: x=0, y=0, vx=0, vy=0.
    state = np.array([0.0, 0.0, 0.0, 0.0])

    # Store every state so plots and animations can use the full trajectory.
    times = [time_s]
    states = [state.copy()]

    while True:
        # Advance one fixed time step using the RK4 solver.
        next_state = rk4_step(time_s, state, TIME_STEP_S)
        next_time_s = time_s + TIME_STEP_S

        # Stop once the rocket has lifted off and crossed back down to ground level.
        if next_time_s > 0.0 and next_state[1] <= 0.0 and state[1] > 0.0:
            # Interpolate between the last above-ground point and first below-ground
            # point so the final sample lands exactly on y=0.
            ground_fraction = state[1] / (state[1] - next_state[1])
            ground_time_s = time_s + ground_fraction * TIME_STEP_S
            ground_state = state + ground_fraction * (next_state - state)
            ground_state[1] = 0.0

            times.append(ground_time_s)
            states.append(ground_state.copy())
            break

        # Keep the new point and continue the flight.
        times.append(next_time_s)
        states.append(next_state.copy())
        time_s = next_time_s
        state = next_state

    # Convert the collected Python lists into NumPy arrays for analysis and plotting.
    state_history = np.array(states)

    # -----------------------------------------------------------------------
    # 6. Results
    # -----------------------------------------------------------------------

    return {
        "time_s": np.array(times),
        "x_m": state_history[:, 0],
        "altitude_m": state_history[:, 1],
        "vx_mps": state_history[:, 2],
        "vy_mps": state_history[:, 3],
    }


def main() -> None:
    """Print a small text summary when the simulator is run directly."""
    results = simulate()
    apogee_m = float(np.max(results["altitude_m"]))
    flight_time_s = float(results["time_s"][-1])
    downrange_m = float(results["x_m"][-1])

    print(f"Apogee: {apogee_m:.1f} m")
    print(f"Flight time: {flight_time_s:.1f} s")
    print(f"Downrange distance: {downrange_m:.1f} m")


if __name__ == "__main__":
    main()
