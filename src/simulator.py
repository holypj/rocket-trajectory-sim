"""2D point-mass rocket trajectory simulator."""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# User-tunable simulation parameters
# ---------------------------------------------------------------------------

# Rocket masses, in kilograms.
DRY_MASS_KG = 50.0
PROPELLANT_MASS_KG = 100.0

# Engine thrust and burn duration.
THRUST_N = 2_000.0
BURN_TIME_S = 8.0

# Launch geometry and numerical integration step.
LAUNCH_ANGLE_DEG = 75.0
TIME_STEP_S = 0.02


# Constant downward gravitational acceleration, in meters per second squared.
GRAVITY_MPS2 = 9.81


def rocket_mass(time_s: float) -> float:
    """Return rocket mass at a given time as propellant burns linearly."""
    if time_s >= BURN_TIME_S:
        return DRY_MASS_KG

    propellant_remaining = PROPELLANT_MASS_KG * (1.0 - time_s / BURN_TIME_S)
    return DRY_MASS_KG + propellant_remaining


def thrust_acceleration(time_s: float) -> np.ndarray:
    """Return thrust acceleration components at a given time."""
    if time_s >= BURN_TIME_S:
        return np.array([0.0, 0.0])

    # This simple model keeps thrust direction fixed at the launch angle.
    launch_angle_rad = math.radians(LAUNCH_ANGLE_DEG)
    thrust_direction = np.array(
        [math.cos(launch_angle_rad), math.sin(launch_angle_rad)]
    )

    # Newton's second law: acceleration equals force divided by current mass.
    return (THRUST_N / rocket_mass(time_s)) * thrust_direction


def derivatives(time_s: float, state: np.ndarray) -> np.ndarray:
    """Return derivatives for position and velocity at the current state."""
    velocity = state[2:4]

    # Gravity always accelerates the rocket downward.
    gravity_acceleration = np.array([0.0, -GRAVITY_MPS2])

    # Net acceleration is thrust acceleration plus gravity acceleration.
    acceleration = thrust_acceleration(time_s) + gravity_acceleration

    # The state is [x, y, vx, vy], so its derivative is [vx, vy, ax, ay].
    return np.array([velocity[0], velocity[1], acceleration[0], acceleration[1]])


def rk4_step(time_s: float, state: np.ndarray, dt_s: float) -> np.ndarray:
    """Advance the simulation by one fixed time step using RK4 integration."""
    # RK4 samples the slope at the beginning, midpoint, and end of the step.
    k1 = derivatives(time_s, state)
    k2 = derivatives(time_s + 0.5 * dt_s, state + 0.5 * dt_s * k1)
    k3 = derivatives(time_s + 0.5 * dt_s, state + 0.5 * dt_s * k2)
    k4 = derivatives(time_s + dt_s, state + dt_s * k3)

    return state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate() -> dict[str, np.ndarray]:
    """Run the rocket trajectory simulation until the rocket returns to ground."""
    time_s = 0.0

    # The rocket starts on the ground at rest. State is [x, y, vx, vy].
    state = np.array([0.0, 0.0, 0.0, 0.0])

    times = [time_s]
    states = [state.copy()]

    while True:
        next_state = rk4_step(time_s, state, TIME_STEP_S)
        next_time_s = time_s + TIME_STEP_S

        # Stop once the rocket has lifted off and returned to the ground.
        if next_time_s > 0.0 and next_state[1] <= 0.0 and state[1] > 0.0:
            # Linearly interpolate the final point so the trajectory ends at y=0.
            ground_fraction = state[1] / (state[1] - next_state[1])
            ground_time_s = time_s + ground_fraction * TIME_STEP_S
            ground_state = state + ground_fraction * (next_state - state)
            ground_state[1] = 0.0

            times.append(ground_time_s)
            states.append(ground_state.copy())
            break

        times.append(next_time_s)
        states.append(next_state.copy())

        time_s = next_time_s
        state = next_state

    state_history = np.array(states)

    return {
        "time_s": np.array(times),
        "x_m": state_history[:, 0],
        "altitude_m": state_history[:, 1],
        "vx_mps": state_history[:, 2],
        "vy_mps": state_history[:, 3],
    }
