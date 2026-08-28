import math
import random

from .kinematics_test import fk


# Actual mechanical joint limits in degrees
LIMITS = {
    "j1": (-180.0, 180.0),
    "j2": (-104.9, 107.1),
    "j3": (-136.5, 135.5),
}


def position_error(target, angles):
    """
    Return squared TCP position error in m².
    """

    T = fk(
        math.radians(angles[0]),
        math.radians(angles[1]),
        math.radians(angles[2]),
        0.0
    )

    actual = T[:3, 3]

    dx = actual[0] - target[0]
    dy = actual[1] - target[1]
    dz = actual[2] - target[2]

    return dx * dx + dy * dy + dz * dz


def movement_cost(angles, current):
    """
    Measure how much the joints moved from the current position.

    J1 wrap-around is handled so that:
    +179° -> -179° is treated as a 2° movement,
    not a 358° movement.
    """

    total = 0.0

    for i in range(3):

        difference = angles[i] - current[i]

        if i == 0:
            while difference > 180.0:
                difference -= 360.0

            while difference < -180.0:
                difference += 360.0

        total += difference * difference

    return total


def numerical_ik(
    target,
    current=(0.0, 0.0, 0.0),
    iterations=5000
):
    """
    Numerical inverse kinematics.

    Strategy:
    1. Search many configurations.
    2. Find highly accurate TCP solutions.
    3. Among accurate solutions, prefer the one
       requiring the least joint movement.
    """

    candidates = []

    # Starting configurations
    starts = [
        list(current),

        [0.0, 0.0, 0.0],

        [0.0, 30.0, -30.0],
        [0.0, -30.0, 30.0],

        [90.0, 0.0, 0.0],
        [-90.0, 0.0, 0.0],

        [180.0, 0.0, 0.0],
        [-180.0, 0.0, 0.0],

        [0.0, 80.0, -40.0],
        [0.0, -80.0, 40.0],
    ]

    for start in starts:

        angles = start.copy()

        # Make sure starting configuration is valid
        for i, name in enumerate(["j1", "j2", "j3"]):

            minimum, maximum = LIMITS[name]

            angles[i] = max(
                minimum,
                min(maximum, angles[i])
            )

        error = position_error(
            target,
            angles
        )

        step_size = 8.0

        for iteration in range(iterations):

            candidate = angles.copy()

            # Try all three joints in rotation
            joint = iteration % 3

            candidate[joint] += random.uniform(
                -step_size,
                step_size
            )

            name = ["j1", "j2", "j3"][joint]

            minimum, maximum = LIMITS[name]

            candidate[joint] = max(
                minimum,
                min(maximum, candidate[joint])
            )

            candidate_error = position_error(
                target,
                candidate
            )

            if candidate_error < error:

                angles = candidate
                error = candidate_error

            step_size *= 0.9995

            if error < 1e-12:
                break

        candidates.append(
            (
                angles.copy(),
                math.sqrt(error)
            )
        )

    # --------------------------------------------------
    # Select solution
    # --------------------------------------------------

    # Maximum acceptable TCP error:
    # 1 mm
    MAX_POSITION_ERROR = 0.001

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate[1] <= MAX_POSITION_ERROR
    ]

    if not valid_candidates:

        # No sufficiently accurate solution
        # Return the most accurate one found.
        best_angles, best_error = min(
            candidates,
            key=lambda x: x[1]
        )

        return best_angles, best_error

    # Among accurate solutions choose the one
    # requiring the least movement from current.
    best_angles, best_error = min(
        valid_candidates,
        key=lambda x: movement_cost(
            x[0],
            current
        )
    )

    return best_angles, best_error


if __name__ == "__main__":

    target = (
        0.05,
        0.00,
        0.30
    )

    current = (
        0.0,
        0.0,
        0.0
    )

    solution, error = numerical_ik(
        target,
        current=current
    )

    print("Numerical IK")
    print("-------------------------")

    print("Target:")
    print(f"X = {target[0]:.4f} m")
    print(f"Y = {target[1]:.4f} m")
    print(f"Z = {target[2]:.4f} m")

    if solution is None:

        print("\nNo solution found.")

    else:

        print("\nSolution:")
        print(f"J1 = {solution[0]:.2f}°")
        print(f"J2 = {solution[1]:.2f}°")
        print(f"J3 = {solution[2]:.2f}°")
        print("J4 = 0.00°")

        print(
            f"\nPosition error = "
            f"{error * 1000:.3f} mm"
        )