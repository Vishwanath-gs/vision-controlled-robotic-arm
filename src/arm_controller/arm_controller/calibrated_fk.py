import math


# Measured zero-pose position
X0 = 0.005783
Y0 = 0.0
Z0 = 0.354923


def position_fk(j1, j2, j3):
    """
    Position FK for the first 3 joints.

    Angles are supplied in radians.
    """

    # Effective arm lengths from the URDF chain
    L2 = 0.10
    L3 = 0.14
    L4 = 0.0849965

    # Approximate shoulder/elbow pitch contribution
    reach = (
        L2 * math.sin(j2)
        + L3 * math.sin(j2 - j3)
        + L4 * math.sin(j2 - j3)
    )

    height = (
        Z0
        + L2 * (math.cos(j2) - 1.0)
        + L3 * (
            math.cos(j2 - j3) - 1.0
        )
        + L4 * (
            math.cos(j2 - j3) - 1.0
        )
    )

    x = X0 - reach * math.cos(j1)
    y = Y0 - reach * math.sin(j1)

    return x, y, height


if __name__ == "__main__":

    j1 = math.radians(0)
    j2 = math.radians(20)
    j3 = math.radians(0)

    x, y, z = position_fk(j1, j2, j3)

    print("Calibrated FK")
    print(f"X = {x:.6f} m")
    print(f"Y = {y:.6f} m")
    print(f"Z = {z:.6f} m")
    