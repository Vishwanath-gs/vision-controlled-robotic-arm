import math


# Link lengths from the URDF
L1 = 0.10
L2 = 0.14
L3 = 0.085


def inverse_kinematics(x, y, z):
    """
    Basic position IK.

    J1 controls the base rotation.
    J2 and J3 control the arm reach.
    J4 is kept at zero for now.
    """

    # -------------------------
    # J1: base rotation
    # -------------------------
    j1 = math.atan2(y, x)

    # Horizontal distance from base axis
    r = math.sqrt(x*x + y*y)

    # -------------------------
    # Vertical position
    # -------------------------
    # Base height
    z_relative = z - 0.03

    # -------------------------
    # 2-link planar IK
    # -------------------------
    distance_sq = r*r + z_relative*z_relative

    cos_j3 = (
        distance_sq - L1*L1 - L2*L2
    ) / (2 * L1 * L2)

    # Check reachability
    if cos_j3 < -1.0 or cos_j3 > 1.0:
        return None

    cos_j3 = max(-1.0, min(1.0, cos_j3))

    # Elbow-down solution
    j3 = math.acos(cos_j3)

    j2 = (
        math.atan2(z_relative, r)
        -
        math.atan2(
            L2 * math.sin(j3),
            L1 + L2 * math.cos(j3)
        )
    )

    # J4 orientation
    j4 = 0.0

    return [
        math.degrees(j1),
        math.degrees(j2),
        math.degrees(j3),
        math.degrees(j4)
    ]


if __name__ == "__main__":

    # Test target
    x = 0.10
    y = 0.00
    z = 0.20

    result = inverse_kinematics(x, y, z)

    if result is None:
        print("Target is unreachable.")

    else:
        print("IK solution:")
        print(f"J1 = {result[0]:.2f}°")
        print(f"J2 = {result[1]:.2f}°")
        print(f"J3 = {result[2]:.2f}°")
        print(f"J4 = {result[3]:.2f}°")