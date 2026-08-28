import math
from ik_numeric import numerical_ik
from kinematics_test import fk


def verify(target, current=(0.0, 0.0, 0.0)):

    solution, error = numerical_ik(
        target,
        current=current
    )

    if solution is None:
        print("❌ No IK solution found.")
        return

    j1, j2, j3 = solution

    # Run the solution through FK
    T = fk(
        math.radians(j1),
        math.radians(j2),
        math.radians(j3),
        0.0
    )

    actual = T[:3, 3]

    print("\n==============================")
    print("        IK VERIFICATION")
    print("==============================")

    print("\nTarget:")
    print(f"X = {target[0]:.6f} m")
    print(f"Y = {target[1]:.6f} m")
    print(f"Z = {target[2]:.6f} m")

    print("\nIK angles:")
    print(f"J1 = {j1:.2f}°")
    print(f"J2 = {j2:.2f}°")
    print(f"J3 = {j3:.2f}°")
    print("J4 = 0.00°")

    print("\nFK result:")
    print(f"X = {actual[0]:.6f} m")
    print(f"Y = {actual[1]:.6f} m")
    print(f"Z = {actual[2]:.6f} m")

    print(f"\nPosition error = {error * 1000:.3f} mm")

    if error < 0.001:
        print("✅ IK + FK verification successful!")
    else:
        print("⚠️ Position error is too large.")


if __name__ == "__main__":

    targets = [
       (0.05, 0.00, 0.30),
       (0.08, 0.00, 0.30),
       (0.03, 0.00, 0.35),
       (0.00, 0.05, 0.30),
       (-0.05, 0.00, 0.30),
    ]

    for target in targets:
      verify(target)