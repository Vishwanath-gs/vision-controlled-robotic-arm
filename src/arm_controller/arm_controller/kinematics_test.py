import math
import numpy as np


def joint_rotation(axis, angle):
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis

    c = math.cos(angle)
    s = math.sin(angle)
    v = 1 - c

    R = np.array([
        [x*x*v + c,   x*y*v - z*s, x*z*v + y*s],
        [y*x*v + z*s, y*y*v + c,   y*z*v - x*s],
        [z*x*v - y*s, z*y*v + x*s, z*z*v + c]
    ])

    T = np.eye(4)
    T[:3, :3] = R

    return T


def transform(xyz, rpy):
    """
    Homogeneous transform from XYZ + RPY.
    """

    x, y, z = xyz
    roll, pitch, yaw = rpy

    cr = math.cos(roll)
    sr = math.sin(roll)

    cp = math.cos(pitch)
    sp = math.sin(pitch)

    cy = math.cos(yaw)
    sy = math.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    T = np.eye(4)

    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]

    return T


def fk(j1, j2, j3, j4):
    """
    Forward kinematics.

    Returns the TRANSFORM of the physical end-effector/TCP,
    not the part_4 link origin.
    """

    T = np.eye(4)

    # --------------------------------------------------
    # J1
    # --------------------------------------------------

    T = T @ transform(
        [0, 0, 0.03],
        [0, 0, 0]
    )

    T = T @ joint_rotation(
        [0, 0, -1],
        j1
    )

    # --------------------------------------------------
    # J2
    # --------------------------------------------------

    T = T @ transform(
        [0, 0, 0.10],
        [0, 0.0191658, 0]
    )

    T = T @ joint_rotation(
        [0, -1, 0],
        j2
    )

    # --------------------------------------------------
    # J3
    # --------------------------------------------------

    T = T @ transform(
        [0, 0, 0.14],
        [0, 0, 0]
    )

    T = T @ joint_rotation(
        [0, 1, 0],
        j3
    )

    # --------------------------------------------------
    # J4
    # --------------------------------------------------

    T = T @ transform(
        [0.000775924, 0, 0.0849965],
        [0.0010424, 0.00906894, 0.114441]
    )

    T = T @ joint_rotation(
        [0, 0, -1],
        j4
    )

    # --------------------------------------------------
    # PART 4 VISUAL / COLLISION OFFSET
    #
    # URDF:
    # <origin xyz="0 0 -0.355"/>
    #
    # This is the physical end-effector point
    # we want IK to target.
    # --------------------------------------------------

    T = T @ transform(
        [0, 0, -0.355],
        [0, 0, 0]
    )

    return T


def test(name, angles):

    T = fk(
        *[math.radians(a) for a in angles]
    )

    p = T[:3, 3]

    print(name)
    print(f"X = {p[0]:.6f} m")
    print(f"Y = {p[1]:.6f} m")
    print(f"Z = {p[2]:.6f} m")
    print()


if __name__ == "__main__":

    test("Zero", [0, 0, 0, 0])

    test("J2=20", [0, 20, 0, 0])

    test("J3=20", [0, 0, 20, 0])

    test("J2=20 J3=20", [0, 20, 20, 0])

    test("J2=30 J3=30", [0, 30, 30, 0])