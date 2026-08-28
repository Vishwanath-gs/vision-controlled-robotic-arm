import math
import numpy as np


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def T_from_xyz_rpy(x, y, z, roll, pitch, yaw):

    T = np.eye(4)

    # URDF RPY convention
    T[:3, :3] = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    T[:3, 3] = [x, y, z]

    return T


def joint_rotation(axis, angle):

    ax = np.array(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)

    x, y, z = ax

    c = math.cos(angle)
    s = math.sin(angle)
    v = 1.0 - c

    return np.array([
        [x*x*v+c,   x*y*v-z*s, x*z*v+y*s],
        [y*x*v+z*s, y*y*v+c,   y*z*v-x*s],
        [z*x*v-y*s, z*y*v+x*s, z*z*v+c]
    ])


def rotation_transform(axis, angle):

    T = np.eye(4)
    T[:3, :3] = joint_rotation(axis, angle)
    return T


def forward_kinematics(j1, j2, j3, j4):

    # Gazebo model spawn position
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 0.10]

    # world -> root -> part_1
    # fixed joints have zero transform

    # --------------------------------------------------
    # J1
    # origin = (0, 0, 0.03)
    # rpy ≈ (0, 0, 0)
    # axis = (0, 0, -1)
    # --------------------------------------------------

    T = T @ T_from_xyz_rpy(
        0.0, 0.0, 0.03,
        0.0, 0.0, 0.0
    )

    T = T @ rotation_transform(
        [0, 0, -1],
        j1
    )

    # --------------------------------------------------
    # J2
    # origin = (0, 0, 0.10)
    # rpy = (0, 0.0191658, 0)
    # axis = (0, -1, 0)
    # --------------------------------------------------

    T = T @ T_from_xyz_rpy(
        0.0, 0.0, 0.10,
        0.0, 0.0191658, 0.0
    )

    T = T @ rotation_transform(
        [0, -1, 0],
        j2
    )

    # --------------------------------------------------
    # J3
    # origin = (0, 0, 0.14)
    # axis = (0, 1, 0)
    # --------------------------------------------------

    T = T @ T_from_xyz_rpy(
        0.0, 0.0, 0.14,
        0.0, 0.0, 0.0
    )

    T = T @ rotation_transform(
        [0, 1, 0],
        j3
    )

    # --------------------------------------------------
    # J4
    # origin = (0.000775924, 0, 0.0849965)
    # rpy = (0.0010424, 0.00906894, 0.114441)
    # axis = (0, 0, -1)
    # --------------------------------------------------

    T = T @ T_from_xyz_rpy(
        0.000775924,
        0.0,
        0.0849965,
        0.0010424,
        0.00906894,
        0.114441
    )

    T = T @ rotation_transform(
        [0, 0, -1],
        j4
    )

        # Physical TCP is 0.355 m along part_4 local -Z
    TCP_OFFSET = np.array([0.0, 0.0, -0.355])

    tcp_position = T[:3, 3] + T[:3, :3] @ TCP_OFFSET

    return tcp_position