import math
import numpy as np
import xml.etree.ElementTree as ET


URDF = "/home/vishwanath/Downloads/robot_arm_ws/src/pkg_04_assembly/urdf/pkg_04_assembly.urdf"


def rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

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

    return Rz @ Ry @ Rx


def transform(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def axis_rotation(axis, angle):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    x, y, z = axis

    c = math.cos(angle)
    s = math.sin(angle)
    v = 1 - c

    R = np.array([
        [x*x*v+c,   x*y*v-z*s, x*z*v+y*s],
        [y*x*v+z*s, y*y*v+c,   y*z*v-x*s],
        [z*x*v-y*s, z*y*v+x*s, z*z*v+c]
    ])

    T = np.eye(4)
    T[:3, :3] = R
    return T


def get_joint_data():

    root = ET.parse(URDF).getroot()

    joints = {}

    for joint in root.findall("joint"):

        name = joint.get("name")

        if name not in [
            "revolute_1",
            "revolute_2",
            "revolute_3",
            "revolute_4"
        ]:
            continue

        origin = joint.find("origin")
        axis = joint.find("axis")

        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
        rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
        axis_xyz = [float(x) for x in axis.get("xyz").split()]

        joints[name] = {
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis_xyz
        }

    return joints


def forward_kinematics(j1, j2, j3, j4):

    joints = get_joint_data()

    angles = {
        "revolute_1": j1,
        "revolute_2": j2,
        "revolute_3": j3,
        "revolute_4": j4
    }

    # Gazebo model spawn position
    T = np.eye(4)
    T[:3, 3] = [0, 0, 0.10]

    for name in [
        "revolute_1",
        "revolute_2",
        "revolute_3",
        "revolute_4"
    ]:

        joint = joints[name]

        # Joint origin
        T = T @ transform(
            joint["xyz"],
            joint["rpy"]
        )

        # Joint rotation
        T = T @ axis_rotation(
            joint["axis"],
            angles[name]
        )

    # Part 4 visual/end-effector offset
    tip_offset = np.array([
        0.0,
        0.0,
        -0.355
    ])

    tip = T[:3, 3] + T[:3, :3] @ tip_offset

    return tip


if __name__ == "__main__":

    angles = [
        math.radians(0),
        math.radians(0),
        math.radians(0),
        math.radians(0)
    ]

    x, y, z = forward_kinematics(*angles)

    print("URDF-based FK")
    print(f"X = {x:.6f} m")
    print(f"Y = {y:.6f} m")
    print(f"Z = {z:.6f} m")