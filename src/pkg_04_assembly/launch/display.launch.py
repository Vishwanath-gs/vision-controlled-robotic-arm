from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    urdf_file = "/home/vishwanath/Downloads/robot_arm_ws/src/pkg_04_assembly/urdf/pkg_04_assembly.urdf"

    robot_description = ParameterValue(
        Command(["cat ", urdf_file]),
        value_type=str
    )

    return LaunchDescription([

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[
                {"robot_description": robot_description}
            ]
        ),

        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui"
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2"
        )
    ])