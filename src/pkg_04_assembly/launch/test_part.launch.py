from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command


def generate_launch_description():

    urdf_file = "/home/vishwanath/Downloads/robot_arm_ws/src/pkg_04_assembly/urdf/test_part.urdf"

    robot_description = ParameterValue(
        Command(["cat ", urdf_file]),
        value_type=str
    )

    return LaunchDescription([

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="test_robot_state_publisher",
            parameters=[
                {"robot_description": robot_description}
            ]
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2"
        )
    ])