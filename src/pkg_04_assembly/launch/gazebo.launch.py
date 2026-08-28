from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_path = get_package_share_directory('pkg_04_assembly')
    share_path = os.path.dirname(pkg_path)
    world_file = os.path.join(
    pkg_path,
    'worlds',
    'robot_test.sdf'
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': f'-r {world_file}'
        }.items()
    )

    robot_description = os.path.join(
        pkg_path,
        'urdf',
        'pkg_04_assembly.urdf'
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'vision_robot_arm',
            '-file', robot_description,
            '-x', '0',
            '-y', '0',
            '-z', '0.1'
        ],
        output='screen'
    )

    return LaunchDescription([
       SetEnvironmentVariable(
          name='GZ_SIM_RESOURCE_PATH',
          value=share_path
       ),
       gazebo,
       spawn_robot
    ])