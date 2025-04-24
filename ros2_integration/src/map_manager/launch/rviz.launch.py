import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get the package directory
    package_name = 'map_manager'  
    rviz_config_path = os.path.join(
        get_package_share_directory(package_name), 'rviz', 'map.rviz' 
    )

    # Declare the launch argument for RViz configuration file
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=rviz_config_path,
        description='Full path to the RViz config file'
    )

    # RViz Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')]
    )

    return LaunchDescription([
        rviz_config_arg,
        rviz_node
    ])
