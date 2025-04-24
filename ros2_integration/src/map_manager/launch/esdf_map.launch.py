from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Define the path to the YAML parameter file
    param_file_path = os.path.join(
        get_package_share_directory('map_manager'),
        'cfg',
        'map_param.yaml'
    )

    # Create the node with the parameter file
    esdf_map_node = Node(
        package='map_manager',
        executable='esdf_map_node',
        name='map_manager_node',
        output='screen',
        parameters=[param_file_path]
    )

    return LaunchDescription([
        esdf_map_node
    ])