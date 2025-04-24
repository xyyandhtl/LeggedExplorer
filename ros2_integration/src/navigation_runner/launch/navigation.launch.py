from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Define the path to the YAML parameter file
    navigation_param_path = os.path.join(
        get_package_share_directory('navigation_runner'),
        'cfg',
        'navigation_param.yaml'
    )

    # Create the node with the parameter file
    safe_navigation_node = Node(
        package='navigation_runner',
        executable='navigation_node.py',
        # name='navigation_node',
        output='screen',
        parameters=[navigation_param_path]
    )

    return LaunchDescription([
        safe_navigation_node
    ])