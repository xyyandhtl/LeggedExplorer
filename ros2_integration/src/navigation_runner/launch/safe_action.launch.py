from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Define the path to the YAML parameter file
    safe_action_param_path = os.path.join(
        get_package_share_directory('navigation_runner'),
        'cfg',
        'safe_action_param.yaml'
    )

    # Create the node with the parameter file
    safe_action_node = Node(
        package='navigation_runner',
        executable='safe_action_node',
        name='safe_action_node',
        output='screen',
        parameters=[safe_action_param_path]
    )

    return LaunchDescription([
        safe_action_node
    ])