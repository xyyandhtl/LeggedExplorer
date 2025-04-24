from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Define the path to the YAML parameter file
    map_param_path = os.path.join(
        get_package_share_directory('navigation_runner'),
        'cfg',
        'map_param.yaml'
    )

    dynamic_detector_param_path = os.path.join(
        get_package_share_directory('navigation_runner'),
        'cfg',
        'dynamic_detector_param.yaml'
    )

    yolo_detector_param_path = os.path.join(
        get_package_share_directory('navigation_runner'),
        'cfg',
        'yolo_detector_param.yaml'
    )

    occupancy_map_node = Node(
        package='map_manager',
        executable='occupancy_map_node',
        name='map_manager_node',
        output='screen',
        parameters=[map_param_path]
    )

    dynamic_detector_node = Node(
        package='onboard_detector',
        executable='dynamic_detector_node',
        name='dynamic_detector_node',
        output='screen',
        parameters=[dynamic_detector_param_path]
    )

    yolo_detector_node = Node(
        package='onboard_detector',
        executable='yolo_detector_node.py',  
        name='yolo_detector_node',
        output='screen',
        parameters=[yolo_detector_param_path]
    )

    return LaunchDescription([
        occupancy_map_node,
        dynamic_detector_node,
        yolo_detector_node,
    ])