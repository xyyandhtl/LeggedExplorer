from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Define the path to the YAML parameter file
    dynamic_detector_param_path = os.path.join(
        get_package_share_directory('onboard_detector'),
        'cfg',
        'dynamic_detector_param.yaml'
    )

    yolo_detector_param_path = os.path.join(
        get_package_share_directory('onboard_detector'),
        'cfg',
        'yolo_detector_param.yaml'
    )

    # Create the node with the parameter file
    dynamic_detector_node = Node(
        package='onboard_detector',
        executable='dynamic_detector_node',
        name='dynamic_detector_node',
        output='screen',
        parameters=[dynamic_detector_param_path]
    )

    # Create the yolo node
    yolo_detector_node = Node(
        package='onboard_detector',
        executable='yolo_detector_node.py',  
        name='yolo_detector_node',      
        output='screen',
        parameters=[yolo_detector_param_path]
    )

    return LaunchDescription([
        dynamic_detector_node,
        yolo_detector_node
    ])