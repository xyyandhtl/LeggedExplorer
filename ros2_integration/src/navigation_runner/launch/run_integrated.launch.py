import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 声明 package_share_directory 变量，并提供默认值
    declare_package_share_directory = DeclareLaunchArgument(
        'package_share_directory',
        default_value=FindPackageShare("navigation_runner"),
        description='Path to the package share directory'
    )

    # 使用 PathJoinSubstitution 代替 os.path.join
    include_perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([LaunchConfiguration('package_share_directory'), 'launch', 'perception.launch.py'])
        )
    )

    include_safe_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([LaunchConfiguration('package_share_directory'), 'launch', 'safe_action.launch.py'])
        )
    )

    include_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([LaunchConfiguration('package_share_directory'), 'launch', 'rviz.launch.py'])
        )
    )

    include_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([LaunchConfiguration('package_share_directory'), 'launch', 'navigation.launch.py'])
        )
    )

    return LaunchDescription([
        declare_package_share_directory,
        include_perception,
        # include_safe_action,
        include_rviz,
        include_navigation
    ])
