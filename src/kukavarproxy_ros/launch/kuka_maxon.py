import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.conditions import (
    IfCondition,
)  # Importing IfCondition for conditional launching
from ament_index_python.packages import get_package_share_directory
import os

pkg_share = get_package_share_directory("kukavarproxy_ros")
kuka_params = os.path.join(pkg_share, "config", "kuka_control.yaml")


def generate_launch_description():
    return LaunchDescription(
        [
            # Declare launch arguments for flexibility
            DeclareLaunchArgument(
                "kuka_control",
                default_value="True",
                description="Launch the kukavarproxy contrl node",
            ),
            DeclareLaunchArgument(
                "maxon_node",
                default_value="True",
                description="Launch the maxon node",
            ),
            # Launch the maxon node with IfCondition
            Node(
                package="maxon",
                executable="maxon_node",
                name="maxon",
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("maxon_node")
                ),  # Conditional launch based on serial_reader argument
                # arguments=["--ros-args", "--log-level", "INFO"],
            ),
            # Launch the kuka node with IfCondition
            Node(
                package="kukavarproxy_ros",
                executable="kuka_control",
                name="kuka_control",
                output="screen",
                parameters=[kuka_params],
                condition=IfCondition(
                    LaunchConfiguration("kuka_control")
                ),  # Conditional launch based on magnet_tracker argument
            ),
        ]
    )
