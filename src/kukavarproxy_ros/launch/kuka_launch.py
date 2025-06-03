#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1) Launch your standalone publisher script
        ExecuteProcess(
            cmd=[
                'python3',
                '/home/dev/ros2_ws/Stijn/recorder.py'  # or wherever you placed it
            ],
            output='screen'
        ),

        # 2) Launch the kuka_control node
        Node(
            package='kukavarproxy_ros',
            executable='kuka_control',
            name='kuka_control',
            output='screen'
        ),

        # 3) Launch the kuka_commander node
        Node(
            package='kukavarproxy_ros',
            executable='kuka_commander',
            name='kuka_commander',
            output='screen'
        ),
    ])
