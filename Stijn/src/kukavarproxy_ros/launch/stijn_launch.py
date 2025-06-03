from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
def generate_launch_description():
    return LaunchDescription([
        # 1) Launch your standalone publisher script
        ExecuteProcess(
            cmd=[
                'python3',
                'Stijn/recorder.py' 
            ],
            output='screen'
        ),
        Node(
            package='kukavarproxy_ros',
            executable='kuka_commander',
            name='kuka_commander',
            output='screen'
        ),
    ])

