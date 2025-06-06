from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
def generate_launch_description():
    return LaunchDescription([
        # 1) Launch your standalone publisher script
        ExecuteProcess(
            cmd=[
                'python3',
                'src/python/recorder.py' 
            ],
            output='screen'
        ),
        Node(
            package='kuka_python_node',
            executable='kuka_commander',
            name='kuka_commander',
            output='screen'
        ),
    ])

