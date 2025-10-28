from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='siren_pkg',
            executable='siren_node',
            name='siren_node',
            output='screen'
        )
        
        

        

    ])