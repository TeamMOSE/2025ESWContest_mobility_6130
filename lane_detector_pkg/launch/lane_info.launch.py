from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # lane_state_switcher 노드
        Node(
            package='lane_detector_pkg',
            executable='lane_state_switcher',
            name='lane_state_switcher',
            output='screen',
        ),

        # lane_detector 노드
        Node(
            package='lane_detector_pkg',
            executable='lane_detector_node',
            name='lane_detector_node',
            output='screen',
        ),

    ])
