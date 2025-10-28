#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    라즈베리파이용 런치 파일
    - 카메라 노드 (전방/후방)
    - 사이렌 감지 노드
    """
    return LaunchDescription([
        # 후방 카메라 노드
        Node(
            package='camera_pkg',
            executable='camera_node',
            name='rear_camera_node',
            output='screen',
            parameters=[{
                'camera_id': '/dev/backcam',
                'width': 640,
                'height': 480,
                'fps': 30.0,
                'topic_name': 'back_camera',
                'frame_id': 'rear_camera_frame',
                'show_image': False
            }]
        ),

        # 전방 카메라 노드
        Node(
            package='camera_pkg',
            executable='camera_node',
            name='front_camera_node',
            output='screen',
            parameters=[{
                'camera_id': '/dev/frontcam',
                'width': 640,
                'height': 480,
                'fps': 30.0,
                'topic_name': 'front_camera',
                'frame_id': 'front_camera_frame', 
                'show_image': False
            }]
        ),

        # 사이렌 감지 노드
        Node(
            package='siren_pkg',
            executable='siren_node',
            name='siren_node',
            output='screen'
        ),
    ])

