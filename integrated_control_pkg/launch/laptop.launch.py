#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    노트북용 런치 파일
    - YOLO 감지 노드들 (소방차, 신호등)
    - 사이렌 처리 노드들
    - 차선 감지 노드
    - 햅틱 피드백 노드
    - 통합 제어 노드
    - HUD 노드
    - 디버그 시각화 노드
    """
    
    # Launch arguments
    firetruck_timeout_arg = DeclareLaunchArgument(
        'firetruck_timeout_sec',
        default_value='3.0',
        description='Timeout for firetruck detection validity in seconds'
    )
    
    control_frequency_arg = DeclareLaunchArgument(
        'control_frequency_hz',
        default_value='10.0', 
        description='Control loop frequency in Hz'
    )
    
    min_speed_arg = DeclareLaunchArgument(
        'min_speed_for_action_kmh',
        default_value='5.0',
        description='Minimum vehicle speed for taking action in km/h'
    )
    
    return LaunchDescription([
        # Launch arguments
        firetruck_timeout_arg,
        control_frequency_arg,
        min_speed_arg,
        
        # YOLO 후방 소방차 감지 노드
        Node(
            package='camera_pkg',
            executable='yolo_firetruck_node',
            name='yolo_firetruck_rear',
            output='screen',
        ),

        # YOLO 전방 신호등 감지 노드
        Node(
            package='camera_pkg',
            executable='yolo_trafficlight_node',
            name='yolo_trafficlight_front',
            output='screen',
        ),

        # YOLO 전방 차량 감지 노드
        Node(
            package='camera_pkg',
            executable='yolo_frontcar_node',
            name='yolo_frontcar',
            output='screen',
        ),

        # 사이렌 감지기 노드
        Node(
            package='siren_pkg',
            executable='siren_detector_node',
            name='siren_detector_node',
            output='screen'
        ),

        # 이미지 뷰어 노드
        Node(
            package='siren_pkg',
            executable='image_viewer_node',
            name='image_viewer_node',
            output='screen'
        ),

        # 차선 감지 노드
        Node(
            package='lane_detector_pkg',
            executable='lane_detector_node',
            name='lane_detector_node',
            output='screen',
        ),

        # 통합 제어 노드 (메인 제어 로직)
        Node(
            package='integrated_control_pkg',
            executable='integrated_control_node',
            name='integrated_control_node',
            output='screen',
        ),

        # 디버그 시각화 노드 (소방차 감지 오버레이)
        Node(
            package='debug_pkg',
            executable='firetruck_visualizer_node',
            name='firetruck_visualizer_node',
            output='screen'
        ),

        # 디버그 시각화 노드 (신호등 감지 오버레이)
        Node(
            package='debug_pkg',
            executable='trafficlight_visualizer_node',
            name='trafficlight_visualizer_node',
            output='screen'
        ),

        # 디버그 시각화 노드 (앞차량 감지 오버레이)
        Node(
            package='debug_pkg',
            executable='frontcar_visualizer_node',
            name='frontcar_visualizer_node',
            output='screen'
        ),
    ])

