#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Launch arguments
    qt_platform_arg = DeclareLaunchArgument(
        'qt_platform',
        default_value='xcb',
        description='Qt platform plugin (xcb for X11, wayland for Wayland)'
    )
    
    return LaunchDescription([
        qt_platform_arg,
        Node(
            package='emergency_hud',
            executable='emergency_hud_node',
            name='emergency_hud',
            output='screen',
            parameters=[{
                'use_sim_time': False
            }],
            # Wayland 환경 변수 설정
            # 투명 배경이 안될 경우 아래 옵션들을 변경해보세요:
            # 1. QT_QPA_PLATFORM=xcb (X11 사용, 가장 안정적)
            # 2. QT_QPA_PLATFORM=wayland (Wayland 네이티브)
            environment={
                'QT_QPA_PLATFORM': LaunchConfiguration('qt_platform'),
                # 'QT_WAYLAND_DISABLE_WINDOWDECORATION': '1',  # Wayland 사용 시
            }
        )
    ])

