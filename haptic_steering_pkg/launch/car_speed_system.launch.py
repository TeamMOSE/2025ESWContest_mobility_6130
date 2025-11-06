#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Launch arguments
    qt_platform_arg = DeclareLaunchArgument(
        'qt_platform',
        default_value='xcb',
        description='Qt platform plugin (xcb for X11, wayland for Wayland)'
    )
    
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM0',
        description='Serial port for handle.ino communication'
    )
    
    baud_rate_arg = DeclareLaunchArgument(
        'baud_rate',
        default_value='115200',
        description='Serial baud rate'
    )
    
    return LaunchDescription([
        qt_platform_arg,
        serial_port_arg,
        baud_rate_arg,
        
        # Qt 플랫폼 환경 변수 설정
        SetEnvironmentVariable(
            'QT_QPA_PLATFORM',
            LaunchConfiguration('qt_platform')
        ),
        
        # 1. Handle Serial Reader Node (아두이노에서 시리얼 데이터 읽기)
        Node(
            package='haptic_steering_pkg',
            executable='handle_serial_reader_node',
            name='handle_serial_reader_node',
            output='screen',
            parameters=[{
                'serial_port': LaunchConfiguration('serial_port'),
                'baud_rate': LaunchConfiguration('baud_rate'),
                'publish_rate': 10.0
            }]
        ),
        
        # 2. Car Speed Map Node (raw 값을 0~60 km/h로 변환)
        Node(
            package='haptic_steering_pkg',
            executable='car_speed_map_node',
            name='car_speed_map_node',
            output='screen',
            parameters=[{
                'idle_max': 360.0,
                'min_raw': 360.0,
                'max_raw': 735.0,
                'max_speed': 60.0
            }]
        ),
        
        # 3. Emergency HUD Node (속도 게이지 표시)
        Node(
            package='emergency_hud',
            executable='emergency_hud_node',
            name='emergency_hud',
            output='screen',
            parameters=[{
                'use_sim_time': False
            }]
        )
    ])

