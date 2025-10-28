from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'firetruck_timeout_sec',
            default_value='3.0',
            description='Timeout for firetruck detection validity in seconds'
        ),
        DeclareLaunchArgument(
            'control_frequency_hz',
            default_value='10.0', 
            description='Control loop frequency in Hz'
        ),
        DeclareLaunchArgument(
            'min_speed_for_action_kmh',
            default_value='5.0',
            description='Minimum vehicle speed for taking action in km/h'
        ),

        # 1. 후방 카메라 노드 (카메라 ID: 0)
        Node(
            package='rear_camera_pkg',
            executable='rear_camera_node',
            name='rear_camera_node',
            output='screen',
            parameters=[{
                'camera_id': 0,
                'width': 640,
                'height': 480,
                'fps': 30.0,
                'topic_name': 'rear_camera',
                'frame_id': 'rear_camera_frame',
                'show_image': False
            }]
        ),

        # 2. 전방 카메라 노드 (카메라 ID: 1)
        Node(
            package='rear_camera_pkg',
            executable='rear_camera_node',
            name='front_camera_node',
            output='screen',
            parameters=[{
                'camera_id': 1,
                'width': 640,
                'height': 480,
                'fps': 30.0,
                'topic_name': 'front_camera',
                'frame_id': 'front_camera_frame', 
                'show_image': False
            }]
        ),

        # 3. YOLO 후방 소방차 감지 노드
        Node(
            package='rear_camera_pkg',
            executable='yolo_firetruck_node',
            name='yolo_firetruck_rear',
            output='screen',
            parameters=[{
                'device': 'cpu',
                'threshold': 0.7,
                'enable': True,
                'image_topic': 'rear_camera',
                'side_deadband_frac': 0.10
            }]
        ),

        # 4. Lane Detector Node
        Node(
            package='lane_detector_pkg',
            executable='lane_detector_node',
            name='lane_detector_node',
            output='screen',
            parameters=[{
                'input_topic': 'front_camera',
                'overlay_topic': '/lane/overlay',
                'lane_id_topic': '/lane/info',
                'lane_angle_topic': '/lane/angle',
                'show_debug': False
            }]
        ),

        # 5. HUD Display Node
        Node(
            package='hud_ui_ros2',
            executable='hud_display_node',
            name='hud_display_node',
            output='screen',
            parameters=[{
                'display_mode': 'console',
                'alert_timeout_sec': 5.0,
            }]
        ),

                # 6. 햅틱 피드백 노드 (실제 하드웨어 제어)
        Node(
            package='haptic_steering_pkg',
            executable='haptic_feedback_node',
            name='haptic_feedback_node',
            output='screen',
            parameters=[{
                'left_intensity': 70,
                'right_intensity': 70,
                'stop_intensity': 100,
                'feedback_duration_ms': 500
            }]
        ),

        # 7. Integrated control node (메인 제어 로직)
        Node(
            package='integrated_control_pkg',
            executable='integrated_control_node',
            name='integrated_control_node',
            output='screen',
            parameters=[{
                'firetruck_timeout_sec': LaunchConfiguration('firetruck_timeout_sec'),
                'control_frequency_hz': LaunchConfiguration('control_frequency_hz'),
                'min_speed_for_action_kmh': LaunchConfiguration('min_speed_for_action_kmh'),
            }]
        ),
    ])