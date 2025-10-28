from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 후방 카메라 노드 (카메라 ID: 0)
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

        # 전방 카메라 노드 (카메라 ID: 1)
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

        # # YOLO 후방 소방차 감지 노드
        # Node(
        #     package='camera_pkg',
        #     executable='yolo_firetruck_node',
        #     name='yolo_firetruck_rear',
        #     output='screen',
        #     parameters=[{
        #         'device': 'cpu',
        #         'threshold': 0.7,
        #         'enable': True,
        #         'image_topic': 'back_camera',
        #         'side_deadband_frac': 0.10
        #     }]
        # ),

        # # YOLO 전방 소방차 감지 노드
        # Node(
        #     package='camera_pkg',
        #     executable='yolo_trafficlight_node',
        #     name='yolo_trafficlight_rear',
        #     output='screen',
        #     parameters=[{
        #         'device': 'cpu',
        #         'threshold': 0.7,
        #         'enable': True,
        #         'image_topic': 'front_camera',
        #         'side_deadband_frac': 0.10
        #     }]
        # ),
    ])