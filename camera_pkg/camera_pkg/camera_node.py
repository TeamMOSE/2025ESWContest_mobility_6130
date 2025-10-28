import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

from rcl_interfaces.msg import ParameterDescriptor

class RearCameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # ---------------- Params ----------------
        self.declare_parameter('camera_id', '/dev/video0', ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('topic_name', '', ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('frame_id', 'back_camera_frame')
        self.declare_parameter('show_image', False)            # 디버그용 창 표시 여부
        self.declare_parameter('jpeg_quality', 85)             # JPEG 압축 품질 (무선 통신 최적화)
        # ----------------------------------------

        self.camera_id = self.get_parameter('camera_id').value
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)
        self.topic_name = str(self.get_parameter('topic_name').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.show_image = bool(self.get_parameter('show_image').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        # QoS: 무선 통신 최적화 (BEST_EFFORT)
        self.qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # 속도 우선
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.bridge = CvBridge()

        # Open camera
        cam_path = f"/dev/video{self.camera_id}" if isinstance(self.camera_id, int) else str(self.camera_id)
        self.cap = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
             self.get_logger().error(f'Cannot open camera device: {cam_path}')

        # Prefer MJPG over raw YUYV for better throughput on many UVC webcams
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass

        # Set properties (best-effort; some cameras ignore)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Publisher & Timer (CompressedImage로 변경)
        self.pub = self.create_publisher(CompressedImage, self.topic_name + '/compressed', self.qos_profile)
        period = 1.0 / self.fps if self.fps > 0 else 0.033  # fallback ~30fps
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f'camera_node started: camera_id={self.camera_id}, '
            f'resolution={self.width}x{self.height}@{self.fps}fps, '
            f'topic={self.topic_name}/compressed, frame_id={self.frame_id}'
        )
        self.get_logger().info(f'무선 통신 최적화: JPEG 압축(품질={self.jpeg_quality}), QoS=BEST_EFFORT')

    def _on_timer(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn('Failed to read frame from camera')
            return

        # 일부 카메라가 요청 해상도를 무시하므로 안전하게 맞춤
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        # JPEG 압축 (무선 통신 최적화)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, compressed_data = cv2.imencode('.jpg', frame, encode_param)

        # CompressedImage 메시지 생성
        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.format = "jpeg"
        msg.data = compressed_data.tobytes()
        
        self.pub.publish(msg)

        if self.show_image:
            cv2.imshow('camera', frame)
            cv2.waitKey(1)

    def destroy_node(self):
        try:
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RearCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()