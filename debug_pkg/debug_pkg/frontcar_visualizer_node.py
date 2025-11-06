from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32MultiArray
from interfaces_pkg.msg import DetectionArray, BoundingBox2D

from cv_bridge import CvBridge
import cv2


class FrontCarVisualizerNode(Node):
    def __init__(self):
        super().__init__("frontcar_visualizer_node")

        # ---- Parameters ----
        self.declare_parameter("image_topic", "front_camera/compressed")
        self.declare_parameter("bboxes_topic", "car_bboxes")
        self.declare_parameter("lane_info_topic", "/lane/info/yolo")
        self.declare_parameter("output_topic", "frontcar/visualized")

        self.declare_parameter("line_thickness", 2)
        self.declare_parameter("font_scale", 0.8)

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.bboxes_topic = self.get_parameter("bboxes_topic").get_parameter_value().string_value
        self.lane_info_topic = self.get_parameter("lane_info_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        self.thickness = int(self.get_parameter("line_thickness").get_parameter_value().integer_value)
        self.font_scale = float(self.get_parameter("font_scale").get_parameter_value().double_value)

        # QoS: 최신 프레임 우선
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=3
        )

        self.bridge = CvBridge()

        # Latest caches
        self._last_dets: Optional[DetectionArray] = None
        self._last_lane: Optional[Tuple[int, int]] = None  # (total_lanes, my_lane)

        # Subs / Pub
        self._sub_img = self.create_subscription(CompressedImage, self.image_topic, self._on_image, sensor_qos)
        self._sub_det = self.create_subscription(DetectionArray, self.bboxes_topic, self._on_dets, sensor_qos)
        self._sub_lane = self.create_subscription(Int32MultiArray, self.lane_info_topic, self._on_lane, sensor_qos)
        self._pub_img = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(
            f"FrontCarVisualizerNode started: img='{self.image_topic}', dets='{self.bboxes_topic}', "
            f"lane='{self.lane_info_topic}' -> out='{self.output_topic}'"
        )

    def _on_dets(self, msg: DetectionArray):
        self._last_dets = msg

    def _on_lane(self, msg: Int32MultiArray):
        # 기대 포맷: [total_lanes, my_lane]
        if len(msg.data) >= 2:
            self._last_lane = (int(msg.data[0]), int(msg.data[1]))
        else:
            self._last_lane = None

    def _on_image(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().warn(f"cv_bridge conversion failed: {e}")
            return

        h, w = frame.shape[:2]

        # 그리기용 복사본
        img = frame.copy()

        # 2) bbox 시각화
        if self._last_dets is not None:
            self._draw_bboxes(img, self._last_dets.detections, color=(0, 255, 0))

        # 3) 우상단 텍스트 요약
        self._draw_hud(img, w, h)

        # Publish
        try:
            out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            out_msg.header = msg.header  # 입력 이미지와 같은 timestamp/frame_id 사용
            self._pub_img.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f"cv_bridge publish failed: {e}")

    # ---------- Drawing helpers ----------
    def _draw_bboxes(self, img, detections, color=(0, 255, 0)):
        """DetectionArray의 bbox를 사각형으로 그린다."""
        for det in detections:
            # interfaces_pkg/BoundingBox2D: center(x,y), size(w,h)
            bbox: BoundingBox2D = det.bbox
            cx, cy = float(bbox.center.x), float(bbox.center.y)
            bw, bh = float(bbox.size.x), float(bbox.size.y)
            x1 = int(max(0, cx - bw * 0.5))
            y1 = int(max(0, cy - bh * 0.5))
            x2 = int(min(img.shape[1] - 1, cx + bw * 0.5))
            y2 = int(min(img.shape[0] - 1, cy + bh * 0.5))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=self.thickness)

            label = getattr(det, "class_name", "cars")
            cv2.putText(
                img, f"{label}",
                (x1, max(0, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, max(1, self.thickness - 1), cv2.LINE_AA
            )

    def _draw_hud(self, img, w, h):
        """우상단 HUD 텍스트: '전체차로,현재차로' 형식으로 표시."""
        if self._last_lane is not None:
            total_lanes, my_lane = self._last_lane
            text = f"{total_lanes},{my_lane}"
        else:
            text = "-,-"
        # 우상단 위치 계산
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2)
        x = max(10, w - 10 - tw)
        y = max(10 + th, 10 + th)  # 상단 여백 10px
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 255), 2, cv2.LINE_AA)


def main():
    rclpy.init()
    node = FrontCarVisualizerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
