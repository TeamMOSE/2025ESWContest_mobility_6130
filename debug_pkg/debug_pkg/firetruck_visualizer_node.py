import cv2
import random
import numpy as np
from typing import Tuple, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.duration import Duration

import message_filters
from cv_bridge import CvBridge

# Ultralytics import는 지연 로드 (노드 생성 후)
try:
    from ultralytics.utils.plotting import Annotator, colors
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from sensor_msgs.msg import Image, CompressedImage
from visualization_msgs.msg import Marker, MarkerArray

from interfaces_pkg.msg import BoundingBox2D, KeyPoint2D, KeyPoint3D, Detection, DetectionArray


class FiretruckVisualizerNode(Node):

    def __init__(self) -> None:
        super().__init__("firetruck_visualizer_node")

        # color cache: class_name -> (b,g,r)
        self._class_to_color = {}
        self.cv_bridge = CvBridge()

        # params
        # topics
        self.declare_parameter("image_topic", "back_camera/compressed")
        self.declare_parameter("detections_topic", "detections")

        # reliability for image subscriber
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        # filtering
        self.declare_parameter("allowed_names", ["fire truck", "fire_truck", "firetruck", "truck", "car"])
        self.declare_parameter("allowed_ids", [])
        self.declare_parameter("min_confidence", 0.25)

        # optional ROI to reduce 오탐 (정수 픽셀 좌표: [xmin, ymin, xmax, ymax])
        self.declare_parameter("roi", [])  # [] or 4개 정수

        # output encoding control (rgb8 or bgr8)
        self.declare_parameter("output_encoding", "rgb8")

        self.get_logger().info("🔥 Firetruck visualizer node created")
        
        # 파라미터 가져오기
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        self.allowed_names: List[str] = list(self.get_parameter("allowed_names").value)
        self.allowed_ids: List[int] = list(self.get_parameter("allowed_ids").value)
        self.min_conf = float(self.get_parameter("min_confidence").value)
        self.roi = list(self.get_parameter("roi").value)  # [] or [xmin,ymin,xmax,ymax]
        self.output_enc = (self.get_parameter("output_encoding").get_parameter_value().string_value or "rgb8").lower()
        if self.output_enc not in ("rgb8", "bgr8", "rgba8", "bgra8"):
            self.get_logger().warn(f"Invalid output_encoding '{self.output_enc}', falling back to 'rgb8'")
            self.output_enc = "rgb8"

        self.image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=3
        )

        # publishers
        self._img_pub = self.create_publisher(Image, "firetruck_visualized_img", 10)
        self._bb_pub = self.create_publisher(MarkerArray, "dbg_firetruck_bb_markers", 10)
        self._kp_pub = self.create_publisher(MarkerArray, "dbg_firetruck_kp_markers", 10)
        
        # subscribers + time sync (CompressedImage 구독)
        self.image_sub = message_filters.Subscriber(self, CompressedImage, self.image_topic, qos_profile=self.image_qos_profile)
        self.det_sub = message_filters.Subscriber(self, DetectionArray, self.detections_topic, qos_profile=10)
        
        self.get_logger().info(f"✅ Subscribing to {self.image_topic} with BEST_EFFORT QoS")
        self.get_logger().info(f"✅ Subscribing to {self.detections_topic}")

        self._sync = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.det_sub), queue_size=5, slop=0.1
        )
        self._sync.registerCallback(self.cb_synced)
        self.get_logger().info(f"✅ Output image will be published to firetruck_visualized_img as {self.output_enc}")
        self.get_logger().info(f"✅ Firetruck visualizer ready (filtering: {self.allowed_names})")

    # ------------ helpers ------------
    def _in_roi(self, box: BoundingBox2D, img_w: int, img_h: int) -> bool:
        """Check if bbox center is inside ROI (if ROI is set)."""
        if len(self.roi) != 4:
            return True
        xmin, ymin, xmax, ymax = self.roi
        cx = int(round(box.center.x))
        cy = int(round(box.center.y))
        # clamp ROI to image bounds
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(img_w - 1, xmax); ymax = min(img_h - 1, ymax)
        return (xmin <= cx <= xmax) and (ymin <= cy <= ymax)

    def _is_firetruck(self, det: Detection) -> bool:
        if det.score < self.min_conf:
            return False
        # ID 우선
        if self.allowed_ids:
            try:
                return int(det.class_id) in self.allowed_ids
            except Exception:
                pass
        # 이름 필터
        name = (det.class_name or "").strip().lower()
        return name in {n.lower() for n in self.allowed_names}

    # (removed: _prepare_image_for_rviz - now unused)

    # ------------ drawing ------------
    def draw_box(self, cv_image: np.ndarray, detection: Detection, color: Tuple[int, int, int]) -> np.ndarray:
        box: BoundingBox2D = detection.bbox
        min_pt = (round(box.center.x - box.size.x / 2.0),
                  round(box.center.y - box.size.y / 2.0))
        max_pt = (round(box.center.x + box.size.x / 2.0),
                  round(box.center.y + box.size.y / 2.0))
        cv2.rectangle(cv_image, min_pt, max_pt, color, 2)
        label = f"{detection.class_name or 'truck'} ({detection.score:.2f})"
        cv2.putText(cv_image, label, (min_pt[0] + 5, min_pt[1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
        return cv_image

    def draw_mask(self, cv_image: np.ndarray, detection: Detection, color: Tuple[int, int, int]) -> np.ndarray:
        mask_msg = detection.mask
        if not mask_msg.data:
            return cv_image
        mask_array = np.array([[int(p.x), int(p.y)] for p in mask_msg.data])
        layer = cv_image.copy()
        layer = cv2.fillPoly(layer, pts=[mask_array], color=color)
        cv2.addWeighted(cv_image, 0.4, layer, 0.6, 0, cv_image)
        cv_image = cv2.polylines(cv_image, [mask_array], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        return cv_image

    def draw_keypoints(self, cv_image: np.ndarray, detection: Detection) -> np.ndarray:
        kp_msg = detection.keypoints
        if not kp_msg.data or not ULTRALYTICS_AVAILABLE:
            return cv_image
        try:
            ann = Annotator(cv_image)
            for kp in kp_msg.data:
                color_k = [int(x) for x in ann.kpt_color[kp.id - 1]] if len(kp_msg.data) == 17 else colors(kp.id - 1)
                cv2.circle(cv_image, (int(kp.point.x), int(kp.point.y)), 5, color_k, -1, lineType=cv2.LINE_AA)

            def get_xy(kid: int) -> Optional[Tuple[int, int]]:
                for k in kp_msg.data:
                    if k.id == kid:
                        return (int(k.point.x), int(k.point.y))
                return None

            for i, sk in enumerate(ann.skeleton):
                p1 = get_xy(sk[0]); p2 = get_xy(sk[1])
                if p1 and p2:
                    cv2.line(cv_image, p1, p2, [int(x) for x in ann.limb_color[i]], 2, lineType=cv2.LINE_AA)
        except Exception as e:
            self.get_logger().debug(f"Keypoints drawing error: {e}")
        return cv_image

    def create_bb_marker(self, detection: Detection, color: Tuple[int, int, int], ns: str = "firetruck") -> Marker:
        b3 = detection.bbox3d
        frame_id = getattr(b3, "frame_id", "") or (b3.header.frame_id if hasattr(b3, "header") else "")
        m = Marker()
        m.header.frame_id = frame_id
        m.ns = ns
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.frame_locked = False

        m.pose.position.x = b3.center.position.x
        m.pose.position.y = b3.center.position.y
        m.pose.position.z = b3.center.position.z
        m.pose.orientation.w = 1.0

        m.scale.x = b3.size.x; m.scale.y = b3.size.y; m.scale.z = b3.size.z

        # NOTE: Marker 색상은 rgba (여기선 BGR을 RGBA로 매핑)
        m.color.b = color[0] / 255.0
        m.color.g = color[1] / 255.0
        m.color.r = color[2] / 255.0
        m.color.a = 0.4

        m.lifetime = Duration(seconds=0.5).to_msg()
        m.text = detection.class_name
        return m

    def create_kp_marker(self, keypoint: KeyPoint3D, ns: str = "firetruck") -> Marker:
        m = Marker()
        m.ns = ns
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.frame_locked = False
        m.pose.position.x = keypoint.point.x
        m.pose.position.y = keypoint.point.y
        m.pose.position.z = keypoint.point.z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.r = (1.0 - keypoint.score) * 255.0 / 255.0
        m.color.g = 0.0
        m.color.b = keypoint.score * 255.0 / 255.0
        m.color.a = 0.4
        m.lifetime = Duration(seconds=0.5).to_msg()
        m.text = str(keypoint.id)
        return m

    # ------------ main callback ------------
    def cb_synced(self, img_msg: CompressedImage, det_msg: DetectionArray) -> None:
        # CompressedImage 압축 해제
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # One-time debug log
        if not hasattr(self, '_enc_logged'):
            self.get_logger().info(f"[firetruck_visualizer] Receiving compressed images (JPEG)")
            self._enc_logged = True

        h, w = cv_img.shape[:2]

        bb_arr = MarkerArray()
        kp_arr = MarkerArray()

        # draw ROI box (optional 시각화용)
        if len(self.roi) == 4:
            x1, y1, x2, y2 = self.roi
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 255), 1)

        for det in det_msg.detections:
            if not self._is_firetruck(det):
                continue
            if not self._in_roi(det.bbox, w, h):
                continue

            # class color caching
            label = det.class_name or "truck"
            if label not in self._class_to_color:
                self._class_to_color[label] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
            color = self._class_to_color[label]

            # overlay
            cv_img = self.draw_box(cv_img, det, color)
            cv_img = self.draw_mask(cv_img, det, color)
            cv_img = self.draw_keypoints(cv_img, det)

            # 3D marker (있을 때만)
            b3 = det.bbox3d
            b3_frame_id = getattr(b3, "frame_id", "") or (b3.header.frame_id if hasattr(b3, "header") else "")
            if b3_frame_id:
                m = self.create_bb_marker(det, color)
                m.header.stamp = img_msg.header.stamp
                m.header.frame_id = b3_frame_id
                m.id = len(bb_arr.markers)
                bb_arr.markers.append(m)

            k3 = det.keypoints3d
            k3_frame_id = getattr(k3, "frame_id", "") or (k3.header.frame_id if hasattr(k3, "header") else "")
            if k3_frame_id:
                for kp in k3.data:
                    m = self.create_kp_marker(kp)
                    m.header.frame_id = k3_frame_id
                    m.header.stamp = img_msg.header.stamp
                    m.id = len(kp_arr.markers)
                    kp_arr.markers.append(m)

        # Publish using the requested output encoding
        pub_img = cv_img
        pub_enc = self.output_enc
        if self.output_enc == 'rgb8':
            if cv_img.ndim == 3 and cv_img.shape[2] == 3:
                pub_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            elif cv_img.ndim == 3 and cv_img.shape[2] == 4:
                pub_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
                pub_enc = 'rgba8'
        elif self.output_enc == 'bgr8':
            # already BGR from cv_bridge; keep as-is
            pub_enc = 'bgr8'
        elif self.output_enc == 'rgba8':
            if cv_img.ndim == 3 and cv_img.shape[2] == 3:
                pub_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGBA)
            else:
                pub_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
            pub_enc = 'rgba8'
        elif self.output_enc == 'bgra8':
            if cv_img.ndim == 3 and cv_img.shape[2] == 3:
                pub_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA)
            else:
                pub_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGRA)
            pub_enc = 'bgra8'
        else:
            pub_enc = 'rgb8'
            pub_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        self._img_pub.publish(self.cv_bridge.cv2_to_imgmsg(pub_img, encoding=pub_enc))
        self._bb_pub.publish(bb_arr)
        self._kp_pub.publish(kp_arr)


def main():
    rclpy.init()
    node = FiretruckVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
