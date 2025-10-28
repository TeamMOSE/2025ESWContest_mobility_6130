from typing import List, Dict, Set

import rclpy
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState

from cv_bridge import CvBridge
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes, Masks, Keypoints
from torch import cuda

from sensor_msgs.msg import CompressedImage
from interfaces_pkg.msg import Point2D, BoundingBox2D, Mask, KeyPoint2D, KeyPoint2DArray, Detection, DetectionArray
from std_srvs.srv import SetBool
from std_msgs.msg import String
import numpy as np
import cv2


class YoloFiretruckNode(LifecycleNode):
    def __init__(self, **kwargs) -> None:
        super().__init__("yolo_firetruck_node", **kwargs)

        # -------- Parameters (구조 동일) --------
        # 모델 경로는 고정 (하드코딩)
        self.model = "/home/mose/ros2_esw/src/camera_pkg/models/best_firetruck.pt"
        self.declare_parameter("device", "cpu")  # "cuda:0" 가능
        self.declare_parameter("threshold", 0.7)
        self.declare_parameter("enable", True)
        # self.declare_parameter("image_reliability", QoSReliabilityPolicy.RELIABLE)

        # 구독할 카메라 토픽 (기본: back_camera)
        self.declare_parameter("image_topic", "back_camera")

        # 타깃 클래스 필터(이름/ID) — 기본은 firetruck, car만 허용(정확한 이름 매칭)
        # (커스텀 가중치에서는 'fire truck' 클래스를 직접 지정 가능)
        self.declare_parameter("target_names", ["firetruck", "car"])
        self.declare_parameter("target_ids", [])  # 예: [0, 1, 2]  (커스텀 모델 사용 시)
        self.declare_parameter("side_deadband_frac", 0.10)  # 이미지 폭 대비 무감대 비율

        self.get_logger().info("YoloFiretruckNode created")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Configuring {self.get_name()}")

        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        # self.reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value

        self.side_deadband_frac = self.get_parameter("side_deadband_frac").get_parameter_value().double_value
        self._side_pub = self.create_lifecycle_publisher(String, "firetruck_side", 10)

        # 파라미터 배열 취득
        self.target_names: Set[str] = set(
            [s.lower() for s in self.get_parameter("target_names").get_parameter_value().string_array_value]
        )
        self.target_ids_param = list(self.get_parameter("target_ids").get_parameter_value().integer_array_value)
        self.model_derived_ids: List[int] = []
        self.firetruck_ids: List[int] = []

        # QoS: 무선 통신 최적화 (BEST_EFFORT)
        self.image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self._pub = self.create_lifecycle_publisher(DetectionArray, "detections", 10)
        self._srv = self.create_service(SetBool, "enable", self.enable_cb)
        self.cv_bridge = CvBridge()

        return TransitionCallbackReturn.SUCCESS

    def enable_cb(self, request, response):
        self.enable = request.data
        response.success = True
        return response

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Activating {self.get_name()}")

        try:
            self.yolo = YOLO(self.model)
            self.yolo.fuse()
            # Derive target class IDs from the model names (exact match)
            try:
                if hasattr(self.yolo, "names") and isinstance(self.yolo.names, (dict, list)):
                    names_dict = self.yolo.names if isinstance(self.yolo.names, dict) else {i: n for i, n in enumerate(self.yolo.names)}
                    # Resolve class IDs for all target names (exact lowercased match; no underscore/space normalization needed per user spec)
                    lower_target_names = {s.lower() for s in self.target_names}
                    self.model_derived_ids = [int(cid) for cid, n in names_dict.items() if str(n).lower() in lower_target_names]
                    # Fallback: if the custom model has a single class, assume it is the (only) target
                    if not self.model_derived_ids and len(names_dict) == 1:
                        self.model_derived_ids = [int(list(names_dict.keys())[0])]
                    self.get_logger().info(f"Target class IDs resolved: {self.model_derived_ids}")
                    # Keep a dedicated list of firetruck IDs for side decision (case-insensitive exact match)
                    self.firetruck_ids = [int(cid) for cid, n in names_dict.items() if str(n).lower() == "firetruck"]
            except Exception as e:
                self.get_logger().warn(f"Could not resolve target class IDs from model names: {e}")
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' not found!")
            return TransitionCallbackReturn.FAILURE
        except Exception as e:
            self.get_logger().error(f"Error while loading model '{self.model}': {str(e)}")
            return TransitionCallbackReturn.FAILURE

        # CompressedImage 구독 (압축 이미지)
        self._sub = self.create_subscription(
            CompressedImage,
            self.image_topic + '/compressed',  # back_camera/compressed
            self.image_cb,
            self.image_qos_profile
        )
        self.get_logger().info(f"Subscribing to {self.image_topic}/compressed with BEST_EFFORT QoS")

        super().on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Deactivating {self.get_name()}")

        del self.yolo
        if 'cuda' in self.device:
            self.get_logger().info("Clearing CUDA cache")
            cuda.empty_cache()

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Cleaning up {self.get_name()}")
        self.destroy_publisher(self._pub)
        self.destroy_publisher(self._side_pub)
        del self.image_qos_profile
        return TransitionCallbackReturn.SUCCESS

    def parse_hypothesis(self, results: Results) -> List[Dict]:
        out = []
        for box_data in results.boxes or []:
            cls_id = int(box_data.cls)
            name = self.yolo.names.get(cls_id, str(cls_id)) if hasattr(self.yolo, "names") else str(cls_id)
            out.append({
                "class_id": cls_id,
                "class_name": name,
                "score": float(box_data.conf)
            })
        return out

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        boxes_list: List[BoundingBox2D] = []
        for box_data in results.boxes or []:
            msg = BoundingBox2D()
            xywh = box_data.xywh[0]
            # center is Pose2D (x, y, theta), not geometry_msgs/Pose
            msg.center.x = float(xywh[0])
            msg.center.y = float(xywh[1])
            msg.center.theta = 0.0
            # size is Vector2 (x, y)
            msg.size.x = float(xywh[2])
            msg.size.y = float(xywh[3])
            boxes_list.append(msg)
        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:
        masks_list: List[Mask] = []

        def p2d(x: float, y: float) -> Point2D:
            p = Point2D(); p.x = x; p.y = y; return p

        for mask in results.masks or []:
            msg = Mask()
            msg.data = [p2d(float(ele[0]), float(ele[1])) for ele in mask.xy[0].tolist()]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]
            masks_list.append(msg)
        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
        out: List[KeyPoint2DArray] = []
        for points in results.keypoints or []:
            if points.conf is None:
                continue
            arr = KeyPoint2DArray()
            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):
                if float(conf) >= self.threshold:
                    kp = KeyPoint2D()
                    kp.id = kp_id + 1
                    kp.point.x = float(p[0])
                    kp.point.y = float(p[1])
                    kp.score = float(conf)
                    arr.data.append(kp)
            out.append(arr)
        return out

    # Filtering priority: param target_ids > model-derived IDs (from model names) > strict name match
    def _is_target(self, cls_id: int, cls_name: str) -> bool:
        # 1) Explicit override via parameter
        if self.target_ids_param:
            return cls_id in self.target_ids_param
        # 2) Model-derived IDs (from self.yolo.names)
        if getattr(self, "model_derived_ids", []):
            return cls_id in self.model_derived_ids
        # 3) Fallback: exact name match against target_names (lowercased)
        if cls_name:
            return cls_name.lower() in {s.lower() for s in self.target_names}
        return False

    def _pick_representative_index(self, hypothesis) -> int:
        """
        대표 firetruck 하나의 인덱스를 고른다. 기준: score 최댓값
        """
        if not hypothesis:
            return -1
        best_i, best_val = -1, -1.0
        for i, det in enumerate(hypothesis):
            if det["score"] > best_val:
                best_val = det["score"]
                best_i = i
        return best_i

    def image_cb(self, msg: CompressedImage) -> None:
        if not self.enable:
            return

        # CompressedImage 압축 해제
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        results_list = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            device=self.device
        )
        results: Results = results_list[0].cpu()

        detections_msg = DetectionArray()
        log_entries = []
        detections_msg.header = msg.header

        # 박스/마스크/키포인트 사전 파싱
        hypothesis = self.parse_hypothesis(results) if results.boxes else []
        boxes = self.parse_boxes(results) if results.boxes else []
        masks = self.parse_masks(results) if results.masks else []
        keypoints = self.parse_keypoints(results) if results.keypoints else []

        # 안전 루프: 박스 수 기준
        num = len(hypothesis)

        # 검출/퍼블리시는 car+firetruck 모두, **좌우 판정 후보는 firetruck만** 수집
        tg_hypothesis = []
        tg_boxes = []

        for i in range(num):
            cls_id = hypothesis[i]["class_id"]
            cls_name = hypothesis[i]["class_name"]

            # ---- 타깃(예: firetruck, car)만 통과 ----
            if not self._is_target(cls_id, cls_name):
                continue

            det = Detection()
            det.class_id = cls_id
            det.class_name = cls_name
            det.score = hypothesis[i]["score"]
            det.bbox = boxes[i]  # boxes와 hypothesis 인덱스 정렬 보장(둘 다 boxes 기준 생성)

            # 선택적으로 같은 인덱스의 마스크/키포인트를 붙일 수 있지만,
            # 수가 다를 수 있으므로 존재 범위를 체크
            if i < len(masks):
                det.mask = masks[i]
            if i < len(keypoints):
                det.keypoints = keypoints[i]

            detections_msg.detections.append(det)
            log_entries.append(f"{det.class_name} ({det.score:.2f})")

            # 좌/우 판정 후보는 **firetruck만** 수집
            if (self.firetruck_ids and cls_id in self.firetruck_ids) or (cls_name and cls_name.lower() == "firetruck"):
                tg_hypothesis.append(hypothesis[i])
                tg_boxes.append(boxes[i])

        if log_entries:
            self.get_logger().info("Detections: " + ", ".join(log_entries))

        # --- 좌/우/없음 판정 및 퍼블리시 ---
        side_msg = String()
        side_msg.data = "None"

        # 이미지 폭
        W = results.orig_img.shape[1] if hasattr(results, "orig_img") else None

        if tg_hypothesis and tg_boxes and W:
            rep_idx = self._pick_representative_index(tg_hypothesis)
            if rep_idx >= 0:
                cx = tg_boxes[rep_idx].center.x
                mid = W * 0.5

                # 좌우 표기 스왑: (기존 Left↔Right)
                if cx < mid:
                    side_msg.data = "Rear_Right"
                elif cx > mid:
                    side_msg.data = "Rear_Left"
                else:
                    # 정확히 중앙이면 임의로 Rear_Right로 분류 (결정적 동작)
                    side_msg.data = "Rear_Right"

        self.get_logger().info(f"Firetruck side decision: {side_msg.data}")

        # firetruck_side 토픽 퍼블리시 (검출 없으면 "None")
        self._side_pub.publish(side_msg)

        # 퍼블리시 (비어 있어도 헤더 포함해 발행)
        self._pub.publish(detections_msg)

        del results
        del cv_image


def main():
    rclpy.init()
    node = YoloFiretruckNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()