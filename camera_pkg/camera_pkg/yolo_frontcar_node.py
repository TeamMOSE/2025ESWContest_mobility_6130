from typing import List, Dict, Set

import rclpy
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState

from cv_bridge import CvBridge
from ultralytics import YOLO
from ultralytics.engine.results import Results
from torch import cuda

from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from interfaces_pkg.msg import (
    BoundingBox2D,
    Detection, DetectionArray
)

class YoloFrontCarNode(LifecycleNode):
    def __init__(self, **kwargs) -> None:
        super().__init__("yolo_frontcar_node", **kwargs)

        # ---- Parameters ----
        # 모델 경로: 필요 시 변경
        self.declare_parameter("model", "/home/cyjung/ESW_ws/src/camera_pkg/models/best_cars.pt")
        self.declare_parameter("device", "cuda:0")          # "cuda:0" 가능
        self.declare_parameter("threshold", 0.6)
        self.declare_parameter("enable", True)

        # 전방 카메라 토픽
        self.declare_parameter("image_topic", "front_camera")

        # 타깃 클래스 이름/ID (단일 클래스 'cars' 전제)
        self.declare_parameter("target_names", ["cars"])
        self.declare_parameter("target_ids", [])  # 정수 ID 직접 지정 시 우선

        self.get_logger().info("YoloFrontCarNode created")

    # ---------- Lifecycle ----------
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Configuring {self.get_name()}")

        self.model_path = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value

        # 파라미터 배열 취득
        self.target_names: Set[str] = set(
            [s.lower() for s in self.get_parameter("target_names").get_parameter_value().string_array_value]
        )
        self.target_ids_param = list(self.get_parameter("target_ids").get_parameter_value().integer_array_value)
        self.model_derived_ids: List[int] = []

        # 카메라 QoS (최신 프레임 우선)
        self.image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 퍼블리셔 (시각화를 위한 bbox + 통합 차로 정보 퍼블리시)
        self._bboxes_pub = self.create_lifecycle_publisher(DetectionArray, "car_bboxes", 10)
        # 통합 퍼블리셔: [전체차로수, 현재주행차로] (Int32MultiArray)
        self._lane_info_pub = self.create_lifecycle_publisher(Int32MultiArray, "/lane/info/yolo", 10)
        self.cv_bridge = CvBridge()

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Activating {self.get_name()}")

        try:
            self.yolo = YOLO(self.model_path)
            self.yolo.fuse()

            # 모델 names로부터 타깃 클래스 ID 해석
            try:
                if hasattr(self.yolo, "names") and isinstance(self.yolo.names, (dict, list)):
                    names_dict = self.yolo.names if isinstance(self.yolo.names, dict) else {i: n for i, n in enumerate(self.yolo.names)}
                    def norm(s: str) -> str: return str(s).lower().replace("_", " ")
                    target_norm = {norm(s) for s in self.target_names}
                    self.model_derived_ids = [
                        int(cid) for cid, n in names_dict.items() if norm(n) in target_norm
                    ]
                    # 단일 클래스 모델일 경우 해당 ID 사용
                    if not self.model_derived_ids and len(names_dict) == 1:
                        self.model_derived_ids = [int(list(names_dict.keys())[0])]
                    self.get_logger().info(f"Target class IDs resolved: {self.model_derived_ids}")
            except Exception as e:
                self.get_logger().warn(f"Could not resolve target class IDs from model names: {e}")

        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model_path}' not found!")
            return TransitionCallbackReturn.FAILURE
        except Exception as e:
            self.get_logger().error(f"Error while loading model '{self.model_path}': {str(e)}")
            return TransitionCallbackReturn.FAILURE

        # 전방 카메라 구독
        self._sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            self.image_qos_profile
        )

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
        self.destroy_publisher(self._bboxes_pub)
        self.destroy_publisher(self._lane_info_pub)
        del self.image_qos_profile
        return TransitionCallbackReturn.SUCCESS

    # ---------- Parsing helpers ----------
    def parse_hypothesis(self, results: Results) -> List[Dict]:
        out = []
        # handle names as dict or list for compatibility
        names = getattr(self.yolo, "names", None)
        for box_data in results.boxes or []:
            # cls/conf may be 0-dim tensors; cast safely
            try:
                cls_id = int(box_data.cls)
            except Exception:
                # fallback for 1-element tensors/arrays
                cls_id = int(float(box_data.cls[0]))

            try:
                score = float(box_data.conf)
            except Exception:
                score = float(box_data.conf[0])

            # resolve class name
            if isinstance(names, dict):
                name = names.get(cls_id, str(cls_id))
            elif isinstance(names, list):
                name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
            else:
                name = str(cls_id)

            out.append({
                "class_id": cls_id,
                "class_name": name,
                "score": score
            })
        return out

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        boxes_list: List[BoundingBox2D] = []
        for box_data in results.boxes or []:
            msg = BoundingBox2D()
            xywh = box_data.xywh[0]
            msg.center.x = float(xywh[0])
            msg.center.y = float(xywh[1])
            msg.center.theta = 0.0
            msg.size.x = float(xywh[2])
            msg.size.y = float(xywh[3])
            boxes_list.append(msg)
        return boxes_list

    # 타깃 필터링: param IDs > model-derived IDs > 이름(느슨 매칭)
    def _is_target(self, cls_id: int, cls_name: str) -> bool:
        if self.target_ids_param:
            return cls_id in self.target_ids_param
        if getattr(self, "model_derived_ids", []):
            return cls_id in self.model_derived_ids
        if cls_name:
            name_l = cls_name.lower().replace("_", " ")
            return name_l in {s.replace("_", " ") for s in self.target_names}
        return False

    def estimate_my_lane_from_boxes(self, boxes: List[BoundingBox2D], img_w: int, img_h: int):
        """
        요청한 규칙 기반 휴리스틱:
        - 탐지된 차량 수 = 전체 차로 수
        - 2차선: 왼쪽 멀리 차 → 내 차로=2, 오른쪽 멀리 차 → 내 차로=1
        - 3차선: 양쪽 멀리 차가 보이면 가운데(2), 왼쪽만 멀리면 3, 오른쪽만 멀리면 1
        반환: (num_lanes, my_lane)  # 1=오른쪽, 증가할수록 왼쪽
        """
        num_lanes = len(boxes)

        if num_lanes <= 0:
            return 0, 0
        if num_lanes == 1:
            return 1, 1

        cx0 = img_w * 0.5
        margin = img_w * 0.05  # 중앙 5%는 center 취급

        def side_of(b: BoundingBox2D) -> str:
            x = b.center.x
            if x < cx0 - margin:
                return "L"
            if x > cx0 + margin:
                return "R"
            return "C"

        # 거리 점수: bbox 높이 작을수록 멀다
        def far_score(b: BoundingBox2D) -> float:
            h = max(b.size.y, 1e-6)
            return 1.0 / h

        left_candidates  = [(far_score(b), b) for b in boxes if side_of(b) == "L"]
        right_candidates = [(far_score(b), b) for b in boxes if side_of(b) == "R"]
        left_far  = max(left_candidates)[0]  if left_candidates  else None
        right_far = max(right_candidates)[0] if right_candidates else None

        if num_lanes == 2:
            # 더 멀리 보이는 쪽이 반대 차로 → 그 반대편이 내 차로
            if left_far is None and right_far is None:
                avg_cx = sum(b.center.x for b in boxes) / len(boxes)
                return 2, (1 if avg_cx > cx0 else 2)
            if right_far is None:
                return 2, 2
            if left_far is None:
                return 2, 1
            return (2, 2) if left_far > right_far else (2, 1)

        if num_lanes == 3:
            if left_far and right_far:
                return 3, 2
            if left_far and not right_far:
                return 3, 3
            if right_far and not left_far:
                return 3, 1
            # 둘 다 애매 → 보수적으로 가운데
            return 3, 2

        # 4차선 이상 등은 중앙 근사치 반환
        return num_lanes, max(1, min(num_lanes, (num_lanes + 1) // 2))

    # ---------- Inference callback ----------
    def image_cb(self, msg: Image) -> None:
        if not self.enable:
            return

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        results_list = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            device=self.device
        )
        results: Results = results_list[0].cpu()

        bboxes_msg = DetectionArray()
        bboxes_msg.header = msg.header

        hypothesis = self.parse_hypothesis(results) if results.boxes else []
        boxes = self.parse_boxes(results) if results.boxes else []

        num = len(hypothesis)
        car_count = 0

        for i in range(num):
            cls_id = hypothesis[i]["class_id"]
            cls_name = hypothesis[i]["class_name"]

            if not self._is_target(cls_id, cls_name):
                continue

            car_count += 1

            det = Detection()
            # 단일 클래스 'cars'로 고정, bbox만 사용
            det.class_id = 0
            det.class_name = "cars"
            det.score = 1.0
            det.bbox = boxes[i]

            bboxes_msg.detections.append(det)

        # 로깅
        self.get_logger().info(f"Cars in front: {car_count}")

        # ---- 퍼블리시 ----
        # 1) bbox만 포함된 DetectionArray
        self._bboxes_pub.publish(bboxes_msg)

        # 2) 통합 차로 정보 [전체차로수, 현재주행차로] (Int32MultiArray)
        img_w = cv_image.shape[1]
        img_h = cv_image.shape[0]
        _, my_lane = self.estimate_my_lane_from_boxes(
            [d.bbox for d in bboxes_msg.detections],
            img_w=img_w, img_h=img_h
        )
        lane_msg = Int32MultiArray()
        lane_msg.data = [int(car_count), int(my_lane)]
        self._lane_info_pub.publish(lane_msg)
        self.get_logger().info(f"[LaneInfo] {int(car_count)},{int(my_lane)}")

        # 메모리 정리
        del results
        del cv_image


# ---------- main ----------
def main():
    rclpy.init()
    node = YoloFrontCarNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
