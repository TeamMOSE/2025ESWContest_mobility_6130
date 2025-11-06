# yolo_trafficlight_node.py
from typing import List, Dict, Set

import rclpy
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState

from cv_bridge import CvBridge
from ultralytics import YOLO
from ultralytics.engine.results import Results
from torch import cuda

from sensor_msgs.msg import CompressedImage
from interfaces_pkg.msg import Point2D, BoundingBox2D, Mask, KeyPoint2D, KeyPoint2DArray, Detection, DetectionArray
from std_srvs.srv import SetBool
from std_msgs.msg import Bool
import numpy as np
import cv2


class YoloTrafficLightNode(LifecycleNode):
    def __init__(self, **kwargs) -> None:
        super().__init__("yolo_trafficlight_node", **kwargs)

        # ---- Parameters ----
        # 모델 경로(요청 사양)
        self.model = "/home/cyjung/ESW_ws/src/camera_pkg/models/best_trafficlight.pt"
        self.declare_parameter("device", "cuda:0")          # "cuda:0" 가능
        self.declare_parameter("threshold", 0.6)
        self.declare_parameter("enable", True)

        # 전방 카메라 구독 토픽 (요청 사양: front_camera/compressed)
        self.declare_parameter("image_topic", "front_camera/compressed")

        # 타깃 클래스 (이름/ID) — 기본은 traffic light만
        # 모델에 따라 이름이 'traffic light', 'traffic_light', 'Traffic_Light' 등일 수 있어 약간 너그럽게 포함
        self.declare_parameter("target_names", ["traffic_light", "traffic light", "Traffic_Light"])
        self.declare_parameter("target_ids", [])  # 정수 ID 직접 지정 시 최우선
        
        # 프레임 스킵 설정 (극한 최적화 - CPU 부하 최소화)
        self.declare_parameter("process_every_n_frames", 4)  # 매 4프레임 (8fps → 2fps 추론)

        self.get_logger().info("YoloTrafficLightNode created")

    # ---------- Lifecycle ----------
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Configuring {self.get_name()}")

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
        
        # 프레임 스킵 설정 가져오기
        self.process_every_n_frames = self.get_parameter("process_every_n_frames").get_parameter_value().integer_value
        self.frame_counter = 0

        # QoS: 무선 통신 최적화 (BEST_EFFORT + 버퍼링)
        self.image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=3  # 1 → 3 (네트워크 지연 시 버퍼링)
        )

        # 퍼블리셔/서비스
        self._detections_pub = self.create_lifecycle_publisher(DetectionArray, "trafficlight_detections", 10)
        self._intersection_pub = self.create_lifecycle_publisher(Bool, "intersection", 10)
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
            # 모델의 names에서 타깃 클래스 ID 해석
            try:
                if hasattr(self.yolo, "names") and isinstance(self.yolo.names, (dict, list)):
                    names_dict = self.yolo.names if isinstance(self.yolo.names, dict) else {i: n for i, n in enumerate(self.yolo.names)}
                    lower_target_names = {s.lower() for s in self.target_names}
                    # 스페이스/언더스코어 차이를 완화
                    def norm(s: str) -> str:
                        return str(s).lower().replace("_", " ")
                    target_norm = {norm(s) for s in lower_target_names}
                    self.model_derived_ids = [
                        int(cid) for cid, n in names_dict.items()
                        if norm(n) in target_norm
                    ]
                    # 커스텀 모델이 단일 클래스일 때는 그것을 타깃으로 가정
                    if not self.model_derived_ids and len(names_dict) == 1:
                        self.model_derived_ids = [int(list(names_dict.keys())[0])]
                    self.get_logger().info(f"Target class IDs resolved: {self.model_derived_ids}")
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
            self.image_topic,  # e.g. front_camera/compressed
            self.image_cb,
            self.image_qos_profile
        )
        self.get_logger().info(f"Subscribing to {self.image_topic} with BEST_EFFORT QoS")
        self.get_logger().info(f"프레임 스킵: 매 {self.process_every_n_frames}프레임마다 처리 (CPU 부하 감소)")

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
        self.destroy_publisher(self._detections_pub)
        self.destroy_publisher(self._intersection_pub)
        del self.image_qos_profile
        return TransitionCallbackReturn.SUCCESS

    # ---------- Parsing helpers ----------
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
            msg.center.x = float(xywh[0])
            msg.center.y = float(xywh[1])
            msg.center.theta = 0.0
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

    # 타깃 필터링: param IDs > model-derived IDs > 이름(정확/느슨 매칭)
    def _is_target(self, cls_id: int, cls_name: str) -> bool:
        if self.target_ids_param:
            return cls_id in self.target_ids_param
        if getattr(self, "model_derived_ids", []):
            return cls_id in self.model_derived_ids
        if cls_name:
            name_l = cls_name.lower().replace("_", " ")
            return name_l in {s.replace("_", " ") for s in self.target_names}
        return False

    # ---------- Inference callback ----------
    def image_cb(self, msg: CompressedImage) -> None:
        if not self.enable:
            return
        
        # 프레임 스킵 (무선 통신 최적화 - CPU 부하 감소)
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n_frames != 0:
            return

        # CompressedImage 압축 해제 (CvBridge 사용)
        cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
        results_list = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            device=self.device
        )
        results: Results = results_list[0].cpu()

        detections_msg = DetectionArray()
        detections_msg.header = msg.header
        log_entries = []

        hypothesis = self.parse_hypothesis(results) if results.boxes else []
        boxes = self.parse_boxes(results) if results.boxes else []
        masks = self.parse_masks(results) if results.masks else []
        keypoints = self.parse_keypoints(results) if results.keypoints else []

        num = len(hypothesis)
        any_trafficlight = False

        for i in range(num):
            cls_id = hypothesis[i]["class_id"]
            cls_name = hypothesis[i]["class_name"]

            if not self._is_target(cls_id, cls_name):
                continue

            det = Detection()
            det.class_id = cls_id
            det.class_name = cls_name
            det.score = hypothesis[i]["score"]
            det.bbox = boxes[i]

            if i < len(masks):
                det.mask = masks[i]
            if i < len(keypoints):
                det.keypoints = keypoints[i]

            detections_msg.detections.append(det)
            log_entries.append(f"{det.class_name} ({det.score:.2f})")
            any_trafficlight = True

        if log_entries:
            self.get_logger().info("TrafficLight detections: " + ", ".join(log_entries))

        # ---- intersection 퍼블리시 (firetruck 스타일) ----
        msg_bool = Bool()
        msg_bool.data = True if any_trafficlight else False
        self._intersection_pub.publish(msg_bool)
        self.get_logger().info(f"TrafficLight decision: {'True' if msg_bool.data else 'False'}")

        # ---- 보조: 검출 결과 퍼블리시 ----
        self._detections_pub.publish(detections_msg)

        del results
        del cv_image


    # ---------- main ----------
def main():
    rclpy.init()
    node = YoloTrafficLightNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
