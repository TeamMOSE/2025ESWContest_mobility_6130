from typing import Tuple, Dict
import math
import random

import rclpy
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState

from sensor_msgs.msg import Image, CompressedImage
from interfaces_pkg.msg import DetectionArray
from cv_bridge import CvBridge
import cv2
import numpy as np

from message_filters import Subscriber, ApproximateTimeSynchronizer


class TrafficLightVisualizerNode(LifecycleNode):
    def __init__(self, **kwargs):
        super().__init__("trafficlight_visualizer_node", **kwargs)

        self.declare_parameter("image_topic", "front_camera/compressed")
        self.declare_parameter("detections_topic", "trafficlight_detections")
        self.declare_parameter("output_topic", "trafficlight_visualized_img")
        self.declare_parameter("min_confidence", 0.0)
        self.declare_parameter("thickness", 2)
        self.declare_parameter("font_scale", 0.6)
        self.declare_parameter("draw_keypoints", True)
        self.declare_parameter("draw_masks", True)
        self.declare_parameter("output_encoding", "bgr8")
        self.declare_parameter("sync_slop_sec", 0.10)
        self.declare_parameter("sync_queue", 5)

        self.get_logger().info("TrafficLightVisualizerNode created")

        self._bridge = CvBridge()
        self._color_cache: Dict[str, Tuple[int, int, int]] = {}

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring")

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.min_conf = self.get_parameter("min_confidence").get_parameter_value().double_value
        self.thickness = int(self.get_parameter("thickness").get_parameter_value().integer_value or 2)
        self.font_scale = float(self.get_parameter("font_scale").get_parameter_value().double_value or 0.6)
        self.draw_keypoints = self.get_parameter("draw_keypoints").get_parameter_value().bool_value
        self.draw_masks = self.get_parameter("draw_masks").get_parameter_value().bool_value
        self.output_encoding = self.get_parameter("output_encoding").get_parameter_value().string_value
        self.sync_slop_sec = float(self.get_parameter("sync_slop_sec").get_parameter_value().double_value or 0.10)
        self.sync_queue = int(self.get_parameter("sync_queue").get_parameter_value().integer_value or 5)

        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=3
        )
        self.det_qos = 10

        self._img_pub = self.create_lifecycle_publisher(Image, self.output_topic, 10)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Activating")

        self._sub_img = Subscriber(self, CompressedImage, self.image_topic, qos_profile=self.image_qos)
        self._sub_det = Subscriber(self, DetectionArray, self.detections_topic, qos_profile=self.det_qos)

        self._sync = ApproximateTimeSynchronizer(
            [self._sub_img, self._sub_det],
            queue_size=self.sync_queue,
            slop=self.sync_slop_sec
        )
        self._sync.registerCallback(self.sync_cb)

        super().on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating")
        self._sync = None
        self._sub_img = None
        self._sub_det = None
        super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up")
        self.destroy_publisher(self._img_pub)
        return TransitionCallbackReturn.SUCCESS

    def _color_for(self, key: str) -> Tuple[int, int, int]:
        if key not in self._color_cache:
            rnd = random.Random(hash(key) & 0xFFFFFFFF)
            self._color_cache[key] = (rnd.randint(30, 255), rnd.randint(30, 255), rnd.randint(30, 255))
        return self._color_cache[key]

    @staticmethod
    def _clip_box(xmin, ymin, xmax, ymax, w, h):
        return (max(0, min(int(xmin), w - 1)),
                max(0, min(int(ymin), h - 1)),
                max(0, min(int(xmax), w - 1)),
                max(0, min(int(ymax), h - 1)))

    @staticmethod
    def _draw_label(img, tl, label, color, font_scale=0.6, thickness=1):
        x, y = tl
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness + 1)
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), color, -1)
        cv2.putText(img, label, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)

    def sync_cb(self, img_msg: CompressedImage, dets_msg: DetectionArray):
        cv_img = self._bridge.compressed_imgmsg_to_cv2(img_msg)
        if cv_img.ndim == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)

        h, w = cv_img.shape[:2]

        num_dets = len(dets_msg.detections)
        log_items = []

        for det in dets_msg.detections:
            if det.score < self.min_conf:
                continue

            cx = det.bbox.center.x
            cy = det.bbox.center.y
            bw = det.bbox.size.x
            bh = det.bbox.size.y

            xmin = cx - bw / 2.0
            ymin = cy - bh / 2.0
            xmax = cx + bw / 2.0
            ymax = cy + bh / 2.0

            xmin, ymin, xmax, ymax = self._clip_box(xmin, ymin, xmax, ymax, w, h)

            label = f"{det.class_name or det.class_id} ({det.score:.2f})"
            color = self._color_for(det.class_name or str(det.class_id))

            cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), color, max(1, self.thickness))

            if self.draw_masks and det.mask and len(det.mask.data) > 0:
                pts = np.array([[int(p.x), int(p.y)] for p in det.mask.data], dtype=np.int32)
                overlay = cv_img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.35, cv_img, 0.65, 0, cv_img)

            if self.draw_keypoints and det.keypoints and len(det.keypoints.data) > 0:
                for kp in det.keypoints.data:
                    if kp.score <= 0.0:
                        continue
                    cv2.circle(cv_img, (int(kp.point.x), int(kp.point.y)), 3, color, -1, cv2.LINE_AA)

            self._draw_label(cv_img, (xmin, max(15, ymin)), label, color, self.font_scale, 1)

            log_items.append(label)

        if log_items:
            self.get_logger().info("Drawn: " + ", ".join(log_items))

        out_msg = self._bridge.cv2_to_imgmsg(cv_img, encoding=self.output_encoding)
        out_msg.header = img_msg.header
        self._img_pub.publish(out_msg)


def main():
    rclpy.init()
    node = TrafficLightVisualizerNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

