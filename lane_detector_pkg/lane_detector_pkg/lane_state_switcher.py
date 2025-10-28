import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray


class lane_state_switcher(Node):
    def __init__(self):
        super().__init__('lane_state_switcher')

        # 구독자 설정
        self.yolo_state_sub = self.create_subscription(
            Int32MultiArray,
            '/lane/info/yolo',
            self.yolo_state_callback,
            10
        )

        self.cv_state_sub = self.create_subscription(
            Int32MultiArray,
            '/lane/info/cv',
            self.cv_state_callback,
            10
        )

        # 발행자 설정
        self.lane_state_pub = self.create_publisher(
            Int32MultiArray,
            '/lane/info',   
            10
        )

        # 초기 상태 변수들
        self.yolo_lane_info = [0, 0]
        self.cv_lane_info = [0, 0]

        # 20Hz 타이머 (0.05초마다 실행)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def yolo_state_callback(self, msg):
        self.yolo_lane_info = msg.data  # [total_lanes, current_lane]

    def cv_state_callback(self, msg):
        self.cv_lane_info = msg.data  # [total_lanes, current_lane]

    def timer_callback(self):
        # YOLO 데이터가 [0,0]이 아니면 YOLO 우선, 그렇지 않으면 CV 사용
        if (len(self.yolo_lane_info) >= 2 and not (self.yolo_lane_info[0] == 0 and self.yolo_lane_info[1] == 0)):
            selected_data = self.yolo_lane_info
            source = "YOLO"
        else:
            selected_data = self.cv_lane_info
            source = "CV"
        
        # /lane/info로 선택된 데이터 발행
        lane_msg = Int32MultiArray()
        lane_msg.data = selected_data
        self.lane_state_pub.publish(lane_msg)
        
        self.get_logger().info(f'Published to /lane/info: {selected_data} (Source: {source})')


def main(args=None):
    rclpy.init(args=args)
    node = lane_state_switcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()   
