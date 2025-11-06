import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32


class CarSpeedMapper(Node):
    def __init__(self):
        super().__init__('car_speed_map_node')

        # 파라미터화 해두면 나중에 ros2 param으로 조정도 가능
        self.declare_parameter('idle_max', 360.0)
        self.declare_parameter('min_raw', 360.0)
        self.declare_parameter('max_raw', 735.0)
        self.declare_parameter('max_speed', 60.0)

        self.idle_max = float(self.get_parameter('idle_max').value)
        self.min_raw = float(self.get_parameter('min_raw').value)
        self.max_raw = float(self.get_parameter('max_raw').value)
        self.max_speed = float(self.get_parameter('max_speed').value)

        # 구독: car_speed_raw (Int32)
        self.subscription = self.create_subscription(
            Int32,
            '/car_speed_raw',
            self.car_speed_raw_callback,
            10
        )

        # 퍼블리셔: car_speed
        self.publisher = self.create_publisher(
            Float32,
            'car_speed',
            10
        )

        self.get_logger().info(
            f'CarSpeedMapper node started. '
            f'raw[{self.min_raw} ~ {self.max_raw}] -> 0 ~ {self.max_speed} km/h'
        )

    def car_speed_raw_callback(self, msg: Int32):
        raw = float(msg.data)

        # 1) 페달 안 밟은 구간: 0 km/h
        if raw <= self.idle_max:
            speed = 0.0

        # 2) 풀악셀 이상: max_speed 고정
        elif raw >= self.max_raw:
            speed = self.max_speed

        # 3) 중간 구간: 선형 매핑
        else:
            normalized = (raw - self.min_raw) / (self.max_raw - self.min_raw)  # 0~1
            speed = normalized * self.max_speed

        # 퍼블리시
        out_msg = Float32()
        out_msg.data = float(speed)
        self.publisher.publish(out_msg)

        # 디버그 로그 (원하면 주석 해제)
        # self.get_logger().info(f'raw: {raw:.1f} -> speed: {speed:.1f} km/h')


def main(args=None):
    rclpy.init(args=args)
    node = CarSpeedMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

