#!/usr/bin/env python3
"""
Velocity Node
시리얼 통신으로 아날로그 값(350~740)을 읽어서
0~60 범위의 정수로 변환하여 velocity 토픽으로 발행합니다.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import serial
import serial.tools.list_ports


class VelocityNode(Node):
    def __init__(self):
        super().__init__('velocity_node')
        
        # 파라미터 선언
        self.declare_parameter('serial_port', '/dev/ttyUSB0')  # 윈도우의 경우 'COM3' 같은 형식
        self.declare_parameter('baud_rate', 9600)
        self.declare_parameter('publish_rate', 20.0)  # Hz
        self.declare_parameter('min_analog', 350)
        self.declare_parameter('max_analog', 740)
        self.declare_parameter('min_velocity', 0)
        self.declare_parameter('max_velocity', 60)
        
        # 파라미터 가져오기
        serial_port = self.get_parameter('serial_port').value
        baud_rate = self.get_parameter('baud_rate').value
        publish_rate = self.get_parameter('publish_rate').value
        self.min_analog = self.get_parameter('min_analog').value
        self.max_analog = self.get_parameter('max_analog').value
        self.min_velocity = self.get_parameter('min_velocity').value
        self.max_velocity = self.get_parameter('max_velocity').value
        
        # Publisher 생성
        self.velocity_pub = self.create_publisher(Int32, 'velocity', 10)
        
        # 시리얼 포트 초기화
        try:
            self.serial_conn = serial.Serial(
                port=serial_port,
                baudrate=baud_rate,
                timeout=1.0
            )
            self.get_logger().info(f'시리얼 포트 {serial_port} 연결 성공 (baudrate: {baud_rate})')
        except serial.SerialException as e:
            self.get_logger().error(f'시리얼 포트 연결 실패: {e}')
            self.get_logger().info('사용 가능한 시리얼 포트:')
            for port in serial.tools.list_ports.comports():
                self.get_logger().info(f'  - {port.device}: {port.description}')
            self.serial_conn = None
        
        # 타이머 생성 - 주기적으로 시리얼 읽기 및 발행
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('Velocity Node 시작됨')
        self.get_logger().info(f'아날로그 범위: {self.min_analog}~{self.max_analog}')
        self.get_logger().info(f'속도 범위: {self.min_velocity}~{self.max_velocity}')
    
    def map_value(self, value, in_min, in_max, out_min, out_max):
        """
        값을 한 범위에서 다른 범위로 매핑합니다.
        """
        # 입력 범위 체크
        value = max(in_min, min(in_max, value))
        
        # 선형 매핑
        mapped = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        
        return int(round(mapped))
    
    def timer_callback(self):
        """
        타이머 콜백 - 시리얼에서 데이터를 읽고 velocity 토픽으로 발행
        """
        if self.serial_conn is None:
            return
        
        try:
            # 시리얼에서 한 줄 읽기 (줄바꿈까지)
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                
                if line:
                    try:
                        # 아날로그 값 파싱
                        analog_value = int(line)
                        
                        # 속도 값으로 변환 (350~740 -> 0~60)
                        velocity = self.map_value(
                            analog_value,
                            self.min_analog,
                            self.max_analog,
                            self.min_velocity,
                            self.max_velocity
                        )
                        
                        # 토픽 발행
                        msg = Int32()
                        msg.data = velocity
                        self.velocity_pub.publish(msg)
                        
                        self.get_logger().debug(f'아날로그: {analog_value} -> 속도: {velocity}')
                        
                    except ValueError:
                        self.get_logger().warn(f'잘못된 값: {line}')
                        
        except serial.SerialException as e:
            self.get_logger().error(f'시리얼 읽기 오류: {e}')
        except Exception as e:
            self.get_logger().error(f'예상치 못한 오류: {e}')
    
    def destroy_node(self):
        """
        노드 종료 시 시리얼 포트 닫기
        """
        if self.serial_conn is not None and self.serial_conn.is_open:
            self.serial_conn.close()
            self.get_logger().info('시리얼 포트 닫힘')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VelocityNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

