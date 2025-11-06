#!/usr/bin/env python3
"""
Handle Serial Reader Node
아두이노 handle.ino에서 보내는 시리얼 데이터(analogRead(A0) 값)를 읽어서
ROS2 토픽으로 발행하는 노드입니다.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import serial
import serial.tools.list_ports


class HandleSerialReaderNode(Node):
    def __init__(self):
        super().__init__('handle_serial_reader_node')
        
        # 파라미터 선언
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('publish_rate', 10.0)  # Hz
        
        # 파라미터 가져오기
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        # Publisher 생성 - 아날로그 값을 그대로 발행
        self.analog_pub = self.create_publisher(Int32, '/car_speed_raw', 10)
        
        # 시리얼 포트 초기화
        self.serial_conn = None
        try:
            self.serial_conn = serial.Serial(
                port=serial_port,
                baudrate=baud_rate,
                timeout=1.0
            )
            self.get_logger().info(f'시리얼 포트 연결 성공: {serial_port} @ {baud_rate} baud')
        except serial.SerialException as e:
            self.get_logger().error(f'시리얼 포트 연결 실패: {e}')
            self.get_logger().info('사용 가능한 시리얼 포트:')
            for port in serial.tools.list_ports.comports():
                self.get_logger().info(f'  - {port.device}: {port.description}')
            self.serial_conn = None
        except Exception as e:
            self.get_logger().error(f'시리얼 포트 초기화 오류: {e}')
            self.serial_conn = None
        
        # 타이머 생성 - 주기적으로 시리얼 읽기 및 발행
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # 버퍼 초기화 (초기 메시지 무시)
        self.buffer_cleared = False
        
        self.get_logger().info('Handle Serial Reader Node 시작됨')
        self.get_logger().info(f'발행 주기: {publish_rate} Hz')
    
    def timer_callback(self):
        """
        타이머 콜백 - 시리얼에서 데이터를 읽고 토픽으로 발행
        """
        if self.serial_conn is None:
            return
        
        try:
            # 시리얼에서 한 줄 읽기 (줄바꿈까지)
            if self.serial_conn.in_waiting > 0:
                # 버퍼에 있는 모든 데이터 읽기 (한 줄씩)
                while self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    
                    if not line:
                        continue
                    
                    # 초기 버퍼 클리어 (아두이노 시작 메시지 무시)
                    if not self.buffer_cleared:
                        if any(keyword in line for keyword in ['Haptic', 'Commands', 'Pins', 'Speed', 'Ready']):
                            continue
                        self.buffer_cleared = True
                    
                    try:
                        # 아날로그 값 파싱
                        analog_value = int(line)
                        
                        # 토픽 발행
                        msg = Int32()
                        msg.data = analog_value
                        self.analog_pub.publish(msg)
                        
                    except ValueError:
                        # 숫자가 아닌 경우 무시 (명령어 응답 등)
                        pass
                        
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
    node = HandleSerialReaderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

