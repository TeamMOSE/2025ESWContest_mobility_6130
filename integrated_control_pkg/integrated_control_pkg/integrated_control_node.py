#!/usr/bin/env python3
"""
통합 제어 노드 (Integrated Control Node)
소방차 사이렌 감지 시 차량 방향 유도 프로그램
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int32
from std_msgs.msg import Int32MultiArray
import random
import serial
import serial.tools.list_ports


class IntegratedControlNode(Node):
    def __init__(self):
        super().__init__('integrated_control_node')
        
        # ========== 구독자 설정 ==========
        # 사이렌 감지 (siren_pkg)
        self.siren_detected_sub = self.create_subscription(
            String, 'siren_detected', self.siren_detected_callback, 10)
        self.sound_direction_sub = self.create_subscription(
            String, 'sound_direction', self.sound_direction_callback, 10)
        
        # 교차로 감지 (전방 카메라)
        self.intersection_sub = self.create_subscription(
            Bool, 'intersection', self.intersection_callback, 10)
        
        # 소방차 감지 (후방 카메라)
        self.firetruck_side_sub = self.create_subscription(
            String, 'firetruck_side', self.firetruck_side_callback, 10)
        
        # 차선 감지 (전방 카메라)
        self.lane_info_sub = self.create_subscription(
            Int32MultiArray, 'lane/info', self.lane_info_callback, 10)
        
        # 속도 정보
        self.velocity_sub = self.create_subscription(
            Int32, 'velocity', self.velocity_callback, 10)
        
        # ========== 발행자 설정 ==========
        self.decision_pub = self.create_publisher(String, 'decision', 10)
        
        # ========== 상태 변수 ==========
        self.siren_detected = "NOSIREN"  # NOSIREN / SIREN
        self.sound_direction = "back"     # front / back
        self.is_intersection = False      # 교차로 여부
        self.firetruck_side = "None"      # Rear_Right / Rear_Left / None
        self.lane_info = [0, 0]           # [전체 차선 개수, 현재 차선 번호]
        self.velocity = 0                 # 현재 속도 (정수)
        
        # ========== 시리얼 통신 상태 변수 ==========
        self.last_serial_command = None   # 마지막으로 전송한 시리얼 명령 (중복 방지)
        
        # ========== 제어 루프 타이머 ==========
        # 10Hz (0.1초마다 제어 로직 실행)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # ========== 속도 기준 설정 ==========
        self.SPEED_THRESHOLD = 30  # 주행 중/서행 중 구분 기준 (km/h)
        
        # ========== 시리얼 통신 설정 ==========
        self.declare_parameter('serial_port', 'COM3')  # Windows 기본값
        self.declare_parameter('serial_baudrate', 9600)
        self.declare_parameter('serial_timeout', 1.0)
        
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        serial_baudrate = self.get_parameter('serial_baudrate').get_parameter_value().integer_value
        serial_timeout = self.get_parameter('serial_timeout').get_parameter_value().double_value
        
        # 시리얼 포트 초기화
        self.serial_connection = None
        try:
            self.serial_connection = serial.Serial(
                port=serial_port,
                baudrate=serial_baudrate,
                timeout=serial_timeout
            )
            self.get_logger().info(f"시리얼 포트 연결 성공: {serial_port} @ {serial_baudrate} baud")
        except serial.SerialException as e:
            self.get_logger().warning(f"시리얼 포트 연결 실패: {e}")
            self.get_logger().warning("시리얼 통신 없이 계속 실행합니다.")
        except Exception as e:
            self.get_logger().error(f"시리얼 포트 초기화 오류: {e}")
        
        self.get_logger().info("=== Integrated Control Node 시작 ===")
        self.get_logger().info("구독 토픽:")
        self.get_logger().info("  - siren_detected (String)")
        self.get_logger().info("  - sound_direction (String)")
        self.get_logger().info("  - intersection (Bool)")
        self.get_logger().info("  - firetruck_side (String)")
        self.get_logger().info("  - lane/info (Int32MultiArray)")
        self.get_logger().info("  - velocity (Int32)")
        self.get_logger().info("발행 토픽:")
        self.get_logger().info("  - decision (String): Left/Right/Stop/None/Caution")
        self.get_logger().info(f"속도 기준: {self.SPEED_THRESHOLD} km/h (주행 중/서행 중)")
    
    # ========== 콜백 함수들 ==========
    def siren_detected_callback(self, msg):
        """사이렌 감지 여부 콜백"""
        self.siren_detected = msg.data
        self.get_logger().debug(f"사이렌 감지: {self.siren_detected}")
    
    def sound_direction_callback(self, msg):
        """사이렌 방향 콜백"""
        self.sound_direction = msg.data
        self.get_logger().debug(f"사이렌 방향: {self.sound_direction}")
    
    def intersection_callback(self, msg):
        """교차로 감지 콜백"""
        self.is_intersection = msg.data
        self.get_logger().debug(f"교차로 감지: {self.is_intersection}")
    
    def firetruck_side_callback(self, msg):
        """소방차 위치 콜백"""
        self.firetruck_side = msg.data
        self.get_logger().debug(f"소방차 위치: {self.firetruck_side}")
    
    def lane_info_callback(self, msg):
        """차선 정보 콜백"""
        if len(msg.data) >= 2:
            self.lane_info = [msg.data[0], msg.data[1]]
            self.get_logger().debug(
                f"차선 정보: 전체={self.lane_info[0]}, 현재={self.lane_info[1]}")
    
    def velocity_callback(self, msg):
        """속도 정보 콜백"""
        self.velocity = msg.data
        self.get_logger().debug(f"속도: {self.velocity}")
    
    # ========== 제어 로직 ==========
    def control_loop(self):
        """메인 제어 루프 (10Hz)"""
        decision_command = self.calculate_control_command()
        
        # 제어 명령 발행
        msg = String()
        msg.data = decision_command
        self.decision_pub.publish(msg)
        
        # Left, Right 명령일 때만 시리얼로 1회 전송
        if decision_command in ["Left", "Right"]:
            self._send_serial_once(decision_command)
        
        # 로깅 (None이 아닌 경우만)
        if decision_command != "None":
            self.get_logger().info(f"결정 명령: {decision_command}")
    
    def calculate_control_command(self) -> str:
        """
        플로우차트 기반 제어 명령 계산
        Returns: "Left" | "Right" | "Stop" | "None"
        """
        # ========== 1. 사이렌 감지 여부 확인 ==========
        # NOSIREN이면 사이렌이 없는 것
        if self.siren_detected != "SIREN":
            return "None"
        
        # ========== 2. 사이렌 방향 판별 ==========
        # SIREN일 때만 sound_direction으로 전방/후방 결정
        
        # 2-1. 전방 사이렌
        if self.sound_direction == "front":
            # 교차로인 경우에만 정지
            if self.is_intersection:
                return "Stop"
            else:
                return "None"
        
        # 2-2. 후방 사이렌 (플로우차트 로직)
        elif self.sound_direction == "back":
            return self._handle_rear_siren()
        
        return "None"
    
    def _handle_rear_siren(self) -> str:
        """
        후방 사이렌 감지 시 로직
        
        1. 주행 중 (velocity >= 30)
           - 교차로: RIGHT
           - 아니오: 차선 수에 따른 사이드 정렬
        
        2. 서행 중 (velocity < 30)
           - 소방차 보임: 반대 방향으로 Left/Right
           - 소방차 안 보임:
             - 교차로: RIGHT
             - 아니오: 차선 수에 따른 끝쪽 유도
        """
        # 1. 주행 중 (velocity >= 30)
        if self.velocity >= self.SPEED_THRESHOLD:
            # 교차로인가?
            if self.is_intersection:
                return "Right"
            else:
                # 차선 정보 기반 사이드 정렬
                return self._lane_based_alignment_driving()
        
        # 2. 서행 중 (velocity < 30)
        else:
            # 후방카메라에 소방차가 보이는가?
            firetruck_visible = (self.firetruck_side != "None")
            
            if firetruck_visible:
                # 소방차의 반대 방향으로
                return self._opposite_direction()
            else:
                # 교차로인가?
                if self.is_intersection:
                    return "Right"
                else:
                    # 차선 정보 기반 끝쪽 유도
                    return self._lane_based_alignment_slow()
    
    def _opposite_direction(self) -> str:
        """
        서행 중 소방차가 보일 때: 반대 방향으로 회피
        
        Rear_Left → Right
        Rear_Right → Left
        """
        if self.firetruck_side == "Rear_Left":
            return "Right"
        elif self.firetruck_side == "Rear_Right":
            return "Left"
        else:
            # None인 경우 (예외 상황)
            return "None"
    
    def _lane_based_alignment_driving(self) -> str:
        """
        주행 중 차선 기반 사이드 정렬
        
        - 편도 1차선: Right
        - 편도 2차선:
          - 1차선: Right
          - 2차선: Caution
        - 편도 3차선 이상:
          - 끝차선(1차선 또는 최대차선): Caution
          - 가운데차선(홀수 차선에서): Left/Right 랜덤
          - 그 외: 끝쪽으로 (1에 가까우면 Right, 최대에 가까우면 Left)
        """
        total_lanes = self.lane_info[0]
        current_lane = self.lane_info[1]
        
        # 차선 정보가 유효하지 않으면 기본값
        if total_lanes <= 0 or current_lane <= 0:
            return "Right"
        
        # 편도 1차선
        if total_lanes == 1:
            return "Right"
        
        # 편도 2차선
        elif total_lanes == 2:
            if current_lane == 1:
                return "Right"
            else:  # current_lane == 2
                return "Caution"
        
        # 편도 3차선 이상
        else:
            # 끝차선 (1차선 또는 최대차선)
            if current_lane == 1 or current_lane == total_lanes:
                return "Caution"
            
            # 가운데 차선 (홀수 차선 수에서)
            if total_lanes % 2 == 1:  # 홀수
                middle = (total_lanes + 1) // 2
                if current_lane == middle:
                    return random.choice(["Left", "Right"])
            
            # 그 외: 끝쪽으로
            middle = (total_lanes + 1) / 2
            if current_lane < middle:
                # 1차선쪽에 가까움 → 오른쪽으로
                return "Right"
            else:
                # 최대차선쪽에 가까움 → 왼쪽으로
                return "Left"
    
    def _lane_based_alignment_slow(self) -> str:
        """
        서행 중 차선 기반 끝쪽 유도 (소방차 안 보임, 교차로 아님)
        
        - 편도 1차선: Right
        - 편도 2차선:
          - 1차선: Left (왼쪽 끝으로)
          - 2차선: Right (오른쪽 끝으로)
        - 편도 3차선 이상:
          - 더 끝쪽으로 유도 (Caution 없음)
          - 1차선쪽에 가까우면 Right (더 왼쪽으로)
          - 최대차선쪽에 가까우면 Left (더 오른쪽으로)
        """
        total_lanes = self.lane_info[0]
        current_lane = self.lane_info[1]
        
        # 차선 정보가 유효하지 않으면 기본값
        if total_lanes <= 0 or current_lane <= 0:
            return "Right"
        
        # 편도 1차선
        if total_lanes == 1:
            return "Right"
        
        # 편도 2차선
        elif total_lanes == 2:
            if current_lane == 1:
                return "Left"  # 왼쪽 끝으로
            else:  # current_lane == 2
                return "Right"  # 오른쪽 끝으로
        
        # 편도 3차선 이상
        else:
            # 가운데 기준으로 끝쪽으로 유도
            middle = (total_lanes + 1) / 2
            if current_lane < middle:
                # 1차선쪽에 가까움 → 더 왼쪽으로 (Right 방향)
                return "Right"
            elif current_lane > middle:
                # 최대차선쪽에 가까움 → 더 오른쪽으로 (Left 방향)
                return "Left"
            else:
                # 정확히 가운데 (홀수 차선에서)
                # 랜덤으로 끝쪽 선택
                return random.choice(["Left", "Right"])
    
    def _send_serial_once(self, command: str):
        """
        시리얼로 명령 전송 (중복 방지)
        
        Left → "LEFT" 전송
        Right → "RIGHT" 전송
        """
        # 이전과 같은 명령이면 전송하지 않음 (1회만 전송)
        if command == self.last_serial_command:
            return
        
        # 시리얼 연결이 없으면 전송 안 함
        if self.serial_connection is None or not self.serial_connection.is_open:
            return
        
        try:
            # 명령어를 대문자로 변환하여 전송
            serial_msg = command.upper() + '\n'
            self.serial_connection.write(serial_msg.encode('utf-8'))
            self.serial_connection.flush()
            
            # 마지막 전송 명령 업데이트
            self.last_serial_command = command
            
            self.get_logger().info(f"시리얼 전송: {serial_msg.strip()}")
        except serial.SerialException as e:
            self.get_logger().error(f"시리얼 전송 오류: {e}")
        except Exception as e:
            self.get_logger().error(f"시리얼 전송 중 예외 발생: {e}")
    
    def destroy_node(self):
        """노드 종료 시 시리얼 포트 정리"""
        if self.serial_connection is not None and self.serial_connection.is_open:
            self.serial_connection.close()
            self.get_logger().info("시리얼 포트 연결 종료")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("사용자에 의해 중지됨")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

