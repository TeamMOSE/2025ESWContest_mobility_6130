#!/usr/bin/env python3

import sys
import json
import time
import os
from pathlib import Path

# PyQt6 임포트 (GUI 모드에서만)
try:
    from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QGraphicsOpacityEffect
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
    from PyQt6.QtGui import QKeyEvent, QFont, QResizeEvent
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    print("PyQt6 not available, running in console mode only")

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class EmergencyHUDWidget(QWidget):
    """긴급차량 HUD UI 위젯"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 레이아웃 설정 (하단 정렬)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        layout.setContentsMargins(50, 0, 50, 100)
        layout.setSpacing(20)
        self.setLayout(layout)

        # 스타일 설정
        self.setStyleSheet("""
            QWidget {
                background: transparent;
                border: none;
            }
            QLabel {
                background: transparent;
            }
        """)

        # 메인 텍스트 (화살표)
        self.main_text = QLabel()
        self.main_text.setFont(QFont("DejaVu Sans Mono", 150, QFont.Weight.Black))
        self.main_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_text.setStyleSheet("color: rgba(255, 51, 51, 0.95); background-color: transparent; border: none;")
        layout.addWidget(self.main_text)

        # 페이드 효과
        self.opacity = QGraphicsOpacityEffect(self.main_text)
        self.main_text.setGraphicsEffect(self.opacity)
        self.fade_anim = QPropertyAnimation(self.opacity, b"opacity", self)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.fade_anim.setStartValue(0.55)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.setDuration(900)
        self.fade_anim.setLoopCount(-1)

        # 소스 칩 (전면/후면)
        self.source_chip = QLabel()
        self.source_chip.setFont(QFont("Noto Sans KR", 24, QFont.Weight.Medium))
        self.source_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_chip.setStyleSheet("color: rgba(255, 51, 51, 0.9); background-color: transparent;")
        layout.addWidget(self.source_chip)

        # 안전 문구
        self.safety_text = QLabel("주변을 확인하고 양보하세요")
        self.safety_text.setFont(QFont("Noto Sans KR", 20))
        self.safety_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.safety_text.setStyleSheet("color: #EAF2FF; background-color: transparent;")
        layout.addWidget(self.safety_text)

        # 애니메이션 상태
        self.current_arrow_type = None
        self.frame_idx = 0

        # 애니메이션 타이머
        self.anim_timer = QTimer(self)
        self.anim_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.anim_timer.timeout.connect(self._tick)

        # 프레임 패턴
        self.frames = {
            'left': [
                "      <", "     <  ", "    <   ", "   <    ",
                "  <    ", " <     ", "<      ", "       ",
            ],
            'right': [
                ">     ", " >    ", "  >   ", "   >  ",
                "    > ", "     >", "      ", "      ",
            ],
            'stop': [
                "STOP🚨", "     ", "STOP🚨", "     ",
            ],
            'keep': [
                "⚠️", "     ", "⚠️", "     ",
            ],
        }

    def update_display(self, arrow_text, main_text, source_chip, confidence):
        """디스플레이 업데이트"""
        self.source_chip.setText(source_chip)
        self.start_arrow_animation(arrow_text)
        self.show()

    def start_arrow_animation(self, arrow_text):
        """화살표 애니메이션 시작"""
        # 타입 결정
        if arrow_text == '<':
            arrow_type = 'left'
        elif arrow_text == '>':
            arrow_type = 'right'
        elif arrow_text.upper() == 'STOP':
            arrow_type = 'stop'
        elif arrow_text.upper() == 'KEEP':
            arrow_type = 'keep'
        else:
            arrow_type = 'right'

        # 상태가 바뀌면 재시작
        self.current_arrow_type = arrow_type
        self.frame_idx = 0

        # 타이머/페이드 설정
        self.anim_timer.stop()
        self.fade_anim.stop()

        if arrow_type == 'stop':
            # STOP: 빠른 점멸
            self.main_text.setStyleSheet("color: rgba(255, 51, 51, 1.0); background-color: transparent;")
            self.anim_timer.start(500)
        elif arrow_type == 'keep':
            # KEEP: 부드러운 점멸 (녹색)
            self.main_text.setStyleSheet("color: rgba(76, 175, 80, 1.0); background-color: transparent;")
            self.anim_timer.start(600)
        else:
            # 좌/우: 부드러운 이동 + 페이드
            self.main_text.setStyleSheet("color: rgba(255, 51, 51, 0.95); background-color: transparent;")
            self.anim_timer.start(100)
            self.fade_anim.start()

        # 첫 프레임 즉시 반영
        self._render_frame()

    def _tick(self):
        """애니메이션 틱"""
        self.frame_idx = (self.frame_idx + 1) % len(self.frames[self.current_arrow_type])
        self._render_frame()

    def _render_frame(self):
        """프레임 렌더링"""
        arr = self.frames.get(self.current_arrow_type, self.frames['right'])
        text = arr[self.frame_idx]

        # STOP/KEEP일 때는 굵게
        if self.current_arrow_type in ['stop', 'keep']:
            self.main_text.setFont(QFont("DejaVu Sans Mono", 160, QFont.Weight.Black))
        else:
            self.main_text.setFont(QFont("DejaVu Sans Mono", 150, QFont.Weight.Black))

        self.main_text.setText(text)

    def stop_animations(self):
        """모든 애니메이션 중지"""
        self.anim_timer.stop()
        self.fade_anim.stop()
        self.main_text.setText("")
        self.current_arrow_type = None
        self.frame_idx = 0
        self.opacity.setOpacity(1.0)
        print("All animations stopped and cleared")
    
    def keyPressEvent(self, event):
        """키보드 이벤트를 부모 윈도우로 전달"""
        if self.parent():
            self.parent().keyPressEvent(event)
        else:
            super().keyPressEvent(event)


class EmergencyHUDApp(QMainWindow):
    """긴급차량 HUD 애플리케이션"""
    
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        
        # 윈도우 설정
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(0, 0, 1920, 1080)
        
        # UI 위젯 설정
        self.central_widget = EmergencyHUDWidget(self)
        self.setCentralWidget(self.central_widget)
        
        # 키보드 포커스 설정
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

    def keyPressEvent(self, event: QKeyEvent):
        """키보드 입력 처리"""
        if event.key() == Qt.Key.Key_Escape:
            self.ros_node.get_logger().info("ESC - Closing HUD")
            self.close()
        elif event.key() == Qt.Key.Key_A:
            self.ros_node.send_test_message("left")
        elif event.key() == Qt.Key.Key_D:
            self.ros_node.send_test_message("right")
        elif event.key() == Qt.Key.Key_W:
            self.ros_node.send_test_message("keep")
        elif event.key() == Qt.Key.Key_S:
            self.ros_node.send_test_message("stop")
        elif event.key() == Qt.Key.Key_Space:
            self.ros_node.send_test_message("none")


class EmergencyHUDNode(Node):
    """긴급차량 HUD 노드"""
    
    def __init__(self):
        super().__init__('emergency_hud')
        
        # GUI 모드 확인
        self.gui_mode = self.check_gui_mode()
        self.gui_app = None
        self.gui_widget = None
        
        # 설정 로드
        self.config = self.load_config()
        self.i18n = self.load_i18n()
        
        # 상태 변수
        self.current_state = "Idle"
        self.last_message_time = 0
        self.debounce_ms = 100  # 100ms로 단축 (반응 속도 개선)
        
        # 수신된 데이터 저장
        self.sound_direction = None  # front/back
        self.decision = None  # Right/Left/Stop/Caution/None
        
        # ROS2 구독자 설정
        self.setup_subscribers()
        
        # GUI 초기화 (GUI 모드인 경우)
        if self.gui_mode:
            self.init_gui()
        
        self.get_logger().info(f'Emergency HUD Node started (GUI: {self.gui_mode})')

    def check_gui_mode(self):
        """GUI 모드 확인"""
        # 환경 변수로 GUI 모드 강제 설정 가능
        if os.environ.get('HUD_NO_GUI', '').lower() in ['true', '1', 'yes']:
            return False
        
        # PyQt6 사용 가능 여부 확인
        if not PYQT6_AVAILABLE:
            return False
        
        # Qt 플랫폼 플러그인 사용 가능 여부 확인
        try:
            app = QApplication([])
            app.quit()
            return True
        except Exception as e:
            self.get_logger().warn(f"GUI mode disabled due to: {e}")
            return False

    def init_gui(self):
        """GUI 초기화"""
        try:
            self.gui_app = QApplication.instance()
            if self.gui_app is None:
                self.gui_app = QApplication([])
            
            self.gui_widget = EmergencyHUDApp(self)
            self.gui_widget.showFullScreen()
            
            # ROS2 타이머 설정
            self.ros_timer = QTimer()
            self.ros_timer.timeout.connect(self.spin_ros)
            self.ros_timer.start(10)  # 100Hz
            
        except Exception as e:
            self.get_logger().error(f"GUI initialization failed: {e}")
            self.gui_mode = False

    def load_config(self):
        """설정 파일 로드"""
        config_paths = [
            './config/config.json',
            '../config/config.json',
            os.path.join(os.path.dirname(__file__), '../config/config.json'),
            '/home/hyomin/mose_ws/src/emergency_hud/config/config.json'
        ]
        
        for path in config_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.get_logger().info(f'Config loaded from: {path}')
                    return config
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        
        self.get_logger().warn('Config load failed, using defaults')
        return {
            "theme": "dark",
            "low_perf_mode": False,
            "animation_intensity": 1.0,
            "window_position": [0, 0],
            "margins": [20, 20, 20, 20],
            "language": "ko"
        }

    def load_i18n(self):
        """국제화 파일 로드"""
        default_i18n = {
            "ko": {
                "left_move": "← 왼쪽으로 이동",
                "right_move": "→ 오른쪽으로 이동",
                "stay_straight": "직진 유지",
                "stop_yield": "정지 후 양보",
                "keep_lane": "✓ 올바른 차선입니다",
                "front_approach": "전면에서 긴급 차량 접근",
                "rear_approach": "후면에서 긴급 차량 접근",
                "safety": "주변을 확인하고 양보하세요",
                "safety_keep": "올바른 차선입니다."
            }
        }
        
        i18n_paths = [
            './config/i18n.json',
            '../config/i18n.json',
            os.path.join(os.path.dirname(__file__), '../config/i18n.json'),
            '/home/hyomin/mose_ws/src/emergency_hud/config/i18n.json'
        ]
        
        for path in i18n_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)['ko']
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        
        self.get_logger().warn('i18n load failed, using defaults')
        return default_i18n['ko']

    def setup_subscribers(self):
        """
        ROS2 구독자 설정
        
        구독 토픽:
        1. sound_direction - 소리 방향 (String 타입)
           값: "front" (앞) 또는 "back" (뒤)
        
        2. decision - 최종 결정 (String 타입)
           값: "Right" / "Left" / "Stop" / "Caution" / "None"
        """
        # 소리 방향 구독 (front/back)
        self.sound_direction_sub = self.create_subscription(
            String,
            'sound_direction',
            self.sound_direction_callback,
            10
        )
        
        # 최종 결정 구독 (Right/Left/Stop/Caution/None)
        self.decision_sub = self.create_subscription(
            String,
            'decision',
            self.decision_callback,
            10
        )
        
        self.get_logger().info('구독 완료: sound_direction, decision')

    def sound_direction_callback(self, msg):
        """
        소리 방향 콜백
        
        토픽: sound_direction
        타입: String
        값: "front" 또는 "back"
        """
        direction = msg.data.strip().lower()
        if direction in ['front', 'back']:
            self.sound_direction = direction
            self.get_logger().info(f'📍 소리 방향: {direction}')
            self.update_display()
        else:
            self.get_logger().warn(f'잘못된 sound_direction 값: {msg.data}')

    def decision_callback(self, msg):
        """
        최종 결정 콜백
        
        토픽: decision
        타입: String
        값: "Right" / "Left" / "Stop" / "Caution" / "None"
        """
        decision = msg.data.strip()
        
        # 대소문자 정규화
        decision_lower = decision.lower()
        if decision_lower in ['right', 'left', 'stop', 'caution', 'none']:
            self.decision = decision_lower.capitalize()
            self.get_logger().info(f'🎯 결정: {self.decision}')
            self.update_display()
        else:
            self.get_logger().warn(f'잘못된 decision 값: {msg.data}')

    def send_test_message(self, direction):
        """테스트 메시지 전송 (키보드 입력용)"""
        # 키보드 입력 매핑: A=Left, D=Right, W=Caution, S=Stop, Space=None
        decision_map = {
            "left": "Left",
            "right": "Right",
            "stop": "Stop",
            "keep": "Caution",
            "none": "None"
        }
        
        self.sound_direction = "front"
        self.decision = decision_map.get(direction, "None")
        self.update_display()

    def update_display(self):
        """
        화면 업데이트
        
        sound_direction과 decision 값에 따라 HUD를 업데이트합니다.
        """
        # 두 값이 모두 설정되어 있는지 확인
        if self.sound_direction is None or self.decision is None:
            return
        
        current_time = time.time() * 1000
        
        # None인 경우 화면 초기화 (debounce 무시 - 즉시 반응)
        if self.decision == "None":
            self.get_logger().info("🚨 알림 해제 (즉시 처리)")
            if self.gui_mode and self.gui_widget:
                self.gui_widget.central_widget.stop_animations()
                self.gui_widget.central_widget.source_chip.setText("")
                self.gui_widget.central_widget.safety_text.setText("")
            self.current_state = "Idle"
            self.sound_direction = None  # 초기화
            self.decision = None  # 초기화
            self.last_message_time = 0  # 타이머 리셋
            return
        
        # 새로운 상태 계산
        new_state = f"Alert{self.sound_direction.capitalize()}{self.decision}"
        
        # 상태가 바뀌면 debounce 무시 (즉시 반응)
        if new_state != self.current_state:
            self.get_logger().info(f"🔄 상태 변경 감지: {self.current_state} → {new_state}")
            self.transition_to(new_state, self.sound_direction, self.decision)
            self.last_message_time = current_time
            return
        
        # 동일 상태일 때만 debounce 적용 (중복 메시지 방지)
        if current_time - self.last_message_time < self.debounce_ms:
            self.get_logger().debug(f"⏳ 동일 상태 - Debounce 중... ({int(current_time - self.last_message_time)}ms)")
            return
        
        # 동일 상태 유지
        self.get_logger().debug(f"⚠️ 동일 상태 유지: {new_state}")
        self.last_message_time = current_time

    def transition_to(self, state, source, decision):
        """
        상태 전환 및 HUD 화면 업데이트
        
        Args:
            state: 새로운 상태 (예: "AlertFrontLeft")
            source: 소리 방향
                - "front": 앞
                - "back": 뒤
            decision: 최종 결정
                - "Left": 왼쪽으로 이동 → '<' 표시
                - "Right": 오른쪽으로 이동 → '>' 표시
                - "Stop": 정지 후 양보 → 'STOP' 표시
                - "Caution": 주의 (차선 유지) → 'KEEP' 표시 (녹색)
        """
        self.current_state = state
        
        # 결정에 따른 화면 표시 텍스트 결정
        decision_lower = decision.lower()
        action_display = {
            'left': '<',       # 왼쪽 이동
            'right': '>',      # 오른쪽 이동
            'stop': 'STOP',    # 정지
            'caution': 'KEEP', # 주의 (차선 유지)
        }.get(decision_lower, '>')

        # 소리 방향 메시지
        if source == 'front':
            direction_msg = self.i18n.get('front_approach', '전면에서 긴급 차량 접근')
        else:  # back
            direction_msg = self.i18n.get('rear_approach', '후면에서 긴급 차량 접근')
        
        # 안전 문구 선택
        if decision_lower == 'caution':
            safety_text = self.i18n.get('safety_keep', '올바른 차선입니다.')
        else:
            safety_text = self.i18n.get('safety', '주변을 확인하고 양보하세요')
        
        # 콘솔 출력 (디버깅용)
        print(f"\n{'='*50}")
        print(f"🚨 긴급차량 경고")
        print(f"   소리 방향: {source} ({'앞' if source == 'front' else '뒤'})")
        print(f"   결정: {decision} → {action_display}")
        print(f"   상태: {state}")
        print(f"   안내: {safety_text}")
        print(f"{'='*50}\n")
        
        # GUI 화면 업데이트
        if self.gui_mode and self.gui_widget:
            self.gui_widget.central_widget.update_display(action_display, action_display, direction_msg, 0.9)
            self.gui_widget.central_widget.safety_text.setText(safety_text)

    def spin_ros(self):
        """ROS2 스핀 (GUI 모드에서만 사용)"""
        try:
            if rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.001)
            else:
                self.get_logger().warn("ROS2 context is not valid, stopping timer")
                if hasattr(self, 'ros_timer'):
                    self.ros_timer.stop()
        except Exception as e:
            self.get_logger().error(f"Error in spin_ros: {e}")
            if hasattr(self, 'ros_timer'):
                self.ros_timer.stop()


def main(args=None):
    """메인 함수"""
    try:
        rclpy.init(args=args)
        node = EmergencyHUDNode()
        
        if node.gui_mode:
            # GUI 모드: PyQt 이벤트 루프 실행
            try:
                node.gui_app.exec()
            except KeyboardInterrupt:
                print("Emergency HUD GUI stopped by user")
            finally:
                node.destroy_node()
                rclpy.shutdown()
        else:
            # 콘솔 모드: ROS2 스핀 실행
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                print("Emergency HUD node stopped by user")
            finally:
                node.destroy_node()
                rclpy.shutdown()
            
    except Exception as e:
        print(f"Error starting Emergency HUD node: {e}")
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
