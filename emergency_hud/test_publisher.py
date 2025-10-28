#!/usr/bin/env python3
"""
긴급차량 HUD 테스트 스크립트

sound_direction 및 decision 토픽으로 테스트 메시지를 발행합니다.

사용법:
    python3 test_publisher.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import sys


class EmergencyTestPublisher(Node):
    """긴급차량 알림 테스트 퍼블리셔"""
    
    def __init__(self):
        super().__init__('emergency_test_publisher')
        self.direction_pub = self.create_publisher(String, 'sound_direction', 10)
        self.decision_pub = self.create_publisher(String, 'decision', 10)
        self.get_logger().info('Emergency Test Publisher started')
    
    def send_alert(self, direction, decision):
        """
        긴급차량 알림 발송
        
        Args:
            direction: 소리 방향 ("front" or "back")
            decision: 최종 결정 ("Left", "Right", "Stop", "Caution", "None")
        """
        # 소리 방향 발행
        dir_msg = String()
        dir_msg.data = direction
        self.direction_pub.publish(dir_msg)
        
        # 결정 발행
        dec_msg = String()
        dec_msg.data = decision
        self.decision_pub.publish(dec_msg)
        
        direction_kr = "앞" if direction == "front" else "뒤"
        decision_map = {
            "Left": "왼쪽 이동",
            "Right": "오른쪽 이동",
            "Stop": "정지",
            "Caution": "차선 유지",
            "None": "알림 해제"
        }
        decision_kr = decision_map.get(decision, decision)
        
        self.get_logger().info(f'📢 발행: {direction_kr}에서 소리 → {decision_kr}')


def run_test_sequence(publisher):
    """테스트 시퀀스 실행"""
    print("\n" + "="*60)
    print("긴급차량 HUD 테스트 시작")
    print("="*60 + "\n")
    
    tests = [
        ("front", "Left", "테스트 1: 앞에서 소리 - 왼쪽 이동"),
        ("front", "None", "테스트 2: 앞에서 소리 - 알림 해제"),
        ("back", "Right", "테스트 3: 뒤에서 소리 - 오른쪽 이동"),
        ("front", "None", "테스트 4: 앞에서 소리 - 알림 해제"),
        ("back", "Caution", "테스트 5: 뒤에서 소리 - 차선 유지"),
        ("front", "Stop", "테스트 6: 앞에서 소리 - 정지"),
        ("front", "Caution", "테스트 7: 앞에서 소리 - 차선 유지 (주의)"),
    ]
    
    for direction, decision, description in tests:
        print(f"\n{description}")
        publisher.send_alert(direction, decision)
        time.sleep(4)
    
    print("\n테스트 5: 알림 해제")
    publisher.send_alert("front", "None")
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60 + "\n")


def interactive_mode(publisher):
    """대화형 모드"""
    print("\n" + "="*60)
    print("긴급차량 HUD 대화형 테스트")
    print("="*60)
    print("\n명령어:")
    print("  1. 방향 선택 (f=앞, b=뒤)")
    print("  2. 결정 선택 (l=왼쪽, r=오른쪽, s=정지, c=차선유지)")
    print("  예: 'fl' = 앞에서 소리, 왼쪽 이동")
    print("  예: 'br' = 뒤에서 소리, 오른쪽 이동")
    print("  'x' = 알림 해제")
    print("  'q' = 종료")
    print("="*60 + "\n")
    
    while True:
        try:
            cmd = input("명령 입력: ").lower().strip()
            
            if cmd == 'q':
                print("종료합니다.")
                break
            elif cmd == 'x':
                publisher.send_alert("front", "None")
            elif len(cmd) == 2:
                direction_map = {'f': 'front', 'b': 'back'}
                decision_map = {'l': 'Left', 'r': 'Right', 's': 'Stop', 'c': 'Caution'}
                
                direction = direction_map.get(cmd[0])
                decision = decision_map.get(cmd[1])
                
                if direction and decision:
                    publisher.send_alert(direction, decision)
                else:
                    print("❌ 잘못된 명령입니다.")
            else:
                print("❌ 잘못된 명령입니다.")
        
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


def main():
    """메인 함수"""
    rclpy.init()
    
    try:
        publisher = EmergencyTestPublisher()
        
        # 퍼블리셔 초기화 대기
        time.sleep(0.5)
        
        # 모드 선택
        if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
            # 대화형 모드
            interactive_mode(publisher)
        else:
            # 자동 테스트 시퀀스
            run_test_sequence(publisher)
        
        publisher.destroy_node()
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

