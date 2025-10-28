#!/usr/bin/env python3
"""
이미지 뷰어 노드 - /spectrogram_image/compressed 토픽을 구독하여 실시간으로 표시
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class ImageViewerNode(Node):
    def __init__(self):
        super().__init__('image_viewer_node')
        
        # QoS 설정 - BEST_EFFORT로 변경 (송신측과 동일)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 구독자 생성 - CompressedImage로 변경
        self.subscription = self.create_subscription(
            CompressedImage,
            '/spectrogram_image/compressed',
            self.image_callback,
            qos_profile
        )
        self.bridge = CvBridge()
        
        self.get_logger().info('Image Viewer 노드 시작 - /spectrogram_image/compressed 토픽 구독 중')
        
        # 통계 정보
        self.frame_count = 0
        self.last_log_time = time.time()
        self.last_receive_time = time.time()
        
        # OpenCV 창 생성
        self.window_name = 'Spectrogram Image Viewer'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 640)
        
        # 타이머 생성 - 주기적으로 창 상태 확인
        self.timer = self.create_timer(0.1, self.check_window)
        
        self.running = True

    def image_callback(self, msg):
        """이미지 메시지 콜백"""
        try:
            # CompressedImage 메시지를 OpenCV 이미지로 변환 (JPEG 압축 해제)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 타임스탬프 및 프레임 정보 표시
            current_time = time.time()
            elapsed = current_time - self.last_receive_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            
            # 이미지에 정보 오버레이
            display_image = cv_image.copy()
            
            # 배경 사각형 그리기 (텍스트 가독성 향상)
            overlay = display_image.copy()
            cv2.rectangle(overlay, (5, 5), (300, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_image, 0.4, 0, display_image)
            
            # 텍스트 정보 추가
            cv2.putText(display_image, f'FPS: {fps:.1f}', 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_image, f'Frame: {self.frame_count}', 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 이미지 표시
            cv2.imshow(self.window_name, display_image)
            cv2.waitKey(1)
            
            self.frame_count += 1
            self.last_receive_time = current_time
            
            # 1초마다 통계 출력
            if current_time - self.last_log_time >= 1.0:
                self.get_logger().info(f'수신 속도: {fps:.1f} FPS (Frame #{self.frame_count})')
                self.last_log_time = current_time
                
        except Exception as e:
            self.get_logger().error(f'이미지 처리 중 오류: {e}')

    def check_window(self):
        """OpenCV 창 상태 확인"""
        if not self.running:
            return
        
        # 창이 닫혔는지 확인
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            self.get_logger().info('창이 닫혔습니다. 노드를 종료합니다.')
            self.running = False
            rclpy.shutdown()

    def destroy_node(self):
        """노드 종료"""
        self.get_logger().info('Image Viewer 노드 종료 중...')
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    image_viewer_node = ImageViewerNode()
    
    try:
        rclpy.spin(image_viewer_node)
    except KeyboardInterrupt:
        image_viewer_node.get_logger().info('KeyboardInterrupt - 종료합니다.')
    except Exception as e:
        image_viewer_node.get_logger().error(f'오류 발생: {e}')
    finally:
        image_viewer_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()