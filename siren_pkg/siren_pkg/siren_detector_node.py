#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage
import numpy as np
import os
import cv2
from ament_index_python.packages import get_package_share_directory

#siren_cnn.ipynb에서 정의한 모델과 동일한 구조
class SirenCNN(nn.Module):
    def __init__(self):
        super(SirenCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()

        self.fc_block = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2) # 2 classes: nosiren, siren
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x

class SirenDetectorNode(Node):
    def __init__(self):
        super().__init__('siren_detector_node')
        
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
            qos_profile)
        self.bridge = CvBridge()

        # String 타입으로 발행 (SIREN/NOSIREN)
        self.siren_detected_publisher = self.create_publisher(String, 'siren_detected', 10)
        
        # 모델 및 장치 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SirenCNN().to(self.device)
        
        # 모델 가중치 경로 설정 (중요: 실제 경로로 수정 필요)
        # 이 파일이 실행되는 위치 기준으로 siren_cnn_model.pth를 찾습니다.
        # 예: ros2_ws/src/siren_detector/siren_cnn_model.pth

        package_share_directory = get_package_share_directory('siren_pkg')
        
        model_path = os.path.join(package_share_directory, 'models', 'siren_cnn_model.pth')
        
        if not os.path.exists(model_path):
             self.get_logger().error(f"모델 파일 '{model_path}'을(를) 찾을 수 없습니다. 경로를 확인해주세요.")
             # rclpy.shutdown() # 더 강력한 종료를 원할 경우
             return
             
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.get_logger().info(f'Siren Detector 노드 시작, 모델 로드 완료. Device: {self.device}')
        
        # 추론을 위한 이미지 변환 (훈련시 val transform과 동일하게)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.classes = ('nosiren', 'siren')

    def image_callback(self, msg):
        try:
            # CompressedImage 메시지를 OpenCV 이미지로 변환 (JPEG 압축 해제)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # OpenCV 이미지(BGR)를 PIL 이미지(RGB)로 변환
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # 이미지 전처리 및 텐서 변환
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # 모델 추론
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                result = self.classes[predicted.item()]
                confidence_score = confidence.item()
            
            # 매번 탐지 결과 발행 (SIREN/NOSIREN 문자열로)
            siren_status = "SIREN" if (result == 'siren') else "NOSIREN"
            detected_msg = String()
            detected_msg.data = siren_status
            self.siren_detected_publisher.publish(detected_msg)
            
            # 결과 로그 출력 (신뢰도 포함)
            self.get_logger().info(f'탐지 결과: {siren_status} (신뢰도: {confidence_score:.2%})')
            
        except Exception as e:
            self.get_logger().error(f'이미지 처리 중 오류 발생: {e}')

def main(args=None):
    rclpy.init(args=args)
    siren_detector_node = SirenDetectorNode()
    try:
        rclpy.spin(siren_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        siren_detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
