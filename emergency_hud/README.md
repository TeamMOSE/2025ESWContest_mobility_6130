# Emergency HUD (긴급차량 HUD)

긴급차량 접근 시 운전자에게 시각적으로 행동 지침을 제공하는 HUD 시스템입니다.

## 토픽 구독

HUD는 두 개의 토픽을 구독합니다:

### 1. `sound_direction` - 소리 방향

**타입**: `std_msgs/String`  
**값**: 
- `"front"` : 앞에서 소리 들림
- `"back"` : 뒤에서 소리 들림

```bash
# 앞에서 소리
ros2 topic pub sound_direction std_msgs/msg/String "{data: 'front'}" --once

# 뒤에서 소리
ros2 topic pub sound_direction std_msgs/msg/String "{data: 'back'}" --once
```

### 2. `decision` - 최종 결정

**타입**: `std_msgs/String`  
**값**:
- `"Right"` : 오른쪽으로 이동
- `"Left"` : 왼쪽으로 이동
- `"Stop"` : 정지 후 양보
- `"Caution"` : 주의 (현재 차선 유지)
- `"None"` : 알림 해제

```bash
# 왼쪽 이동
ros2 topic pub decision std_msgs/msg/String "{data: 'Left'}" --once

# 오른쪽 이동
ros2 topic pub decision std_msgs/msg/String "{data: 'Right'}" --once

# 정지
ros2 topic pub decision std_msgs/msg/String "{data: 'Stop'}" --once

# 주의 (차선 유지)
ros2 topic pub decision std_msgs/msg/String "{data: 'Caution'}" --once

# 알림 해제
ros2 topic pub decision std_msgs/msg/String "{data: 'None'}" --once
```

## 화면 표시

### 1. 왼쪽 이동 (Left)
- **화면 표시**: `<` (왼쪽 화살표 애니메이션)
- **색상**: 빨간색
- **안내 문구**: "주변을 확인하고 양보하세요"

### 2. 오른쪽 이동 (Right)
- **화면 표시**: `>` (오른쪽 화살표 애니메이션)
- **색상**: 빨간색
- **안내 문구**: "주변을 확인하고 양보하세요"

### 3. 정지 (Stop)
- **화면 표시**: `STOP` (점멸)
- **색상**: 빨간색
- **안내 문구**: "주변을 확인하고 양보하세요"

### 4. 주의 - 차선 유지 (Caution)
- **화면 표시**: `KEEP` (부드러운 점멸)
- **색상**: 녹색
- **안내 문구**: "올바른 차선입니다."

### 5. 알림 해제 (None)
- **화면 표시**: 화면 초기화 (모든 애니메이션 중지)

## 사용 예시

### Python 코드로 메시지 발행

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class EmergencyPublisher(Node):
    def __init__(self):
        super().__init__('emergency_publisher')
        self.direction_pub = self.create_publisher(String, 'sound_direction', 10)
        self.decision_pub = self.create_publisher(String, 'decision', 10)
    
    def send_alert(self, direction, decision):
        """긴급차량 알림 발송"""
        # 소리 방향 발행
        dir_msg = String()
        dir_msg.data = direction  # "front" or "back"
        self.direction_pub.publish(dir_msg)
        
        # 결정 발행
        dec_msg = String()
        dec_msg.data = decision  # "Left", "Right", "Stop", "Caution", "None"
        self.decision_pub.publish(dec_msg)
        
        self.get_logger().info(f'발행: {direction} + {decision}')

# 사용 예시
def main():
    rclpy.init()
    publisher = EmergencyPublisher()
    
    # 1. 앞에서 접근 - 왼쪽으로 이동
    publisher.send_alert("front", "Left")
    
    # 2. 뒤에서 접근 - 오른쪽으로 이동
    publisher.send_alert("back", "Right")
    
    # 3. 앞에서 접근 - 정지
    publisher.send_alert("front", "Stop")
    
    # 4. 앞에서 접근 - 차선 유지
    publisher.send_alert("front", "Caution")
    
    # 5. 알림 해제
    publisher.send_alert("front", "None")
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 커맨드 라인에서 메시지 발행

```bash
# 예시 1: 앞에서 소리 + 왼쪽 이동
ros2 topic pub sound_direction std_msgs/msg/String "{data: 'front'}" --once
ros2 topic pub decision std_msgs/msg/String "{data: 'Left'}" --once

# 예시 2: 뒤에서 소리 + 오른쪽 이동
ros2 topic pub sound_direction std_msgs/msg/String "{data: 'back'}" --once
ros2 topic pub decision std_msgs/msg/String "{data: 'Right'}" --once

# 예시 3: 앞에서 소리 + 정지
ros2 topic pub sound_direction std_msgs/msg/String "{data: 'front'}" --once
ros2 topic pub decision std_msgs/msg/String "{data: 'Stop'}" --once

# 예시 4: 앞에서 소리 + 주의 (차선 유지)
ros2 topic pub sound_direction std_msgs/msg/String "{data: 'front'}" --once
ros2 topic pub decision std_msgs/msg/String "{data: 'Caution'}" --once

# 예시 5: 알림 해제
ros2 topic pub decision std_msgs/msg/String "{data: 'None'}" --once
```

## 키보드 테스트 단축키

HUD 실행 중 다음 키로 테스트 가능:

- `A`: 왼쪽 이동 (Left)
- `D`: 오른쪽 이동 (Right)
- `W`: 주의 - 차선 유지 (Caution)
- `S`: 정지 (Stop)
- `Space`: 알림 해제 (None)
- `ESC` 또는 `Q`: 종료

**참고**: 
- 키 입력이 작동하지 않으면 HUD 화면을 마우스로 한 번 클릭하세요
- 각 키 입력 시 터미널에 로그가 표시됩니다

## 실행 방법

```bash
# 빌드
cd /home/hyomin/mose_ws
colcon build --packages-select emergency_hud

# 소스
source install/setup.bash

# 실행
ros2 launch emergency_hud emergency_hud.launch.py

# 또는 직접 실행
ros2 run emergency_hud emergency_hud_node
```

## Wayland 환경 설정

투명 배경이 작동하지 않을 경우:

```bash
# X11 백엔드 사용 (권장)
ros2 launch emergency_hud emergency_hud.launch.py qt_platform:=xcb

# Wayland 네이티브 사용
ros2 launch emergency_hud emergency_hud.launch.py qt_platform:=wayland
```

## 토픽 구조 요약

| 토픽 | 타입 | 가능한 값 | 설명 |
|------|------|-----------|------|
| `sound_direction` | String | `"front"`, `"back"` | 소리 방향 |
| `decision` | String | `"Left"`, `"Right"`, `"Stop"`, `"Caution"`, `"None"` | 최종 결정 |

## 설정 파일

- `config/config.json`: HUD 설정
- `config/i18n.json`: 다국어 메시지
