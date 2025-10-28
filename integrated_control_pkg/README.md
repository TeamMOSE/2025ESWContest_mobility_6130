# Integrated Control Package

이 패키지는 소방차 사이렌 감지 시 차량 방향 유도를 위한 통합 제어 시스템입니다.

## 개요

소방차, 구급차 등 긴급차량의 사이렌 소리를 감지하고, 카메라 영상 분석을 통해 차량의 안전한 비켜가기를 지원하는 자율주행 보조 시스템입니다.

## 기능

### 입력 정보 수집

1. **사이렌 감지** (`siren_pkg`)
   - `siren_detected` (String): 탐지 여부 (SIREN/NOSIREN)
   - `sound_direction` (String): 방향 (front/back)

2. **교차로 감지** (전방 카메라)
   - `intersection` (Bool): 교차로 감지 여부 (신호등 추적)

3. **소방차 감지** (후방 카메라)
   - `firetruck_side` (String): 후방 소방차 위치 (Rear_Right/Rear_Left/None)

4. **차선 감지** (전방 카메라)
   - `lane/info` (Int32MultiArray): [전체 차선 개수, 현재 차선 번호]

5. **속도 정보**
   - `velocity` (Int32): 현재 차량 속도 (정수)

### 출력 제어

- `decision` (String): 조향 결정 (Left/Right/Stop/None/Caution)
- **시리얼 통신**: Left, Right 명령 시 시리얼 포트로 "LEFT", "RIGHT" 전송 (1회)

## 토픽 인터페이스

### Subscribed Topics
- `siren_detected` (String): 사이렌 감지 여부 (SIREN/NOSIREN)
- `sound_direction` (String): 사이렌 방향 (front/back)
- `intersection` (Bool): 교차로 감지 여부
- `firetruck_side` (String): 후방 소방차 위치 (Rear_Right/Rear_Left/None)
- `lane/info` (Int32MultiArray): [전체 차선 개수, 현재 차선 번호]
- `velocity` (Int32): 현재 차량 속도

### Published Topics
- `decision` (String): 조향 결정 (Left/Right/Stop/None/Caution)
- **시리얼 통신** (UART): Left, Right 명령 발행 시 시리얼로 "LEFT", "RIGHT" 전송

### 파라미터
- `serial_port` (String, default: "COM3"): 시리얼 포트 경로
- `serial_baudrate` (Int, default: 9600): 시리얼 통신 속도
- `serial_timeout` (Float, default: 1.0): 시리얼 타임아웃 (초)

## 제어 로직 (플로우차트 기반)

### 1. 사이렌 감지 판별

- **NOSIREN**: 사이렌이 감지되지 않음 → 제어 없음 (None)
- **SIREN**: 사이렌 감지됨 → `sound_direction`으로 방향 판별

### 2. 전방 사이렌 (front)

```
사이렌이 전방에서 감지될 경우:
  - 교차로인가?
    - 예: Stop (정지)
    - 아니오: None (제어 없음)
```

### 3. 후방 사이렌 (back)

#### 주행 중 (velocity >= 30 km/h)

```
1. 교차로인가?
   - 예: Right (우측 정렬)
   - 아니오: 차선 수에 따른 사이드 정렬
   
차선 수에 따른 사이드 정렬:
  - 편도 1차선: Right
  - 편도 2차선:
    * 현재 1차선: Right
    * 현재 2차선: Caution (이미 최우측)
  - 편도 3차선 이상:
    * 끝차선 (1차선 또는 최대차선): Caution
    * 가운데 차선 (홀수 차선에서): Left/Right 랜덤
    * 그 외: 끝쪽으로 유도
      - 1차선쪽에 가까움: Right
      - 최대차선쪽에 가까움: Left
      
예시:
  - 3차선: 1차선=Caution, 2차선=Left/Right랜덤, 3차선=Caution
  - 4차선: 1차선=Caution, 2차선=Left, 3차선=Right, 4차선=Caution
  - 5차선: 1차선=Caution, 2차선=Right, 3차선=Left/Right랜덤, 4차선=Left, 5차선=Caution
```

#### 서행 중 (velocity < 30 km/h)

```
1. 소방차가 보이는가? (firetruck_side != None)
   - 예: 반대 방향으로 회피
     * Rear_Left 감지 → Right
     * Rear_Right 감지 → Left
   
   - 아니오:
     a) 교차로인가?
        - 예: Right
        
     b) 아니오: 차선 수에 따른 끝쪽 유도
        - 편도 1차선: Right
        - 편도 2차선:
          * 1차선: Left (왼쪽 끝으로)
          * 2차선: Right (오른쪽 끝으로)
        - 편도 3차선 이상:
          * 1차선쪽에 가까움: Right (더 왼쪽으로)
          * 최대차선쪽에 가까움: Left (더 오른쪽으로)
          * 가운데: Left/Right 랜덤
```

## 실행 방법

### 패키지 빌드

```bash
cd ~/ros2_ws
colcon build --packages-select integrated_control_pkg
source install/setup.bash
```

### 노드 단독 실행

```bash
# 기본 실행 (시리얼 포트: COM3)
ros2 run integrated_control_pkg integrated_control_node

# 시리얼 포트 지정하여 실행
ros2 run integrated_control_pkg integrated_control_node --ros-args \
  -p serial_port:=/dev/ttyUSB0 \
  -p serial_baudrate:=115200
```

### Launch 파일로 전체 시스템 실행

```bash
ros2 launch integrated_control_pkg integrated_control.launch.py
```

### 파라미터 설정과 함께 실행

```bash
ros2 launch integrated_control_pkg integrated_control.launch.py \
  firetruck_timeout_sec:=5.0 \
  control_frequency_hz:=20.0 \
  min_speed_for_action_kmh:=10.0
```

## 디버깅

### 제어 명령 확인

```bash
ros2 topic echo /decision
```

### 입력 토픽 확인

```bash
# 사이렌 감지
ros2 topic echo /siren_detected
ros2 topic echo /sound_direction

# 교차로 감지
ros2 topic echo /intersection

# 소방차 위치
ros2 topic echo /firetruck_side

# 차선 정보
ros2 topic echo /lane/info

# 속도
ros2 topic echo /velocity
```

### 노드 상태 확인

```bash
ros2 node info /integrated_control_node
```

### 시리얼 통신 확인

노드 실행 시 로그에서 시리얼 통신 상태를 확인할 수 있습니다:

```
[INFO] [integrated_control_node]: 시리얼 포트 연결 성공: COM3 @ 9600 baud
[INFO] [integrated_control_node]: 결정 명령: Right
[INFO] [integrated_control_node]: 시리얼 전송: RIGHT
```

시리얼 포트 연결이 실패해도 노드는 정상 동작하며, 경고 메시지만 출력됩니다:

```
[WARN] [integrated_control_node]: 시리얼 포트 연결 실패: [Errno 2] could not open port COM3
[WARN] [integrated_control_node]: 시리얼 통신 없이 계속 실행합니다.
```

## 테스트

### 수동 토픽 발행으로 테스트

#### 테스트 1: 후방 사이렌 + 주행 중 + 편도 3차선 2차선 주행
```bash
ros2 topic pub /siren_detected std_msgs/String "data: 'SIREN'" -1
ros2 topic pub /sound_direction std_msgs/String "data: 'back'" -1
ros2 topic pub /velocity std_msgs/Int32 "data: 50" -1
ros2 topic pub /intersection std_msgs/Bool "data: false" -1
ros2 topic pub /lane/info std_msgs/Int32MultiArray "data: [3, 2]" -1

# 예상 출력: Left 또는 Right (랜덤, 가운데 차선)
```

#### 테스트 2: 후방 사이렌 + 서행 중 + 소방차 좌측 감지
```bash
ros2 topic pub /siren_detected std_msgs/String "data: 'SIREN'" -1
ros2 topic pub /sound_direction std_msgs/String "data: 'back'" -1
ros2 topic pub /velocity std_msgs/Int32 "data: 20" -1
ros2 topic pub /firetruck_side std_msgs/String "data: 'Rear_Left'" -1

# 예상 출력: Right (소방차 반대 방향)
```

#### 테스트 3: 후방 사이렌 + 서행 중 + 소방차 안 보임 + 4차선 중 2차선
```bash
ros2 topic pub /siren_detected std_msgs/String "data: 'SIREN'" -1
ros2 topic pub /sound_direction std_msgs/String "data: 'back'" -1
ros2 topic pub /velocity std_msgs/Int32 "data: 15" -1
ros2 topic pub /firetruck_side std_msgs/String "data: 'None'" -1
ros2 topic pub /intersection std_msgs/Bool "data: false" -1
ros2 topic pub /lane/info std_msgs/Int32MultiArray "data: [4, 2]" -1

# 예상 출력: Right (1차선쪽에 가까우므로 더 왼쪽으로)
```

#### 테스트 4: 전방 사이렌 + 교차로
```bash
ros2 topic pub /siren_detected std_msgs/String "data: 'SIREN'" -1
ros2 topic pub /sound_direction std_msgs/String "data: 'front'" -1
ros2 topic pub /intersection std_msgs/Bool "data: true" -1

# 예상 출력: Stop (정지)
```

#### 테스트 5: 후방 사이렌 + 주행 중 + 교차로
```bash
ros2 topic pub /siren_detected std_msgs/String "data: 'SIREN'" -1
ros2 topic pub /sound_direction std_msgs/String "data: 'back'" -1
ros2 topic pub /velocity std_msgs/Int32 "data: 60" -1
ros2 topic pub /intersection std_msgs/Bool "data: true" -1

# 예상 출력: Right (우측 정렬)
```

## 시리얼 통신 세부 사항

### 지원 플랫폼

- **Windows**: `COM1`, `COM3`, `COM5` 등
- **Linux**: `/dev/ttyUSB0`, `/dev/ttyACM0`, `/dev/ttyS0` 등
- **macOS**: `/dev/cu.usbserial-*`, `/dev/tty.usbserial-*` 등

### 전송 형식

- **명령어**: `LEFT\n` 또는 `RIGHT\n` (개행 문자 포함)
- **인코딩**: UTF-8
- **전송 조건**: Left, Right 토픽 발행 시 1회만 전송 (중복 방지)

### 중복 방지 메커니즘

노드는 이전에 전송한 명령을 기억하여, 같은 명령이 연속으로 발행될 때 시리얼로 중복 전송하지 않습니다.

예시:
```
결정 명령: Right → 시리얼 전송: "RIGHT\n"
결정 명령: Right → 시리얼 전송 안 함 (중복)
결정 명령: Left  → 시리얼 전송: "LEFT\n"
결정 명령: Left  → 시리얼 전송 안 함 (중복)
```

### 아두이노 예제 코드

```cpp
// Arduino Serial Receiver Example
String command = "";

void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "LEFT") {
      // 왼쪽 방향 처리
      Serial.println("Received: LEFT");
      // 모터 제어 등...
    }
    else if (command == "RIGHT") {
      // 오른쪽 방향 처리
      Serial.println("Received: RIGHT");
      // 모터 제어 등...
    }
  }
}
```