# Velocity Package

시리얼 통신으로 아날로그 값(350~740)을 읽어서 0~60 범위의 정수로 변환하여 velocity 토픽으로 발행하는 ROS2 패키지입니다.

## 기능

- 시리얼 포트에서 아날로그 값 읽기 (350~740 범위)
- 읽은 값을 속도 값(0~60)으로 선형 변환
- `velocity` 토픽으로 Int32 메시지 발행
- 실시간 속도 시각화 (velocity_visualizer.py)

## 설치

### 의존성 설치

```bash
pip3 install pyserial matplotlib
```

### 빌드

```bash
cd ~/ros2_ws
colcon build --packages-select velocity_pkg
source install/setup.bash
```

## 사용법

### 기본 실행

```bash
ros2 run velocity_pkg velocity_node
```

### 파라미터와 함께 실행

```bash
ros2 run velocity_pkg velocity_node \
  --ros-args \
  -p serial_port:=/dev/ttyUSB0 \
  -p baud_rate:=9600 \
  -p publish_rate:=20.0
```

윈도우의 경우:
```bash
ros2 run velocity_pkg velocity_node \
  --ros-args \
  -p serial_port:=COM3 \
  -p baud_rate:=9600
```

## 파라미터

| 파라미터명 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `serial_port` | string | `/dev/ttyUSB0` | 시리얼 포트 경로 (윈도우: COM3, COM4 등) |
| `baud_rate` | int | 9600 | 시리얼 통신 속도 |
| `publish_rate` | double | 20.0 | 토픽 발행 주기 (Hz) |
| `min_analog` | int | 350 | 최소 아날로그 값 |
| `max_analog` | int | 740 | 최대 아날로그 값 |
| `min_velocity` | int | 0 | 최소 속도 값 |
| `max_velocity` | int | 60 | 최대 속도 값 |

## 토픽

### Published Topics

- `velocity` (`std_msgs/Int32`)
  - 변환된 속도 값 (0~60)

## 시리얼 데이터 형식

시리얼 포트로부터 다음과 같은 형식의 데이터를 받습니다:

```
350\n
450\n
550\n
650\n
740\n
...
```

각 값은 줄바꿈(`\n`)으로 구분되며, 350~740 범위의 정수여야 합니다.

## 속도 시각화

velocity 토픽을 실시간으로 그래프로 시각화하는 스크립트가 포함되어 있습니다.

### 시각화 실행

```bash
# velocity_node가 실행 중인 상태에서 새 터미널을 열어 실행
python3 velocity_visualizer.py
```

또는 패키지 디렉토리에서:

```bash
cd src/velocity_pkg
python3 velocity_visualizer.py
```

### 시각화 기능

- 📈 **실시간 그래프**: 최근 100개 데이터 포인트 표시
- 🏎️ **속도계**: 현재 속도를 바 형태로 표시 (색상 변화)
  - 🟢 0-19: 녹색 (Low)
  - 🟡 20-39: 노란색 (Medium)
  - 🔴 40-60: 빨간색 (High)
- 📊 **통계 정보**: 현재/평균/최대/최소 속도 표시

## 테스트

### 속도 값 확인

```bash
ros2 topic echo /velocity
```

### 시리얼 포트 확인

리눅스:
```bash
ls -l /dev/ttyUSB*
ls -l /dev/ttyACM*
```

윈도우:
```powershell
mode
```

또는 노드를 실행하면 사용 가능한 포트 목록이 표시됩니다.

## 문제 해결

### 시리얼 포트 권한 오류 (리눅스)

```bash
sudo usermod -a -G dialout $USER
# 로그아웃 후 다시 로그인
```

또는 임시로:
```bash
sudo chmod 666 /dev/ttyUSB0
```

### 시리얼 포트를 찾을 수 없음

노드를 실행하면 사용 가능한 포트 목록이 출력됩니다. 해당 포트명으로 파라미터를 설정하세요.

## 전체 시스템 실행 예제

### 1. velocity_node 실행 (터미널 1)

윈도우:
```bash
ros2 run velocity_pkg velocity_node --ros-args -p serial_port:=COM3
```

리눅스:
```bash
ros2 run velocity_pkg velocity_node --ros-args -p serial_port:=/dev/ttyUSB0
```

### 2. 시각화 실행 (터미널 2)

```bash
cd src/velocity_pkg
python3 velocity_visualizer.py
```

그래프 창이 열리면서 실시간으로 속도가 시각화됩니다!

## 예제: 아두이노 코드

다음은 아두이노에서 아날로그 값을 시리얼로 전송하는 예제 코드입니다:

```cpp
void setup() {
  Serial.begin(9600);
}

void loop() {
  int analogValue = analogRead(A0);  // 0~1023 값 읽기
  
  // 0~1023을 350~740으로 매핑
  int mappedValue = map(analogValue, 0, 1023, 350, 740);
  
  Serial.println(mappedValue);
  delay(50);  // 20Hz
}
```

## 데모/테스트용 더미 데이터 발행

시리얼 장치 없이 테스트하려면:

```bash
# 터미널 1: 더미 데이터 발행
ros2 topic pub /velocity std_msgs/Int32 "data: 30" -r 10

# 터미널 2: 시각화
python3 velocity_visualizer.py
```

또는 속도를 변경하면서 테스트:

```bash
ros2 topic pub /velocity std_msgs/Int32 "data: 50" -1
```

