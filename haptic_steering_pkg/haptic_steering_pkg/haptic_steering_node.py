import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Raspberry Pi 5 (Ubuntu 24.04) 호환: gpiozero + lgpio 사용
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory

# ===== Settings (fixed) =====
TOPIC_NAME    = '/haptic/command'  # 구독 토픽
ENA_PIN       = 18                 # PWM
IN1_PIN       = 23                 # 방향
IN2_PIN       = 24                 # 방향
PWM_HZ        = 1000               # PWM 주파수
SPEED_PERCENT = 50                 # 듀티(속도)
PULSE_SEC     = 0.2                # 각 스텝 유지 시간(초)

# ===== Motor HW =====
class MotorController:
    """L298N 단일 DC 모터 제어 (gpiozero + lgpio, Raspberry Pi 5 호환)"""
    def __init__(self):
        factory = LGPIOFactory()  # Pi 5에서 lgpio 백엔드 사용
        # PWMOutputDevice.value: 0.0 ~ 1.0 (소프트 PWM)
        self.ena = PWMOutputDevice(ENA_PIN, frequency=PWM_HZ, pin_factory=factory, initial_value=0.0)
        self.in1 = DigitalOutputDevice(IN1_PIN, pin_factory=factory, initial_value=False)
        self.in2 = DigitalOutputDevice(IN2_PIN, pin_factory=factory, initial_value=False)

    def _duty(self, percent: float) -> float:
        return max(0.0, min(1.0, percent / 100.0))

    def forward(self, speed=SPEED_PERCENT):
        self.in1.on()
        self.in2.off()
        self.ena.value = self._duty(speed)

    def backward(self, speed=SPEED_PERCENT):
        self.in1.off()
        self.in2.on()
        self.ena.value = self._duty(speed)

    def stop(self):
        self.ena.value = 0.0
        self.in1.off()
        self.in2.off()

    def close(self):
        # gpiozero 자원 해제
        try:
            self.stop()
        finally:
            self.ena.close()
            self.in1.close()
            self.in2.close()

# ===== ROS2 Node =====
class MotorDriverNode(Node):
    """
    /haptic/command(String) 구독 → LEFT/RIGHT 시퀀스만 수행
    - 간단한 로그만 출력
    - 실행 중 새 명령은 무시(busy)
    """
    def __init__(self):
        super().__init__('haptic_steering_node')
        self.motor = MotorController()
        self.running = False
        self.worker = None

        # 기본 QoS(depth=10)로 충분
        self.subscription = self.create_subscription(
            String, TOPIC_NAME, self.cmd_callback, 10
        )

        self.get_logger().info(
            f"ready: topic={TOPIC_NAME}, pins(ENA={ENA_PIN},IN1={IN1_PIN},IN2={IN2_PIN}), "
            f"speed={SPEED_PERCENT}%, pulse={PULSE_SEC}s, pwm={PWM_HZ}Hz (gpiozero+lgpio)"
        )

    def cmd_callback(self, msg: String):
        cmd = (msg.data or '').strip().upper()
        self.get_logger().info(f"cmd: {cmd}")

        if cmd in ('STOP', 'NONE', ''):
            self.motor.stop()
            return

        if cmd not in ('LEFT', 'RIGHT'):
            self.get_logger().info("unknown command, ignored")
            return

        if self.running:
            self.get_logger().info("busy, ignored")
            return

        self.worker = threading.Thread(target=self._run_sequence, args=(cmd,), daemon=True)
        self.running = True
        self.worker.start()

    def _run_sequence(self, cmd: str):
        try:
            if cmd == 'LEFT':
                steps = [self.motor.backward, self.motor.forward,
                         self.motor.backward, self.motor.forward]
            else:  # RIGHT
                steps = [self.motor.forward, self.motor.backward,
                         self.motor.forward, self.motor.backward]

            for step in steps:
                step(SPEED_PERCENT)
                time.sleep(PULSE_SEC)

            self.motor.stop()
            self.get_logger().info(f"{cmd} done")
        finally:
            self.running = False

    def destroy_node(self):
        try:
            self.motor.close()
        except Exception:
            pass
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MotorDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()