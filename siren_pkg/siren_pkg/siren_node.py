#!/usr/bin/env python3
import numpy as np
import pyaudio
import threading
import time
import queue
import argparse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 없이 사용
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import cv2


class UnifiedAudioNode(Node):
    """
    방향 감지와 스펙트로그램 생성을 통합한 ROS2 노드
    - 두 마이크로부터 방향 감지 (앞/뒤)
    - 하나의 마이크로부터 스펙트로그램 생성
    """
    def __init__(self, front_card=1, back_card=3, spectrogram_card=1,
                sample_rate=48000, chunk_size=4096, threshold=0.01,
                smoothing_window=10, spectrogram_width=224, spectrogram_height=224,
                time_window=1.0):
        super().__init__('unified_audio_node')

        # 파라미터 설정
        self.front_card = front_card
        self.back_card = back_card
        self.spectrogram_card = spectrogram_card
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.direction_threshold = threshold
        self.smoothing_window = smoothing_window
        self.spectrogram_width = spectrogram_width
        self.spectrogram_height = spectrogram_height
        self.time_window = time_window

        self.baseline_rms_front = 0
        self.baseline_rms_back = 0

        # 오디오 설정 (모든 마이크가 16bit, 모노라고 하셨으니)
        self.audio_format = pyaudio.paInt16
        self.channels = 1

        # 오디오 버퍼
        self.direction_buffers = [queue.Queue(maxsize=200), queue.Queue(maxsize=200)]
        self.spectrogram_buffer = queue.Queue(maxsize=200)

        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.direction_streams = [None, None]
        self.spectrogram_stream = None

        # 스레드 제어
        self.running = False
        self.direction_thread = None
        self.spectrogram_thread = None

        # QoS 설정 - BEST_EFFORT (무선 통신 최적화)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ROS2 퍼블리셔
        self.direction_publisher = self.create_publisher(String, 'sound_direction', 10)
        self.spectrogram_publisher = self.create_publisher(
            CompressedImage, 
            '/spectrogram_image/compressed', 
            qos_profile
        )
        self.bridge = CvBridge()

        # JPEG 압축 품질 (무선 통신 최적화)
        self.jpeg_quality = 85

        self.current_direction = None

        # 스펙트로그램 초기화
        self.setup_spectrogram()

        self.get_logger().info("=== 통합 오디오 노드 초기화 ===")
        self.get_logger().info(f"앞 마이크 인덱스: {self.front_card}")
        self.get_logger().info(f"뒤 마이크 인덱스: {self.back_card}")
        self.get_logger().info(f"스펙트로그램 마이크 인덱스: {self.spectrogram_card}")

    def setup_spectrogram(self):
        """스펙트로그램 관련 초기화"""
        self.fft_size = self.chunk_size
        self.freq_bins = fftfreq(self.fft_size, 1/self.sample_rate)[:self.fft_size//2]

        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.ax.set_ylim(0, 10000)
        self.ax.axis('off')
        self.fig.tight_layout(pad=0)
        self.spectrogram_image = self.ax.imshow(
            np.zeros((100, 50)), aspect='auto', origin='lower', cmap='inferno'
        )
        self.spectrogram_data_display = []

    def start_audio_streams(self):
        """오디오 스트림 시작"""
        def create_direction_callback(buffer_index):
            def audio_callback(in_data, frame_count, time_info, status):
                # 16-bit 모노 데이터로 처리 (장치가 16bit로 들어옴)
                if self.running and in_data:
                    try:
                        samples_int16 = np.frombuffer(in_data, dtype=np.int16)
                        if self.channels > 1:
                            samples_int16 = samples_int16[::self.channels]  # 첫 번째 채널만 사용
                        samples_float = samples_int16.astype(np.float32) / 32768.0
                        # direction buffer에 넣기 (원래 목적)
                        if not self.direction_buffers[buffer_index].full():
                            self.direction_buffers[buffer_index].put(samples_float)
                    except Exception as e:
                        self.get_logger().error(f"direction callback 변환 오류: {e}")
                return (in_data, pyaudio.paContinue)
            return audio_callback

        def spectrogram_callback(in_data, frame_count, time_info, status):
            # 16-bit 처리 (스펙트로그램 전용)
            if self.running and in_data:
                try:
                    samples_int16 = np.frombuffer(in_data, dtype=np.int16)
                    if self.channels > 1:
                        samples_int16 = samples_int16[::self.channels]  # 첫 번째 채널만 사용
                    samples_float = samples_int16.astype(np.float32) / 32768.0
                    if not self.spectrogram_buffer.full():
                        self.spectrogram_buffer.put(samples_float)
                except Exception as e:
                    self.get_logger().error(f"spectrogram callback 변환 오류: {e}")
            return (in_data, pyaudio.paContinue)

        try:
            # 앞/뒤 스트림 열기 (모두 16bit, 모노)
            self.direction_streams[0] = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.front_card,
                frames_per_buffer=self.chunk_size,
                stream_callback=create_direction_callback(0)
            )

            self.direction_streams[1] = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.back_card,
                frames_per_buffer=self.chunk_size,
                stream_callback=create_direction_callback(1)
            )

            # 스펙트로그램용이 별도 카드이면 별도 스트림 생성
            if self.spectrogram_card not in [self.front_card, self.back_card]:
                self.spectrogram_stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.spectrogram_card,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=spectrogram_callback
                )
                self.spectrogram_stream.start_stream()
                self.get_logger().info("스펙트로그램용 별도 스트림 시작")
            else:
                self.get_logger().info("스펙트로그램이 앞 마이크 데이터를 공유합니다")

            self.direction_streams[0].start_stream()
            self.direction_streams[1].start_stream()

            self.get_logger().info("모든 오디오 스트림 시작 완료")
            return True

        except Exception as e:
            self.get_logger().error(f"오디오 스트림 시작 실패: {e}")
            return False

    def calibrate_baselines(self, duration=3):
        """소음 레벨 자동 보정"""
        self.get_logger().info(f"소음 레벨 측정 시작... {duration}초간 조용히 해주세요.")
        start_time = time.time()
        rms_values_front = []
        rms_values_back = []

        for buf in self.direction_buffers:
            while not buf.empty():
                buf.get()

        while time.time() - start_time < duration:
            try:
                audio_data_front = self.direction_buffers[0].get(timeout=1)
                audio_data_back = self.direction_buffers[1].get(timeout=1)
                rms_values_front.append(self.get_rms(audio_data_front))
                rms_values_back.append(self.get_rms(audio_data_back))
            except queue.Empty:
                break

        self.baseline_rms_front = np.mean(rms_values_front) if rms_values_front else 0
        self.baseline_rms_back = np.mean(rms_values_back) if rms_values_back else 0
        self.get_logger().info("소음 레벨 측정 완료")

    @staticmethod
    def get_rms(audio_data):
        return np.sqrt(np.mean(np.square(audio_data)))

    def calculate_spectrogram(self, audio_data):
        fft_data = fft(audio_data * np.hanning(len(audio_data)))
        magnitude = np.abs(fft_data[:len(fft_data)//2])
        return 20 * np.log10(magnitude + 1e-10)

    def publish_direction(self, direction):
        msg = String()
        msg.data = direction
        self.direction_publisher.publish(msg)
        self.get_logger().info(f"방향 발행: {direction}")
        self.current_direction = direction

    def publish_spectrogram(self, image):
        if image is not None:
            # JPEG 압축 (무선 통신 최적화: 150KB → 10~20KB)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            _, compressed_data = cv2.imencode('.jpg', image, encode_param)
            
            # CompressedImage 메시지 생성
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = self.get_clock().now().to_msg()
            compressed_msg.format = "jpeg"
            compressed_msg.data = compressed_data.tobytes()
            
            self.spectrogram_publisher.publish(compressed_msg)

    def direction_analysis_loop(self):
        self.get_logger().info("방향 감지 루프 시작")
        recent_differences = []

        while self.running:
            try:
                if self.direction_buffers[0].empty() or self.direction_buffers[1].empty():
                    time.sleep(0.01)
                    continue

                audio_data_front = self.direction_buffers[0].get()
                audio_data_back = self.direction_buffers[1].get()

                rms_front = self.get_rms(audio_data_front)
                rms_back = self.get_rms(audio_data_back)

                signal_front = max(0, rms_front - self.baseline_rms_front)
                signal_back = max(0, rms_back - self.baseline_rms_back)

                diff = signal_front - signal_back
                recent_differences.append(diff)
                if len(recent_differences) > self.smoothing_window:
                    recent_differences.pop(0)
                avg_diff = np.mean(recent_differences)

                direction = "front" if avg_diff >= self.direction_threshold else "back"
                self.publish_direction(direction)
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f"방향 분석 중 오류: {e}")
                time.sleep(0.1)

    def spectrogram_analysis_loop(self):
        self.get_logger().info("스펙트로그램 루프 시작")

        # 발행 주기 제어 (무선 통신 최적화: 100Hz → 10Hz)
        publish_interval = 0.1  # 10Hz
        last_publish_time = time.time()

        while self.running:
            try:
                current_time = time.time()
                
                if self.spectrogram_stream is not None:
                    if not self.spectrogram_buffer.empty():
                        audio_data = self.spectrogram_buffer.get()
                    else:
                        time.sleep(0.005)
                        continue
                else:
                    if self.direction_buffers[0].empty():
                        time.sleep(0.005)
                        continue
                    audio_data = self.direction_buffers[0].get()

                spectrogram = self.calculate_spectrogram(audio_data)
                self.spectrogram_data_display.append(spectrogram.copy())
                if len(self.spectrogram_data_display) > int(self.time_window * self.sample_rate / self.chunk_size):
                    self.spectrogram_data_display.pop(0)

                # 주기 제어: 0.1초마다만 발행 (무선 대역폭 절약)
                if len(self.spectrogram_data_display) >= 5 and (current_time - last_publish_time) >= publish_interval:
                    spec_array = np.array(self.spectrogram_data_display).T
                    self.spectrogram_image.set_data(spec_array)
                    # extent: x from 0..frames, y from 0..(sample_rate/2)
                    self.spectrogram_image.set_extent([0, spec_array.shape[1], 0, self.sample_rate / 2])
                    self.spectrogram_image.set_clim(vmin=-80, vmax=0)
                    self.fig.canvas.draw()
                    buf = self.fig.canvas.buffer_rgba()
                    image_array = np.asarray(buf)
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                    resized_image = cv2.resize(image_bgr, (self.spectrogram_width, self.spectrogram_height))
                    self.publish_spectrogram(resized_image)
                    last_publish_time = current_time
                    
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f"스펙트로그램 분석 중 오류: {e}")
                time.sleep(0.1)

    def start(self):
        if not self.start_audio_streams():
            return False
        self.calibrate_baselines()
        self.running = True
        self.direction_thread = threading.Thread(target=self.direction_analysis_loop, daemon=True)
        self.direction_thread.start()
        self.spectrogram_thread = threading.Thread(target=self.spectrogram_analysis_loop, daemon=True)
        self.spectrogram_thread.start()
        self.get_logger().info("통합 오디오 노드 시작 완료")
        self.get_logger().info("토픽: /sound_direction, /spectrogram_image/compressed")
        self.get_logger().info(f"무선 통신 최적화: JPEG 압축(품질={self.jpeg_quality}), 발행 주기 10Hz")
        return True

    def stop(self):
        self.running = False
        time.sleep(0.2)
        for stream in self.direction_streams:
            if stream:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
        if self.spectrogram_stream:
            try:
                if self.spectrogram_stream.is_active():
                    self.spectrogram_stream.stop_stream()
            except Exception:
                pass
            try:
                self.spectrogram_stream.close()
            except Exception:
                pass
        try:
            self.audio.terminate()
        except Exception:
            pass
        plt.close(self.fig)
        self.get_logger().info("노드 종료 완료")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--front-card', type=int, default=0)
    parser.add_argument('--back-card', type=int, default=2)
    parser.add_argument('--spectrogram-card', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.008)
    parser.add_argument('--smoothing', type=int, default=10)
    parser.add_argument('--spectrogram-width', type=int, default=224)
    parser.add_argument('--spectrogram-height', type=int, default=224)
    parser.add_argument('--time-window', type=float, default=1.0)
    parsed_args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = UnifiedAudioNode(
        front_card=parsed_args.front_card,
        back_card=parsed_args.back_card,
        spectrogram_card=parsed_args.spectrogram_card,
        threshold=parsed_args.threshold,
        smoothing_window=parsed_args.smoothing,
        spectrogram_width=parsed_args.spectrogram_width,
        spectrogram_height=parsed_args.spectrogram_height,
        time_window=parsed_args.time_window
    )
    try:
        if node.start():
            rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("중단됨")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
