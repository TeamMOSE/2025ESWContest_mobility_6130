#!/usr/bin/env python3
"""
Velocity Visualizer
velocity 토픽을 구독하여 실시간으로 속도를 시각화하는 독립 스크립트
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import sys


class VelocityVisualizer:
    def __init__(self):
        # ROS2 초기화
        rclpy.init()
        
        # 임시 노드 생성
        self.node = rclpy.create_node('velocity_visualizer')
        
        # 데이터 저장용 deque (최대 100개 데이터 포인트)
        self.time_data = deque(maxlen=100)
        self.velocity_data = deque(maxlen=100)
        self.data_counter = 0
        
        # 현재 속도
        self.current_velocity = 0
        
        # Subscriber 생성
        self.velocity_sub = self.node.create_subscription(
            Int32,
            'velocity',
            self.velocity_callback,
            10
        )
        
        print("=" * 60)
        print("🚗 Velocity Visualizer 시작됨")
        print("=" * 60)
        print("velocity 토픽을 구독 중...")
        print("그래프 창을 닫으면 프로그램이 종료됩니다.")
        print("=" * 60)
        
        # 스레드로 ROS2 spin 실행
        self.running = True
        self.spin_thread = threading.Thread(target=self.spin_ros, daemon=True)
        self.spin_thread.start()
        
        # Matplotlib 설정 및 시작
        self.setup_plot()
    
    def velocity_callback(self, msg):
        """velocity 토픽 콜백"""
        self.current_velocity = msg.data
        
        # 데이터 추가
        self.data_counter += 1
        self.time_data.append(self.data_counter)
        self.velocity_data.append(msg.data)
    
    def spin_ros(self):
        """별도 스레드에서 ROS2 spin 실행"""
        while self.running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
    
    def setup_plot(self):
        """Matplotlib 그래프 설정"""
        # 다크 스타일 적용
        plt.style.use('dark_background')
        
        # Figure 및 Axes 생성
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Velocity Monitor')
        
        # 그리드 레이아웃
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. 실시간 그래프 (상단 전체)
        self.ax_graph = self.fig.add_subplot(gs[0:2, :])
        self.line, = self.ax_graph.plot([], [], 'cyan', linewidth=2.5, label='Velocity')
        self.ax_graph.fill_between([], [], alpha=0.3, color='cyan')
        self.ax_graph.set_xlabel('Time (samples)', fontsize=12, fontweight='bold')
        self.ax_graph.set_ylabel('Velocity (km/h)', fontsize=12, fontweight='bold')
        self.ax_graph.set_ylim(-5, 65)
        self.ax_graph.grid(True, alpha=0.2, linestyle='--')
        self.ax_graph.legend(loc='upper left', fontsize=11)
        self.ax_graph.set_title('📈 Real-time Velocity Graph', fontsize=14, fontweight='bold', pad=15)
        
        # 2. 속도계 바 (좌하단)
        self.ax_bar = self.fig.add_subplot(gs[2, 0])
        self.ax_bar.set_xlim(0, 60)
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_xlabel('Velocity (km/h)', fontsize=11, fontweight='bold')
        self.ax_bar.set_title('🏎️ Speed Meter', fontsize=12, fontweight='bold')
        self.ax_bar.set_yticks([])
        
        # 속도계 바 초기화
        self.speed_bar = self.ax_bar.barh(0.5, 0, height=0.6, color='lime', alpha=0.8)
        
        # 속도 텍스트
        self.speed_text = self.ax_bar.text(
            30, 0.5, '0',
            ha='center', va='center',
            fontsize=24, fontweight='bold',
            color='white'
        )
        
        # 3. 통계 정보 (우하단)
        self.ax_stats = self.fig.add_subplot(gs[2, 1])
        self.ax_stats.axis('off')
        self.ax_stats.set_title('📊 Statistics', fontsize=12, fontweight='bold', loc='left')
        
        # 통계 텍스트 초기화
        self.stats_text = self.ax_stats.text(
            0.1, 0.5,
            'Waiting for data...',
            fontsize=11,
            verticalalignment='center',
            family='monospace',
            color='lightgray'
        )
        
        # Figure 제목
        self.fig.suptitle(
            '🚗 VELOCITY MONITOR 🚗',
            fontsize=18,
            fontweight='bold',
            color='cyan'
        )
        
        # 애니메이션 시작
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=50,  # 50ms 업데이트
            blit=False,
            cache_frame_data=False
        )
        
        # 창 닫기 이벤트 핸들러
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        plt.show()
    
    def update_plot(self, frame):
        """그래프 업데이트 (애니메이션 콜백)"""
        if len(self.time_data) > 0:
            # 1. 실시간 그래프 업데이트
            time_list = list(self.time_data)
            velocity_list = list(self.velocity_data)
            
            self.line.set_data(time_list, velocity_list)
            self.ax_graph.set_xlim(
                max(0, self.data_counter - 100),
                max(100, self.data_counter + 10)
            )
            
            # 그래프 아래 영역 채우기
            self.ax_graph.collections.clear()
            self.ax_graph.fill_between(
                time_list,
                velocity_list,
                alpha=0.3,
                color='cyan'
            )
            
            # 2. 속도계 바 업데이트
            velocity = self.current_velocity
            self.speed_bar[0].set_width(velocity)
            
            # 속도에 따라 색상 변경
            if velocity < 20:
                color = 'lime'
                status = '🟢 Low'
            elif velocity < 40:
                color = 'yellow'
                status = '🟡 Medium'
            else:
                color = 'red'
                status = '🔴 High'
            
            self.speed_bar[0].set_color(color)
            self.speed_bar[0].set_alpha(0.8)
            
            # 속도 텍스트 업데이트
            self.speed_text.set_text(f'{velocity}')
            self.speed_text.set_position((max(5, velocity / 2), 0.5))
            
            # 3. 통계 정보 업데이트
            if len(velocity_list) > 0:
                avg_velocity = sum(velocity_list) / len(velocity_list)
                max_velocity = max(velocity_list)
                min_velocity = min(velocity_list)
                
                stats_str = (
                    f"Current:  {velocity:3d} km/h\n"
                    f"Average:  {avg_velocity:6.1f} km/h\n"
                    f"Maximum:  {max_velocity:3d} km/h\n"
                    f"Minimum:  {min_velocity:3d} km/h\n"
                    f"Samples:  {len(velocity_list):3d}\n"
                    f"Status:   {status}"
                )
                
                self.stats_text.set_text(stats_str)
        
        return self.line, self.speed_bar, self.speed_text, self.stats_text
    
    def on_close(self, event):
        """그래프 창 닫기 이벤트"""
        print("\n그래프 창이 닫혔습니다. 프로그램을 종료합니다...")
        self.running = False
        if rclpy.ok():
            self.node.destroy_node()
            rclpy.shutdown()
        plt.close('all')
        sys.exit(0)


def main():
    try:
        visualizer = VelocityVisualizer()
    except KeyboardInterrupt:
        print("\n\nKeyboard Interrupt. 프로그램을 종료합니다...")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        plt.close('all')


if __name__ == '__main__':
    main()

