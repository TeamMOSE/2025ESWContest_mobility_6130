import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32, Float32, Int32MultiArray, String
from cv_bridge import CvBridge

def ema(prev, new, alpha):
    return new if prev is None else (alpha * new + (1.0 - alpha) * prev)

class LaneSlopeDetector(Node):
    def __init__(self):
        super().__init__('lane_slope_detector')

                # Topic Names
        self.declare_parameter('input_topic', '/front_camera')
        self.declare_parameter('overlay_topic', '/lane/overlay')
        self.declare_parameter('lane_info_topic', '/lane/info/cv')

        self.declare_parameter('show_debug', False)

        # ROI 파라미터
        self.declare_parameter('roi_top_y', 0.40)   
        self.declare_parameter('roi_bot_y', 0.98)
        self.declare_parameter('roi_left_x', 0.0)
        self.declare_parameter('roi_right_x', 1.0)

        # 흰색, 노란색만 검출 + 엣지
        self.declare_parameter('white_lower', [0, 0, 200])
        self.declare_parameter('white_upper', [180, 50, 255])
        self.declare_parameter('yellow_lower', [15, 60, 120])
        self.declare_parameter('yellow_upper', [35, 255, 255])
        self.declare_parameter('canny_low', 80)
        self.declare_parameter('canny_high', 160)

        # hough 파라미터
        self.declare_parameter('hough_rho', 1)
        self.declare_parameter('hough_theta_deg', 1.0)
        self.declare_parameter('hough_thresh', 30)
        self.declare_parameter('hough_min_line_len', 40)
        self.declare_parameter('hough_max_line_gap', 20)


        
        # 수평에 가까운 선 제거
        self.declare_parameter('min_angle_deg', 20.0)

        self.declare_parameter('total_lanes', 2)
        
        # 반대 차선 제거
        self.declare_parameter('block_left_of_yellow', True)
        self.declare_parameter('side_margin_px', 16)
        
        # 노란선 검출 파라미터
        self.declare_parameter('yellow_angle_min_deg', 5.0)
        self.declare_parameter('yellow_angle_max_deg', 60.0)
        self.declare_parameter('yellow_xmax_ratio', 0.70)
        self.declare_parameter('yellow_min_len_px', 80)
        self.declare_parameter('yellow_ema_alpha', 0.3)
        self.declare_parameter('keep_yellow_in_centerlines', True)

        # get_parameter
        self.input_topic = self.get_parameter('input_topic').value
        self.overlay_topic = self.get_parameter('overlay_topic').value
        self.lane_info_topic = self.get_parameter('lane_info_topic').value

        self.show_debug = bool(self.get_parameter('show_debug').value)

        self.roi_top_y = float(self.get_parameter('roi_top_y').value)
        self.roi_bot_y = float(self.get_parameter('roi_bot_y').value)
        self.roi_left_x = float(self.get_parameter('roi_left_x').value)
        self.roi_right_x = float(self.get_parameter('roi_right_x').value)

        self.white_lower = np.array(self.get_parameter('white_lower').value, dtype=np.uint8)
        self.white_upper = np.array(self.get_parameter('white_upper').value, dtype=np.uint8)
        self.yellow_lower = np.array(self.get_parameter('yellow_lower').value, dtype=np.uint8)
        self.yellow_upper = np.array(self.get_parameter('yellow_upper').value, dtype=np.uint8)
        self.canny_low = int(self.get_parameter('canny_low').value)
        self.canny_high = int(self.get_parameter('canny_high').value)

        self.hough_rho = int(self.get_parameter('hough_rho').value)
        self.hough_theta = math.radians(float(self.get_parameter('hough_theta_deg').value))
        self.hough_thresh = 40

        
        self.min_angle = float(self.get_parameter('min_angle_deg').value)
        self.total_lanes = int(self.get_parameter('total_lanes').value)
        
        self.block_left_of_yellow = bool(self.get_parameter('block_left_of_yellow').value)
        self.side_margin = int(self.get_parameter('side_margin_px').value)
        
        self.yellow_ang_min = float(self.get_parameter('yellow_angle_min_deg').value)
        self.yellow_ang_max = float(self.get_parameter('yellow_angle_max_deg').value)
        self.yellow_xmax_ratio = float(self.get_parameter('yellow_xmax_ratio').value)
        self.yellow_min_len = int(self.get_parameter('yellow_min_len_px').value)
        self.yellow_ema = float(self.get_parameter('yellow_ema_alpha').value)
        self.keep_yellow = bool(self.get_parameter('keep_yellow_in_centerlines').value)
        
        # 중앙선 EMA
        self.y_m = None
        self.y_c = None

        # 입력용 QoS (카메라 토픽과 매칭 - BEST_EFFORT + 버퍼링)
        qos_input = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=3,  # 1 → 3 (네트워크 지연 시 버퍼링)
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # 출력용 QoS (RViz와 호환 - RELIABLE)
        qos_output = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        self.bridge = CvBridge()
        # CompressedImage 구독 (무선 통신 최적화)
        self.sub = self.create_subscription(CompressedImage, self.input_topic + '/compressed', self.on_image, qos_input)
        
        # /lane/source 구독 (현재 사용 중인 소스 정보)
        self.source_sub = self.create_subscription(String, '/lane/source', self.on_source, 10)
        
        self.pub_overlay = self.create_publisher(Image, self.overlay_topic, qos_output)
        self.pub_lane_info = self.create_publisher(Int32MultiArray, self.lane_info_topic, qos_output)

        self.frame_count = 0  
        
        self.last_centerlines = []
        self.miss_streak = 0
        
        self.last_lane = 1
        
        self.current_source = "CV"  # 기본값

        self.get_logger().info(f'LaneSlopeDetector subscribed: {self.input_topic}/compressed')
        self.get_logger().info(f'Publishing overlay to: {self.overlay_topic}')
        self.get_logger().info(f'QoS: Input=BEST_EFFORT (CompressedImage), Output=RELIABLE')

    def on_source(self, msg: String):
        """현재 사용 중인 소스 정보 업데이트"""
        self.current_source = msg.data

    def find_yellow_line(self, mask_y, h, w):
        """노란 마스크에서 허프로 여러 조각을 뽑고 조건 필터 후 x=m*y+c로 피팅.
           반환: dict(m,c,y_min,y_max,angle) 또는 None
        """
        # 얇은 단선 보강
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,9))
        my = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, k, iterations=1)
        L = cv2.HoughLinesP(my, rho=1, theta=np.deg2rad(1), threshold=30,
                            minLineLength=max(40, int(0.10*h)), maxLineGap=int(0.03*h))
        if L is None: 
            return None

        # 조건: (1) 화면 왼쪽~중앙(노간선 위치 가이드), (2) 최소 길이
        pieces = []
        xmax = int(self.yellow_xmax_ratio * w)
        for l in L[:,0,:]:
            x1,y1,x2,y2 = map(int,l)
            length = math.hypot(x2-x1, y2-y1)
            if length < self.yellow_min_len: 
                continue
            # 두 끝이 너무 오른쪽이면 중앙선 아닐 확률이 높음
            if max(x1,x2) > xmax:
                continue
            pieces.append((x1,y1,x2,y2))

        if len(pieces) < 2:
            return None

        # x = m*y + c
        xs = []
        ys = []
        y_min, y_max = h, 0
        for x1,y1,x2,y2 in pieces:
            xs += [x1,x2]
            ys += [y1,y2]
            y_min = min(y_min, y1, y2)
            y_max = max(y_max, y1, y2)

        ys = np.asarray(ys, dtype=float)
        xs = np.asarray(xs, dtype=float)
        y_mean = ys.mean()
        denom = np.sum((ys - y_mean)**2)
        if denom < 1e-6:
            return None
        m = np.sum((ys - y_mean)*xs) / denom
        c = xs.mean() - m*y_mean

        self.y_m = m if self.y_m is None else (self.yellow_ema*m + (1-self.yellow_ema)*self.y_m)
        self.y_c = c if self.y_c is None else (self.yellow_ema*c + (1-self.yellow_ema)*self.y_c)
        m, c = self.y_m, self.y_c

        return {'m': m, 'c': c, 'y_min': int(y_min), 'y_max': int(y_max)}

    def merge_parallel_lines(self, lines, h):
        """평행·근접 라인을 그룹화하여 중앙선 모델(x = m*y + c)로 변환"""
        if len(lines) == 0:
            return []
        
        line_models = []
        for x1, y1, x2, y2 in lines:
            dy = y2 - y1
            if abs(dy) < 1:
                continue
            m = (x2 - x1) / dy
            c = x1 - m * y1
            line_models.append({
                'm': m,
                'c': c,
                'points': [(x1, y1), (x2, y2)],
                'y_min': min(y1, y2),
                'y_max': max(y1, y2)
            })
        
        if len(line_models) == 0:
            return []
        
        # 거리기반 클러스터링
        clusters = []
        merge_distance_threshold = 30.0
        
        for model in line_models:
            merged = False
            for cluster in clusters:
                repr_model = cluster[0]
                y_test = h * 0.8
                x1 = model['m'] * y_test + model['c']
                x2 = repr_model['m'] * y_test + repr_model['c']
                distance = abs(x1 - x2)
                if distance <= merge_distance_threshold:
                    cluster.append(model)
                    merged = True
                    break
            if not merged:
                clusters.append([model])
        
        # 각 클러스터의 중앙선 계산
        centerlines = []
        for cluster in clusters:
            all_points = []
            y_min_global = float('inf')
            y_max_global = 0
            for model in cluster:
                all_points.extend(model['points'])
                y_min_global = min(y_min_global, model['y_min'])
                y_max_global = max(y_max_global, model['y_max'])
            if len(all_points) < 2:
                continue
            ys = np.array([p[1] for p in all_points], dtype=float)
            xs = np.array([p[0] for p in all_points], dtype=float)
            y_mean = np.mean(ys)
            ys_centered = ys - y_mean
            if np.sum(ys_centered ** 2) < 1e-6:
                continue
            m = np.sum(ys_centered * xs) / np.sum(ys_centered ** 2)
            c = np.mean(xs) - m * y_mean
            centerlines.append({
                'm': m,
                'c': c,
                'y_min': int(y_min_global),
                'y_max': int(y_max_global)
            })
        return centerlines

    def dedupe_centerlines(self, cls, h, w, join_px=25, min_sep_px=60):
        """중복 중앙선 제거/재병합"""
        if not cls:
            return []
        probe_ys = [int(0.85*h), int(0.70*h)]
        clusters = []
        for cl in cls:
            merged = False
            for C in clusters:
                ref = C[0]
                ok = True
                for yy in probe_ys:
                    if abs((cl['m']*yy + cl['c']) - (ref['m']*yy + ref['c'])) > join_px:
                        ok = False
                        break
                if ok:
                    C.append(cl)
                    merged = True
                    break
            if not merged:
                clusters.append([cl])
        
        outs = []
        for C in clusters:
            m = np.mean([c['m'] for c in C])
            c_val = np.mean([c['c'] for c in C])
            y0 = int(min(c_['y_min'] for c_ in C))
            y1 = int(max(c_['y_max'] for c_ in C))
            outs.append({'m': m, 'c': c_val, 'y_min': y0, 'y_max': y1})
        
        # 가까운 건 한 차선임
        outs.sort(key=lambda cl: cl['m']*probe_ys[0] + cl['c'])
        dedup = []
        for cl in outs:
            if not dedup:
                dedup.append(cl)
                continue
            yy = probe_ys[0]
            if abs((cl['m']*yy + cl['c']) - (dedup[-1]['m']*yy + dedup[-1]['c'])) < min_sep_px:
                span = lambda o: o['y_max'] - o['y_min']
                if span(cl) > span(dedup[-1]):
                    dedup[-1] = cl
            else:
                dedup.append(cl)
        return dedup


    def on_image(self, msg: CompressedImage):
        self.frame_count += 1
        
        if self.frame_count % 10 == 0:
            self.get_logger().info(f'Processing frame #{self.frame_count}')
        
        # CompressedImage 압축 해제 (무선 통신 최적화)
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]

        # ROI 
        roi_poly = np.array([[
            (int(self.roi_left_x*w), int(self.roi_top_y*h)),
            (int(self.roi_right_x*w), int(self.roi_top_y*h)),
            (int(self.roi_right_x*w), int(self.roi_bot_y*h)),
            (int(self.roi_left_x*w), int(self.roi_bot_y*h))
        ]], dtype=np.int32)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, roi_poly, 255)
        roi_img = cv2.bitwise_and(bgr, bgr, mask=roi_mask)

        # hsv 변환
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        v_ch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(v_ch)
        hsv = cv2.merge([h_ch, s_ch, v_ch])
        
        # color mask
        mask_w = cv2.inRange(hsv, self.white_lower, self.white_upper)
        mask_y = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # 얇은 선 잘 안 잡히는 거 보강
        kern_w_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_CLOSE, kern_w_v, iterations=1)
        mask_w = cv2.dilate(mask_w, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        
        # 중앙성 끊김 보강
        mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kern_w_v, iterations=1)
        
        mask = cv2.bitwise_or(mask_w, mask_y)
        
        # 점선을 하나의 선으로
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,11)), iterations=1)
        
        edges = cv2.Canny(mask, max(50, self.canny_low-20), max(100, self.canny_high-20))

        # 허프 검출
        hough_min_len = int(0.12 * h)
        hough_max_gap = int(0.03 * h)
        
        lines = cv2.HoughLinesP(edges,
                                rho=self.hough_rho,
                                theta=self.hough_theta,
                                threshold=self.hough_thresh,
                                minLineLength=hough_min_len,
                                maxLineGap=hough_max_gap)

        overlay = bgr.copy()
        filtered_lines = []

        if lines is not None:
            for l in lines[:, 0, :]:
                x1, y1, x2, y2 = map(int, l)
                # 아주 짧은 선은 제외
                length = math.hypot(x2-x1, y2-y1)
                if length < 30:
                    continue
                    
                dx = (x2 - x1)
                dy = (y2 - y1)
                
                # 수평선은 제외
                if abs(dx) > 0:
                    angle_deg = abs(math.degrees(math.atan2(dy, dx)))
                    if angle_deg < self.min_angle:
                        continue
                else:
                    pass

                # ROI 내부만 수집
                if roi_mask[y1, x1] > 0 and roi_mask[y2, x2] > 0:
                    filtered_lines.append((x1, y1, x2, y2))
        
        # 같은 차선 병합
        centerlines = self.merge_parallel_lines(filtered_lines, h)
        centerlines = self.dedupe_centerlines(centerlines, h, w, join_px=25, min_sep_px=60)
        
        # 노란선(중앙선) 검출
        yellow_cl = self.find_yellow_line(mask_y, h, w)
        
        # 노란선도 차선에 포함
        if self.keep_yellow and yellow_cl is not None:
            centerlines.append(yellow_cl)
            centerlines = self.dedupe_centerlines(centerlines, h, w, join_px=20, min_sep_px=50)
        
        # 중앙선 기준 왼쪽 제거
        if yellow_cl is not None and self.block_left_of_yellow:
            yb = yellow_cl['y_max']
            x_yellow_bot = yellow_cl['m']*yb + yellow_cl['c']
            kept = []
            for cl in centerlines:
                xb = cl['m']*yb + cl['c']
                if xb < (x_yellow_bot - self.side_margin):
                    continue
                kept.append(cl)
            centerlines = kept
        
        # 프레임 유지
        if len(centerlines) == 0:
            self.miss_streak += 1
            if self.miss_streak <= 3:
                centerlines = self.last_centerlines
        else:
            self.last_centerlines = centerlines
            self.miss_streak = 0
        
        # 라인 그리기
        for cl in centerlines:
            m, c = cl['m'], cl['c']
            y_min, y_max = cl['y_min'], cl['y_max']
            x_top = int(m * y_min + c)
            x_bottom = int(m * y_max + c)
            x_top = max(0, min(w-1, x_top))
            x_bottom = max(0, min(w-1, x_bottom))
            
            cv2.line(overlay, (x_top, y_min), (x_bottom, y_max), (0, 255, 0), 3)
        
        # 디버그 시에만 노란 경계선 표시
        if self.show_debug and yellow_cl is not None:
            y0, y1 = int(self.roi_top_y*h), int(self.roi_bot_y*h)
            x0 = int(np.clip(yellow_cl['m']*y0 + yellow_cl['c'], 0, w-1))
            x1 = int(np.clip(yellow_cl['m']*y1 + yellow_cl['c'], 0, w-1))
            cv2.line(overlay, (x0,y0), (x1,y1), (0,255,255), 2)

        # 차선 개수/현재 차로 계산 
        y_ref = int(0.90*h) 
        x_center = w * 0.5 
        yellow_match_px = max(12, self.side_margin)  

        line_positions = []
        x_yellow_ref = None
        if yellow_cl is not None:
            x_yellow_ref = yellow_cl['m']*y_ref + yellow_cl['c']

        for cl in centerlines:
            x_ref = cl['m']*y_ref + cl['c']
            is_yellow_line = False
            if x_yellow_ref is not None and abs(x_ref - x_yellow_ref) <= yellow_match_px:
                is_yellow_line = True
            line_positions.append((x_ref, is_yellow_line))

        # 총 차로수: 차선의 총 개수 - 중앙선
        non_yellow_cnt = sum(1 for _, isY in line_positions if not isY)
        total_lanes_dyn = max(1, non_yellow_cnt)

        # 현재 차로: 카메라의 중앙 기준으로 왼쪽 차선 개수
        left_cnt_including_yellow = sum(1 for x_ref, _ in line_positions if x_ref < x_center)
        current_lane = left_cnt_including_yellow
        current_lane = max(1, min(current_lane, total_lanes_dyn))

        # 디버깅 overlay에 텍스트
        cv2.putText(overlay, f"Lane {current_lane}/{total_lanes_dyn}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # 소스 정보 표시
        if self.current_source == "YOLO":
            source_color = (0, 255, 0)  # 초록색
        else:
            source_color = (255, 128, 0)  # 주황색
        
        cv2.putText(overlay, f"Source: {self.current_source}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, source_color, 4)

        # ROI 시각화
        cv2.polylines(overlay, roi_poly, True, (255, 0, 0), 2)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        self.pub_overlay.publish(overlay_msg)

        if self.frame_count <= 10:
            self.get_logger().info(f'Published overlay #{self.frame_count} to {self.overlay_topic}')

        lane_info_msg = Int32MultiArray()
        lane_info_msg.data = [int(total_lanes_dyn), int(current_lane)]
        self.pub_lane_info.publish(lane_info_msg)
        
        if self.frame_count <= 10:
            self.get_logger().info(f'Lane info: [{total_lanes_dyn}, {current_lane}] (Total lanes: {total_lanes_dyn}, Current lane: {current_lane})')

        if self.show_debug:
            cv2.imshow('mask', mask)
            cv2.imshow('edges', edges)
            cv2.imshow('overlay', overlay)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneSlopeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()