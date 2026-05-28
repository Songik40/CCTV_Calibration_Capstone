#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
import os

# ==========================================
# 🛠️ [현장 튜닝용 파라미터]
# ==========================================
# 1. 책상 영역 제한 (이 범위를 벗어난 파란색은 모두 무시!)
# multi-homography.py와 동일한 책상 물리 좌표 세팅
DESK_OFFSET_X = -1.02
DESK_OFFSET_Y = 0.284
DESK_WIDTH = 0.266
DESK_HEIGHT = 0.73

# 2. 물병 뚜껑 -> 몸통 보정 상수 (미터 단위)
# 로봇이 물병 중심보다 X로 2cm 덜 간다면 +0.02, Y로 1.5cm 더 간다면 -0.015
BOTTLE_OFFSET_X = 0.195
BOTTLE_OFFSET_Y = 0.214

class SmartBlueCapPublisher(Node):
    def __init__(self):
        super().__init__('smart_blue_cap_publisher')
        self.publisher_ = self.create_publisher(PointStamped, '/target_bottle_position', 10)
        
        self.lower_blue = np.array([90, 80, 50])  # 파란색 인식 범위를 더 넓고 유연하게 수정
        self.upper_blue = np.array([130, 255, 255])

        # 💡 [사용자 요청] 화면을 키우고 리사이즈 가능하게 설정
        self.window_name = "Smart Blue Vision Node"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, 1600, 450) # 800x450 뷰 2개를 합친 크기

        npz_path = 'calibration_matrix_fin.npz'
        homo_path = 'homography_desk_fixed.npz'
        
        with np.load(npz_path) as X:
            self.mtx, self.dist = [X[i] for i in ('mtx', 'dist')]
        with np.load(homo_path) as data:
            self.H_desk = data['H']

        RTSP_URL = "rtsp://admin:kue0504950!@192.168.0.30:554/onvif1"
        self.cap = cv2.VideoCapture(RTSP_URL)
        self.w, self.h = 1280, 720
        self.newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.w, self.h), 1, (self.w, self.h))

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("✅ [안전 모드] 책상 위 파란색 물병만 정밀 추적합니다.")

    def timer_callback(self):
        success, frame = self.cap.read()
        if not success:
            return

        resized_frame = cv2.resize(frame, (self.w, self.h))
        rotated_frame = cv2.rotate(resized_frame, cv2.ROTATE_180)
        undistorted_frame = cv2.undistort(rotated_frame, self.mtx, self.dist, None, self.newcameramtx)
        display_frame = undistorted_frame.copy()

        hsv_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_frame, self.lower_blue, self.upper_blue)
        
        # 🚨 형태학적 변환 최적화 (작은 뚜껑이 지워지지 않도록 보호!)
        kernel_small = np.ones((3,3), np.uint8) # 노이즈 제거용 작은 지우개
        kernel_large = np.ones((5,5), np.uint8) # 구멍 메우기용 큰 지우개
        
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_small) # 자잘한 노이즈만 살짝 제거
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_large) # 덩어리 안의 구멍 메우기
        blue_mask = cv2.dilate(blue_mask, kernel_small, iterations=1) # 뚜껑 크기를 살짝 팽창시켜 인식 안정화

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bottle_found = False
        full_edge_mask = np.zeros((self.h, self.w), dtype=np.uint8) # 🚨 이 줄이 누락되어 에러가 났습니다! (몸통 엣지 시각화용 빈 도화지)

        if contours:
            # 면적이 10 이상인 파란 덩어리 통과 (작은 뚜껑도 쉽게 통과하도록 허들 더 낮춤)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
            
            for contour in valid_contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_u = int(M["m10"] / M["m00"])
                    center_v = int(M["m01"] / M["m00"])

                    # 1. 물리 좌표로 우선 변환
                    pixel_point = np.array([[[float(center_u), float(center_v)]]])
                    physical_point = cv2.perspectiveTransform(pixel_point, self.H_desk)
                    
                    raw_X = physical_point[0][0][0]
                    raw_Y = physical_point[0][0][1]

                    # 🛡️ 2. 필터링 로직: 마커 오차를 고려해 5cm(0.05m) 여유(Margin) 공간 추가!
                    margin = 0.05
                    min_x = min(DESK_OFFSET_X, DESK_OFFSET_X + DESK_WIDTH) - margin
                    max_x = max(DESK_OFFSET_X, DESK_OFFSET_X + DESK_WIDTH) + margin
                    min_y = min(DESK_OFFSET_Y, DESK_OFFSET_Y + DESK_HEIGHT) - margin
                    max_y = max(DESK_OFFSET_Y, DESK_OFFSET_Y + DESK_HEIGHT) + margin
                    
                    if not (min_x <= raw_X <= max_x and min_y <= raw_Y <= max_y):
                        # 거절된 파란색 덩어리는 화면에 빨간색으로 사유(좌표) 표시 (디버깅용)
                        cv2.putText(display_frame, f"OUT({raw_X:.2f},{raw_Y:.2f})", (center_u + 10, center_v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        continue # 책상 밖으로 판정되어 무시됨

                    # 🕵️ 물병 상태 판별 (서있음 vs 누워있음) - 뚜껑 너머 '투명 몸통' 투시!
                    # 파란 뚜껑만 보면 무조건 원형(1.0)이므로, 뚜껑 주변(ROI)의 플라스틱 빛 반사 테두리를 찾습니다.
                    roi_size = 220 # 🚨 물병 하단이 잘리지 않도록 넉넉하게 확장 (가로세로 440)
                    u_min, u_max = max(0, center_u - roi_size), min(self.w, center_u + roi_size)
                    v_min, v_max = max(0, center_v - roi_size), min(self.h, center_v + roi_size)
                    
                    roi_gray = cv2.cvtColor(display_frame[v_min:v_max, u_min:u_max], cv2.COLOR_BGR2GRAY)
                    
                    # 🚨 가우시안 블러로 책상의 텍스처(노이즈)를 뭉개고, 뚜렷한 플라스틱 테두리만 캡처!
                    roi_blur = cv2.GaussianBlur(roi_gray, (7, 7), 0)
                    edges = cv2.Canny(roi_blur, 50, 150)
                    
                    # 💡 [핵심 해결책] 뚜껑 테두리가 몸통으로 오해받는 현상 완벽 차단!
                    # Canny로 찾은 엣지 중에서, 파란색 뚜껑 영역의 엣지는 지워버립니다.
                    roi_blue = blue_mask[v_min:v_max, u_min:u_max]
                    roi_blue_dilated = cv2.dilate(roi_blue, np.ones((15, 15), np.uint8)) # 뚜껑 경계선까지 넉넉히 덮기
                    edges[roi_blue_dilated > 0] = 0 # 뚜껑 엣지 삭제! 이제 오직 '투명 몸통'만 남습니다.

                    edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2) # 윤곽선 하나로 뭉치기
                    
                    full_edge_mask[v_min:v_max, u_min:u_max] = edges # 오른쪽 마스크 화면에 보여주기 위해 저장
                    
                    body_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    aspect_ratio = 1.0
                    bottle_state = "UNKNOWN"
                    body_box = None
                    
                    if body_contours:
                        largest_body = max(body_contours, key=cv2.contourArea)
                        if cv2.contourArea(largest_body) > 300: # 의미 있는 크기의 몸통일 때만
                            largest_body[:, 0, 0] += u_min # ROI 좌표를 전체 화면 좌표로 복원
                            largest_body[:, 0, 1] += v_min
                            
                            rect = cv2.minAreaRect(largest_body)
                            (rect_c, (rect_w, rect_h), rect_angle) = rect
                            if rect_w > 0 and rect_h > 0:
                                aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
                                # 투명 몸통의 비율이 1.3 이상 길쭉하면 누워있다고 완벽 판정!
                                bottle_state = "LYING" if aspect_ratio > 1.3 else "STANDING"
                                body_box = np.int32(cv2.boxPoints(rect))

                    # 🎯 3. 상태에 따라 최종 타겟 좌표 결정
                    if bottle_state == "LYING" and body_box is not None:
                        # [누워있을 때] 몸통의 중심점을 타겟으로 설정 (오프셋 불필요)
                        target_u, target_v = int(rect_c[0]), int(rect_c[1])
                        
                        # 몸통 중심의 물리 좌표를 새로 계산
                        pixel_point = np.array([[[float(target_u), float(target_v)]]])
                        physical_point = cv2.perspectiveTransform(pixel_point, self.H_desk)
                        final_X = physical_point[0][0][0]
                        final_Y = physical_point[0][0][1]
                    else:
                        # [서있거나, 상태를 모를 때] 뚜껑 좌표에 오프셋을 더해 몸통 중심을 타겟팅
                        target_u, target_v = center_u, center_v # 시각화용
                        final_X = raw_X + BOTTLE_OFFSET_X
                        final_Y = raw_Y + BOTTLE_OFFSET_Y

                    bottle_found = True

                    # 📡 ROS 2 토픽 발행
                    msg = PointStamped()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "camera_floor_link" 
                    msg.point.x = final_X
                    msg.point.y = final_Y
                    msg.point.z = 0.0 
                    
                    self.publisher_.publish(msg)
                    
                    # 터미널에 상태와 좌표, 비율까지 함께 출력!
                    self.get_logger().info(f"🚀 [{bottle_state}] 타겟 좌표: X={final_X:.3f}m, Y={final_Y:.3f}m (비율: {aspect_ratio:.2f})")

                    # 시각화
                    color = (0, 255, 0) if bottle_state == "STANDING" else (0, 165, 255) # 누워있으면 주황색 상자
                    
                    # 최종 타겟 지점에 보라색 십자선 표시
                    cv2.drawMarker(display_frame, (target_u, target_v), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    
                    # 파란 뚜껑 대신, 찾아낸 물병 '전체 몸통'을 기준으로 박스를 그려줍니다!
                    if body_box is not None:
                        cv2.drawContours(display_frame, [body_box], 0, color, 2)
                        x_body, y_body, _, _ = cv2.boundingRect(largest_body)
                        cv2.putText(display_frame, f"{bottle_state} (Ratio:{aspect_ratio:.2f})", 
                                    (x_body, y_body - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        x, y, w_box, h_box = cv2.boundingRect(contour)
                        cv2.rectangle(display_frame, (x, y), (x+w_box, y+h_box), color, 2)
                        cv2.putText(display_frame, f"{bottle_state} (X:{final_X:.2f}, Y:{final_Y:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    break # 가장 먼저 조건에 맞는(책상 위) 파란색 1개만 처리

        if not bottle_found:
            cv2.putText(display_frame, "Searching Bottle on DESK...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # [출력] 파란 뚜껑 마스크 + 엣지 투시(Canny) 마스크를 합쳐서 보여줍니다!
        combined_mask = cv2.bitwise_or(blue_mask, full_edge_mask)
        mask_colored = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        view_main = cv2.resize(display_frame, (800, 450))
        view_mask = cv2.resize(mask_colored, (800, 450))
        
        cv2.putText(view_mask, "AI Mask (Cap + Body Edge)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        combined_view = cv2.hconcat([view_main, view_mask])
        cv2.imshow(self.window_name, combined_view)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SmartBlueCapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 노드 종료")
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()