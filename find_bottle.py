#!/usr/bin/env python3
import cv2
import numpy as np
import os

def main():
    # 1. 캘리브레이션 행렬 로드 (1280x720 기준)
    npz_path = 'calibration_matrix.npz'
    if not os.path.exists(npz_path):
        print(f"❌ '{npz_path}' 파일이 현재 경로에 없습니다.")
        return

    with np.load(npz_path) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
    print("✅ 왜곡 다림질 수식 장전 완료!")

    # 2. RTSP 고정형 IP 카메라 연결
    RTSP_URL = "rtsp://admin:kue0504950!@192.168.0.30:554/onvif1"
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("❌ RTSP 스트림을 열 수 없습니다.")
        return

    w, h = 1280, 720
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1)

    print("\n=== 🔵 실시간 파란색 물병 좌표 추적 (왜곡 완벽 보정) ===")
    print("👉 [알파벳 q]: 스트리밍 종료\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("❌ 프레임을 유실했습니다.")
            break

        # ==========================================
        # ⭐️ [핵심 디버깅 완료] 연산 순서 완벽 교정
        # ==========================================
        # 1단계: 스케일 매칭 (1280x720)
        resized_frame = cv2.resize(frame, (w, h))
        
        # 2단계: 다림질 먼저! (렌즈 원본 방향에서 왜곡을 100% 펴줍니다)
        undistorted_frame = cv2.remap(resized_frame, mapx, mapy, cv2.INTER_LINEAR)
        
        # 3단계: 평평해진 화면을 180도 뒤집어서 천장 뷰 정방향 세팅!
        process_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_180)
        # ==========================================

        # [단계 2] HSV 색상계 변환
        hsv_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2HSV)

        # [단계 3] 파란색 마스크 씌우기
        lower_blue = np.array([90, 80, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # 노이즈 다듬기 (작은 점을 살리기 위해 3x3 커널 사용)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # [단계 4] 윤곽선 및 좌표 추출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_u, target_v = 0, 0 

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # 면적 허들을 50으로 대폭 낮춰서 작은 물병 뚜껑(점)도 완벽 인식!
            if area > 50: 
                x, y, w_box, h_box = cv2.boundingRect(largest_contour)
                cv2.rectangle(process_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    target_u = int(M["m10"] / M["m00"])
                    target_v = int(M["m01"] / M["m00"])

                    cv2.circle(process_frame, (target_u, target_v), 5, (0, 0, 255), -1)
                    cv2.putText(process_frame, f"Target(u:{target_u}, v:{target_v})", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # [출력] 모니터링을 위해 절반 크기(640x360)로 나란히 이어 붙이기
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        view_main = cv2.resize(process_frame, (640, 360))
        view_mask = cv2.resize(mask_colored, (640, 360))
        
        cv2.putText(view_mask, "AI Mask (Blue Only)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        combined_view = cv2.hconcat([view_main, view_mask])
        cv2.imshow("High-Speed Bottle Tracking", combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 스트리밍 종료")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()