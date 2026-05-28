#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import os

# ==========================================
# 1. 캡스톤 맞춤형 물리 환경 세팅 (미터 단위)
# ==========================================
# [바닥 세팅] - 0번 마커를 원점(0,0)으로 사용
FLOOR_WIDTH = 0.406   # 0번 <-> 1번 가로 거리 (예: 80cm)
FLOOR_HEIGHT = 0.302  # 0번 <-> 3번 세로 거리 (예: 60cm)

# [책상 세팅] - 4번 마커 기준
DESK_OFFSET_X = -1.02 # 바닥 0번 마커(원점) 기준 4번 마커의 X 이동 거리
DESK_OFFSET_Y = 0.284 # 바닥 0번 마커(원점) 기준 4번 마커의 Y 이동 거리
DESK_WIDTH = 0.266    # 4번 <-> 5번 가로 거리
DESK_HEIGHT = 0.73   # 4번 <-> 7번 세로 거리

def get_homography(cap, mtx, dist, newcameramtx, w, h, mode):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    
    # 모드에 따른 타겟 ID 및 물리 좌표 설정
    if mode == 'FLOOR':
        target_ids = [6, 5, 0, 7]  # 좌상(TL), 우상(TR), 우하(BR), 좌하(BL)
        obj_points = np.array([
            [0.0,          0.0],
            [FLOOR_WIDTH,  0.0],
            [FLOOR_WIDTH,  FLOOR_HEIGHT],
            [0.0,          FLOOR_HEIGHT]
        ], dtype=np.float32)
        save_name = "homography_floor_fixed.npz"
        color = (0, 255, 255) # 노란색 텍스트
    else:
        target_ids = [4, 3, 2, 10] # 좌상(TL), 우상(TR), 우하(BR), 좌하(BL)
        obj_points = np.array([
            [DESK_OFFSET_X,               DESK_OFFSET_Y],
            [DESK_OFFSET_X + DESK_WIDTH,  DESK_OFFSET_Y],
            [DESK_OFFSET_X + DESK_WIDTH,  DESK_OFFSET_Y + DESK_HEIGHT],
            [DESK_OFFSET_X,               DESK_OFFSET_Y + DESK_HEIGHT]
        ], dtype=np.float32)
        save_name = "homography_desk_fixed.npz"
        color = (255, 100, 255) # 보라색 텍스트

    window_name = f"Homography Extractor - {mode}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1280, 720)

    print(f"\n=== 🎯 [{mode}] 4점 자동 호모그래피 추출 ===")
    print(f"👉 지정된 ID {target_ids} 마커가 모두 인식되면 [스페이스바]를 눌러 저장하세요.")
    print("👉 다른 모드로 넘어가려면 [q]를 누르세요.\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (w, h))
        rotated_frame = cv2.rotate(resized_frame, cv2.ROTATE_180)
        undistorted_frame = cv2.undistort(rotated_frame, mtx, dist, None, newcameramtx)
        display_frame = undistorted_frame.copy()
        
        gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        
        # 🚨 버전에 따른 ArUco 탐지기 호환성 처리
        try:
            # OpenCV 4.7 이상 최신 버전
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            # OpenCV 4.6 이하 구버전
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        detected_centers = {}

        if ids is not None:
            aruco.drawDetectedMarkers(display_frame, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                # 타겟 ID인 경우에만 화면에 표시 및 저장
                if marker_id in target_ids:
                    c = corners[i][0]
                    # 정중앙 픽셀 계산
                    center_u = int(np.mean(c[:, 0]))
                    center_v = int(np.mean(c[:, 1]))
                    detected_centers[marker_id] = (center_u, center_v)
                    cv2.circle(display_frame, (center_u, center_v), 5, (0, 0, 255), -1)
                    # 마커 옆에 현재 인식된 랜덤 ID 띄우기
                    cv2.putText(display_frame, f"ID: {marker_id}", (center_u + 10, center_v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 🚨 [핵심] 지정된 타겟 마커 4개가 모두 인식되었는지 확인
        all_found = all(m_id in detected_centers for m_id in target_ids)

        cv2.putText(display_frame, f"MODE: {mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if all_found:
            cv2.putText(display_frame, "ALL TARGET MARKERS FOUND! Press [SPACE]", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 인규님이 알려주신 좌상, 우상, 우하, 좌하 순서(target_ids) 그대로 배열 구성
            ordered_pts = [
                detected_centers[target_ids[0]], # TL
                detected_centers[target_ids[1]], # TR
                detected_centers[target_ids[2]], # BR
                detected_centers[target_ids[3]]  # BL
            ]
            pts = np.array(ordered_pts, np.int32)
            cv2.polylines(display_frame, [pts], True, color, 2)
        else:
            cv2.putText(display_frame, f"Found {len(detected_centers)}/4 target markers", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and all_found:
            img_points = np.array(ordered_pts, dtype=np.float32)

            H, _ = cv2.findHomography(img_points, obj_points)
            print(f"\n✅ [대성공] {save_name} 추출 완료!")
            np.savez(save_name, H=H)
            print("💾 파일 저장 완료! (창이 3초 뒤 닫힙니다)")
            cv2.waitKey(3000)
            break
        elif key == ord('q'):
            print(f"⚠️ {mode} 추출을 취소했습니다.")
            break

    cv2.destroyWindow(window_name)

def main():
    npz_path = 'calibration_matrix_fin.npz'
    if not os.path.exists(npz_path):
        print(f"❌ '{npz_path}' 파일이 없습니다.")
        return

    with np.load(npz_path) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]

    RTSP_URL = "rtsp://admin:kue0504950!@192.168.0.30:554/onvif1"
    cap = cv2.VideoCapture(RTSP_URL)
    w, h = 1280, 720
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    window_name = "Live Preview - Ready"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n=== 🚀 통합 호모그래피 컨트롤 센터 ===")
    print("카메라 실시간 화면(Live Preview)을 보면서 마커가 잘 나오는지 확인하세요!")
    print("👉 화면이 켜진 상태에서 키보드를 누르세요:")
    print("👉 [f]: 바닥 (Floor) 추출 모드 진입")
    print("👉 [d]: 책상 (Desk) 추출 모드 진입")
    print("👉 [q]: 프로그램 완전 종료\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        resized_frame = cv2.resize(frame, (w, h))
        rotated_frame = cv2.rotate(resized_frame, cv2.ROTATE_180)
        undistorted_frame = cv2.undistort(rotated_frame, mtx, dist, None, newcameramtx)
        
        # 화면 좌측 상단에 안내 문구 표시
        cv2.putText(undistorted_frame, "Ready! Press 'f' (Floor), 'd' (Desk) or 'q' (Quit)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, undistorted_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            get_homography(cap, mtx, dist, newcameramtx, w, h, 'FLOOR')
        elif key == ord('d'):
            get_homography(cap, mtx, dist, newcameramtx, w, h, 'DESK')
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
