#!/usr/bin/env python3
import cv2
import numpy as np
import os

# 1. 책상 호모그래피 행렬 로드
def load_homography(path):
    if not os.path.exists(path):
        print(f"❌ '{path}' 파일이 없습니다. 먼저 추출을 진행해주세요.")
        exit(1)
    with np.load(path) as data:
        return data['H']

# 2. 마우스 클릭 이벤트 처리기
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        H_matrix = param['H']
        display_frame = param['frame']

        # 픽셀 좌표 (u, v)를 동차 좌표계로 변환 후 H 연산
        pixel_point = np.array([[[float(x), float(y)]]])
        physical_point = cv2.perspectiveTransform(pixel_point, H_matrix)
        
        real_X = physical_point[0][0][0]
        real_Y = physical_point[0][0][1]
        
        # 터미널에 출력
        print(f"🎯 픽셀 ({x:4d}, {y:4d}) ➡️ 바닥 원점 기준 물리 좌표: X = {real_X:+.3f}m, Y = {real_Y:+.3f}m")
        
        # 화면에 시각적으로 표시 (빨간색 동그라미와 텍스트)
        cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(display_frame, f"X:{real_X:.2f} Y:{real_Y:.2f}", 
                    (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Coordinate Verifier", display_frame)

def main():
    # 캘리브레이션 파일 로드
    npz_path = 'calibration_matrix_fin.npz'
    with np.load(npz_path) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]

    H_desk = load_homography('homography_desk_fixed.npz')

    RTSP_URL = "rtsp://admin:kue0504950!@192.168.0.30:554/onvif1"
    cap = cv2.VideoCapture(RTSP_URL)
    w, h = 1280, 720
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    window_name = "Coordinate Verifier"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1280, 720)

    # 마우스 콜백 파라미터용 딕셔너리
    callback_params = {'H': H_desk, 'frame': None}
    cv2.setMouseCallback(window_name, mouse_callback, callback_params)

    print("\n=== 🔍 [검증 모드] 책상 평면 실시간 좌표 스캐너 ===")
    print("👉 화면 속 '책상 위'의 아무 곳이나 마우스로 클릭해보세요!")
    print("👉 [q]를 누르면 종료됩니다.\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 왜곡 보정 및 리사이즈
        resized_frame = cv2.resize(frame, (w, h))
        rotated_frame = cv2.rotate(resized_frame, cv2.ROTATE_180)
        undistorted_frame = cv2.undistort(rotated_frame, mtx, dist, None, newcameramtx)
        
        # 콜백에서 그릴 수 있도록 현재 프레임 업데이트
        callback_params['frame'] = undistorted_frame

        cv2.imshow(window_name, undistorted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()