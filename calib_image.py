#!/usr/bin/env python3
import cv2
import numpy as np
import os

def main():
    # 1. 이미 추출 완료된 캘리브레이션 행렬 로드 (.npz)
    npz_path = 'calibration_matrix.npz'
    if not os.path.exists(npz_path):
        print(f"❌ '{npz_path}' 파일이 현재 경로에 없습니다. 경로를 확인해 주세요.")
        return

    with np.load(npz_path) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
    print("✅ 왜곡 다림질 수식 장전 완료! (1280x720 기준)")

    # 2. RTSP 고정형 IP 카메라 연결
    RTSP_URL = "rtsp://admin:kue0504950!@192.168.0.30:554/onvif1"
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("❌ RTSP 스트림을 열 수 없습니다. 카메라 네트워크 상태를 확인하세요.")
        return

    # 3. 캘리브레이션을 진행했던 고정 해상도 
    # (인규님의 행렬은 반드시 이 해상도에서만 완벽하게 작동합니다)
    w, h = 1280, 720

    # 시야각 100% 보존 모드 (Alpha=1) 적용
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # 속도 극대화를 위한 리맵핑 연산 사전 준비
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1)

    print("\n=== 🛠️ 고정형 카메라 정방향 왜곡 보정 스트리밍 시작 ===")
    print("👉 1280x720 리사이즈 ➡️ 180도 회전 ➡️ Alpha=1 왜곡 보정")
    print("👉 [알파벳 q]: 스트리밍 종료\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("❌ 프레임을 유실했습니다.")
            break

        # [핵심 1] 왜곡 다림질을 하기 '전'에 반드시 1280x720으로 줄여야 합니다!
        resized_frame = cv2.resize(frame, (w, h))

        # [핵심 2] 천장 거치 상태 보정 (180도 회전)
        rotated_frame = cv2.rotate(resized_frame, cv2.ROTATE_180)

        # [핵심 3] 인규님의 매트릭스로 렌즈 다림질! (remap 사용으로 렉 제로)
        undistorted_frame = cv2.remap(rotated_frame, mapx, mapy, cv2.INTER_LINEAR)

        # 결과 출력
        cv2.imshow("Perfect Calibration View", undistorted_frame)

        # 키 입력 대기
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 사용자에 의해 스트리밍이 종료되었습니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()