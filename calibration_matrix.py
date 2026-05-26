'''

import cv2
import numpy as np
import os
import glob

# ==========================================
# 1. 캡스톤 맞춤형 환경 설정
# ==========================================
CHECKERBOARD = (10, 7)  # 인규님이 뽑은 완벽한 비대칭 10x7 구조
SQUARE_SIZE = 0.025     # 한 칸의 실제 길이 25mm (미터 단위: 0.025)

IMG_DIR = "calib_images"
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# ==========================================
# 2. RTSP IP 카메라 촬영 모드
# ==========================================
# 🎯 인규님의 실제 RTSP 주소 적용 완료!
RTSP_URL = "rtsp://admin:capstone1234!@192.168.0.15:554/onvif1"
cap = cv2.VideoCapture(RTSP_URL)

print("\n=== 📸 RTSP 카메라 캘리브레이션 스튜디오 ===")
print("👉 [스페이스바]: 사진 찰칵! (20~30장 권장)")
print("👉 [알파벳 q]: 촬영 종료 및 왜곡 다림질 연산 시작\n")

count = 0
while True:
    success, frame = cap.read()
    if not success:
        print("❌ 카메라 영상을 불러올 수 없습니다. IP 주소나 네트워크를 확인해 주세요!")
        break

    # 🎯 핵심: YOLO 돌릴 때랑 똑같이 1280x720으로 축소!
    # 캘리브레이션 사진 해상도와 욜로 해상도가 다르면 나중에 계산이 꼬입니다.
    resized_frame = cv2.resize(frame, (1280, 720))

    cv2.imshow('Calibration Camera', resized_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # 스페이스바 누르면 사진 저장
        img_name = os.path.join(IMG_DIR, f"calib_{count:02d}.jpg")
        cv2.imwrite(img_name, resized_frame) # 줄어든 프레임으로 저장
        print(f"[찰칵] {img_name} 저장 완료! (현재 {count + 1}장)")
        count += 1
    elif key == ord('q'):  # 'q' 누르면 반복문 탈출
        break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 3. OpenCV 왜곡 다림질 연산 모드
# ==========================================
if count < 10:
    print("\n🚨 사진을 너무 적게 찍으셨습니다! 최소 10장 이상 찍고 'q'를 눌러주세요.")
else:
    print("\n⚙️ 촬영 끝! 이제 사진들을 분석해서 왜곡 계수를 계산합니다. (몇 초 정도 걸립니다...)")

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = [] 
    imgpoints = [] 

    images = glob.glob(os.path.join(IMG_DIR, '*.jpg'))
    success_count = 0

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret == True:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            success_count += 1

    print(f"👉 총 {count}장 중 {success_count}장의 패턴을 성공적으로 인식했습니다!")

    if success_count >= 10:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        np.savez("calibration_matrix.npz", mtx=mtx, dist=dist)
        print("\n🎉 [대성공!] calibration_matrix.npz 파일이 완벽하게 생성되었습니다! 🎉")
    else:
        print("\n🚨 [실패] 패턴 인식 성공 사진이 너무 적습니다. 조명이나 각도를 바꿔서 다시 찍어주세요!")



import cv2
import numpy as np

# 1. 왜곡 계수 불러오기
try:
    with np.load('calibration_matrix.npz') as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
    print("✅ 왜곡 다림질 수식 장전 완료!")
except:
    print("🚨 오류: calibration_matrix.npz 파일이 없습니다!")
    exit()

# 2. RTSP IP 카메라 연결
RTSP_URL = "rtsp://admin:capstone1234!@192.168.0.15:554/onvif1"
cap = cv2.VideoCapture(RTSP_URL)

print("\n=== 🔍 비포 & 애프터 검증 스튜디오 ===")
print("👉 화면 가장자리에 문틀이나 책상 모서리를 비춰보세요!")
print("👉 [알파벳 q]: 종료")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # 3. 해상도 맞추기 (1280x720)
    resized_frame = cv2.resize(frame, (1280, 720))
    
    # 4. 왜곡 다림질 실행!
    undistorted_frame = cv2.undistort(resized_frame, mtx, dist, None, mtx)
    
    # 5. 한 화면에 두 개를 나란히 붙이기 위해 크기를 절반(640x360)으로 줄임
    small_original = cv2.resize(resized_frame, (640, 360))
    small_undistort = cv2.resize(undistorted_frame, (640, 360))
    
    # 글씨 써주기
    cv2.putText(small_original, "BEFORE (Distorted)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(small_undistort, "AFTER (Undistorted)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 두 화면을 가로로 이어 붙이기 (총 1280x360 크기가 됨)
    combined_frame = cv2.hconcat([small_original, small_undistort])
    
    # 6. 왜곡 확인용 빨간색 가이드라인(가로줄) 3개 긋기
    height, width, _ = combined_frame.shape
    cv2.line(combined_frame, (0, height//4), (width, height//4), (0, 0, 255), 1)
    cv2.line(combined_frame, (0, height//2), (width, height//2), (0, 0, 255), 1)
    cv2.line(combined_frame, (0, height*3//4), (width, height*3//4), (0, 0, 255), 1)

    cv2.imshow("Calibration Verification", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np

# 1. 왜곡 계수 불러오기
try:
    with np.load('calibration_matrix.npz') as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
    print("✅ 왜곡 다림질 수식 장전 완료!")
except:
    print("🚨 오류: calibration_matrix.npz 파일이 없습니다!")
    exit()

# 2. RTSP IP 카메라 연결
RTSP_URL = "rtsp://admin:capstone1234!@192.168.0.15:554/onvif1"
cap = cv2.VideoCapture(RTSP_URL)

print("\n=== 🔭 시야각 100% 보존 모드 (Alpha=1) 검증 스튜디오 ===")
print("👉 네 모서리의 '검은색 여백'을 관찰해 보세요!")
print("👉 [알파벳 q]: 종료")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # 3. 해상도 맞추기 (1280x720)
    resized_frame = cv2.resize(frame, (1280, 720))
    h, w = resized_frame.shape[:2]
    
    # ==========================================
    # ⭐️ 4. 수석 고문의 핵심 비기: Alpha 최적화 매트릭스
    # ==========================================
    # alpha=0: 빈 공간 없이 꽉 차게 잘라냄 (시야각 좁아짐)
    # alpha=1: 빈 공간(검은 여백)이 생겨도 원본 시야각 100% 보존!
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # 새로 계산된 매트릭스(newcameramtx)를 넣어서 다림질!
    undistorted_frame = cv2.undistort(resized_frame, mtx, dist, None, newcameramtx)
    
    # 5. 비교를 위해 화면 크기 줄여서 이어 붙이기
    small_original = cv2.resize(resized_frame, (640, 360))
    small_undistort = cv2.resize(undistorted_frame, (640, 360))
    
    cv2.putText(small_original, "BEFORE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(small_undistort, "AFTER (Alpha=1)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    combined_frame = cv2.hconcat([small_original, small_undistort])

    cv2.imshow("FOV Preserved Calibration", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

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

    print("\n=== 🔵 실시간 파란색 물병(Bottle) 좌표 추적 시작 ===")
    print("👉 딥러닝 없는 초고속 고전 비전 (Delay: 0.0초)")
    print("👉 [알파벳 q]: 스트리밍 종료\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("❌ 프레임을 유실했습니다.")
            break

        # [단계 1] 전처리 (1280x720 리사이즈 -> 180도 회전 -> 왜곡 다림질)
        resized_frame = cv2.resize(frame, (w, h))
        rotated_frame = cv2.rotate(resized_frame, cv2.ROTATE_180)
        undistorted_frame = cv2.remap(rotated_frame, mapx, mapy, cv2.INTER_LINEAR)

        # [단계 2] BGR 색상계를 인간의 눈과 비슷한 HSV 색상계로 변환
        hsv_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)

        # [단계 3] '파란색'에 해당하는 HSV 색상 범위 지정
        # (만약 파란색이 안 잡히면 이 수치를 미세 조정해야 합니다)
        lower_blue = np.array([90, 80, 50])   # 파란색의 하한값
        upper_blue = np.array([130, 255, 255]) # 파란색의 상한값

        # 지정한 범위의 색상만 남기고 나머지는 흑백(Mask)으로 쳐냅니다.
        mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # 노이즈(작은 파란색 점들) 제거를 위한 모폴로지 연산 (다듬기)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # [단계 4] 살아남은 파란색 덩어리(Contour)들의 외곽선 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_u, target_v = 0, 0 # 목표 픽셀 좌표 초기화

        if contours:
            # 가장 면적이 큰 파란색 덩어리를 물병으로 간주합니다.
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # 면적이 너무 작은 노이즈(예: 500픽셀 이하)는 무시
            if area > 500: 
                # 외곽선에 딱 맞는 네모 박스 그리기
                x, y, w_box, h_box = cv2.boundingRect(largest_contour)
                cv2.rectangle(undistorted_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

                # 무게 중심(Moment) 수학적 연산으로 물병의 정중앙 픽셀(u, v) 좌표 추출!
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    target_u = int(M["m10"] / M["m00"])
                    target_v = int(M["m01"] / M["m00"])

                    # 정중앙 픽셀에 빨간색 점 찍기
                    cv2.circle(undistorted_frame, (target_u, target_v), 5, (0, 0, 255), -1)
                    
                    # 화면에 추출된 좌표 출력 (이 좌표가 바로 로봇으로 넘길 데이터입니다!)
                    cv2.putText(undistorted_frame, f"Bottle (u:{target_u}, v:{target_v})", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # [출력] 메인 화면과 컴퓨터의 뇌가 보는 흑백 마스크 화면을 나란히 출력
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 가로로 붙이기 위해 채널 맞춤
        
        # 화면이 너무 크면 보기 힘드니 절반(640x360)으로 줄여서 모니터링
        view_main = cv2.resize(undistorted_frame, (640, 360))
        view_mask = cv2.resize(mask_colored, (640, 360))
        
        cv2.putText(view_mask, "AI Mask (Blue Only)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 두 화면 이어 붙이기
        combined_view = cv2.hconcat([view_main, view_mask])
        cv2.imshow("High-Speed Bottle Tracking", combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 스트리밍 종료")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
