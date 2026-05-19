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

'''

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
'''

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