#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np

def main():
    RTSP_URL = "rtsp://admin:kue0504950!@192.168.0.30:554/onvif1"
    cap = cv2.VideoCapture(RTSP_URL)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    
    # 버전 호환성 처리
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        use_new_api = True
    except AttributeError:
        use_new_api = False

    window_name = "ArUco ID Scanner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n=== 🔍 10초 컷 ArUco ID 스캐너 ===")
    print("👉 화면을 보고 바닥과 책상의 마커 번호를 메모장에 적어주세요!")
    print("👉 [q]를 누르면 종료됩니다.\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 보기 편하게 리사이즈 및 180도 회전만 적용 (빠른 스캔용)
        resized_frame = cv2.resize(frame, (1280, 720))
        rotated_frame = cv2.rotate(resized_frame, cv2.ROTATE_180)
        
        gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
        
        if use_new_api:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # 테두리 그리기
            aruco.drawDetectedMarkers(rotated_frame, corners, ids)
            
            # 번호를 화면에 아주 크게 띄우기
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0]
                center_u = int(np.mean(c[:, 0]))
                center_v = int(np.mean(c[:, 1]))
                
                # 글자 크기를 줄이고, 가독성을 높이기 위해 검은색 배경 박스 추가
                text = f"ID:{marker_id}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x, text_y = center_u - text_size[0] // 2, center_v - 15
                
                cv2.rectangle(rotated_frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(rotated_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(window_name, rotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()