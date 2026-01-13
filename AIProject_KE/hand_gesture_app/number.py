"""
手勢數字辨識程式 (0~5) - 修正大拇指邏輯版
- 修正了大拇指的幾何判斷方向
- 解決 0變1, 5變4 的問題
"""

import cv2
import mediapipe as mp
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= 設定區 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'gesture_recognizer.task')

# 手部連線定義
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# 全域變數
recognition_result = None

def save_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global recognition_result
    recognition_result = result

def draw_landmarks_on_frame(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for idx, pt in enumerate(points):
            color = (0, 0, 255) if idx in [4, 8, 12, 16, 20] else (255, 0, 0)
            radius = 8 if idx in [4, 8, 12, 16, 20] else 5
            cv2.circle(frame, pt, radius, color, -1)

def count_fingers(hand_landmarks, handedness):
    """
    計算伸出的手指數量 (修正版)
    """
    fingers_status = [] 
    
    # --- 1. 判斷四指 (食指~小指) ---
    # 邏輯：指尖 (Tip) 高度 < 指節 (PIP) 高度 = 伸直
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers_status.append(1)
        else:
            fingers_status.append(0)

    # --- 2. 判斷大拇指 (修正區) ---
    # 這裡我們將邏輯完全反轉，以解決您的問題
    thumb_tip_x = hand_landmarks[4].x
    thumb_ip_x = hand_landmarks[3].x
    hand_label = handedness[0].category_name
    
    is_thumb_up = False
    
    # 之前的邏輯導致判定相反，這裡改用相反的符號
    if hand_label == "Right": 
        # 原本是 <，現在改為 >
        if thumb_tip_x > thumb_ip_x: 
            is_thumb_up = True
    else: # Left
        # 原本是 >，現在改為 <
        if thumb_tip_x < thumb_ip_x: 
            is_thumb_up = True
            
    fingers_status.insert(0, 1 if is_thumb_up else 0)
    
    return sum(fingers_status), fingers_status, hand_label

def main():
    global recognition_result

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=save_result
    )

    try:
        with vision.GestureRecognizer.create_from_options(options) as recognizer:
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("無法開啟攝影機")
                return

            # 設定解析度
            cap.set(3, 1280)
            cap.set(4, 720)

            print("=== 手指數字辨識 (修正版) 已啟動 ===")
            print("按 'q' 離開")

            while True:
                success, frame = cap.read()
                if not success: break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                recognizer.recognize_async(mp_image, int(time.time() * 1000))

                if recognition_result and recognition_result.hand_landmarks:
                    draw_landmarks_on_frame(frame, recognition_result.hand_landmarks)

                    for idx, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                        if idx < len(recognition_result.handedness):
                            handedness = recognition_result.handedness[idx]
                            
                            count, status, label = count_fingers(hand_landmarks, handedness)
                            
                            h, w, _ = frame.shape
                            wrist_x = int(hand_landmarks[0].x * w)
                            wrist_y = int(hand_landmarks[0].y * h)
                            
                            # 顯示數字
                            cv2.putText(frame, str(count), (wrist_x - 20, wrist_y - 20),
                                        cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 255), 4)
                            
                            # Debug 資訊：這行可以幫你看程式目前判定哪幾隻手指是開的 [拇, 食, 中, 無, 小]
                            # 如果還有錯，請告訴我這個列表顯示什麼 (例如 [1, 0, 0, 0, 0])
                            debug_str = f"{label}: {status}"
                            cv2.putText(frame, debug_str, (10, 40 + idx*40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Finger Counter Fixed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            cap.release()
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"錯誤: {e}")

if __name__ == "__main__":
    main()