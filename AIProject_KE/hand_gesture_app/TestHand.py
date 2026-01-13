import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

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
        for pt in points:
            cv2.circle(frame, pt, 5, (255, 0, 0), -1)

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    result_callback=save_result
)

with vision.GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(1)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # 先翻轉再處理
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp = time.time_ns() // 1_000_000
        recognizer.recognize_async(mp_image, timestamp)

        if recognition_result:
            draw_landmarks_on_frame(frame, recognition_result.hand_landmarks)
            if recognition_result.gestures:
                for i, gesture in enumerate(recognition_result.gestures):
                    category_name = gesture[0].category_name
                    score = round(gesture[0].score, 2)
                    cv2.putText(frame, f"{category_name} ({score})", (50, 50 + (i * 40)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Hand Chart', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()