"""
食指踢球應用
- 用食指碰撞球，球會被踢走
- 球碰到牆壁會反彈
"""

import cv2
import mediapipe as mp
import time
import os
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === 設定參數 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'gesture_recognizer.task')
CONFIDENCE_THRESHOLD = 0.5

# 球的參數
BALL_RADIUS = 30
BALL_COLOR = (0, 100, 255)  # 橘色
FRICTION = 0.98  # 摩擦力 (每帧速度衰減)
KICK_POWER = 15  # 踢球力道
BOUNCE_DAMPING = 0.8  # 反彈時的能量損失
HOLD_SPEED_THRESHOLD = 8  # 低於此速度時，球會停在手上
HOLD_DAMPING = 0.3  # 停在手上時的阻尼 (越小越黏)
GRAVITY = 0.5  # 重力加速度

# === 初始化變數 ===
recognition_result = None

# 球的狀態
ball_x, ball_y = 320, 240  # 位置
ball_vx, ball_vy = 0, 0     # 速度

# 食指位置追蹤 (用於計算移動方向)
prev_finger_x, prev_finger_y = 0, 0

# 食指指尖的 landmark 索引
INDEX_FINGER_TIP = 8

# 手部連線定義
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


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

        # 特別標記食指指尖
        cv2.circle(frame, points[INDEX_FINGER_TIP], 12, (0, 255, 255), -1)


def get_index_finger_position(hand_landmarks, frame_width, frame_height):
    """取得食指指尖的位置"""
    index_tip = hand_landmarks[INDEX_FINGER_TIP]
    x = int(index_tip.x * frame_width)
    y = int(index_tip.y * frame_height)
    return x, y


def check_collision(finger_x, finger_y, ball_x, ball_y, radius):
    """檢查食指是否碰到球"""
    dist = math.sqrt((finger_x - ball_x) ** 2 + (finger_y - ball_y) ** 2)
    return dist < radius + 15  # 15 是食指的碰撞半徑


def main():
    global recognition_result, ball_x, ball_y, ball_vx, ball_vy
    global prev_finger_x, prev_finger_y

    # 初始化 MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=CONFIDENCE_THRESHOLD,
        min_hand_presence_confidence=CONFIDENCE_THRESHOLD,
        min_tracking_confidence=CONFIDENCE_THRESHOLD,
        result_callback=save_result
    )

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("=" * 50)
        print("食指踢球應用")
        print("=" * 50)
        print("用食指踢球，球會反彈!")
        print("按 ESC 離開")
        print("=" * 50)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp = time.time_ns() // 1_000_000
            recognizer.recognize_async(mp_image, timestamp)

            # === 物理更新 ===
            # 重力
            ball_vy += GRAVITY

            # 更新球的位置
            ball_x += ball_vx
            ball_y += ball_vy

            # 摩擦力 (只對水平方向)
            ball_vx *= FRICTION

            # 牆壁反彈
            if ball_x - BALL_RADIUS < 0:
                ball_x = BALL_RADIUS
                ball_vx = -ball_vx * BOUNCE_DAMPING
            elif ball_x + BALL_RADIUS > w:
                ball_x = w - BALL_RADIUS
                ball_vx = -ball_vx * BOUNCE_DAMPING

            if ball_y - BALL_RADIUS < 0:
                ball_y = BALL_RADIUS
                ball_vy = -ball_vy * BOUNCE_DAMPING
            elif ball_y + BALL_RADIUS > h:
                ball_y = h - BALL_RADIUS
                ball_vy = -ball_vy * BOUNCE_DAMPING

            finger_detected = False

            if recognition_result and recognition_result.hand_landmarks:
                # 繪製手部骨架
                draw_landmarks_on_frame(frame, recognition_result.hand_landmarks)

                # 取得食指位置
                hand_landmarks = recognition_result.hand_landmarks[0]
                finger_x, finger_y = get_index_finger_position(hand_landmarks, w, h)
                finger_detected = True

                # 計算食指移動速度
                finger_vx = finger_x - prev_finger_x
                finger_vy = finger_y - prev_finger_y

                # 檢查碰撞
                if check_collision(finger_x, finger_y, ball_x, ball_y, BALL_RADIUS):
                    # 計算食指移動速度
                    finger_speed = math.sqrt(finger_vx ** 2 + finger_vy ** 2)

                    if finger_speed < HOLD_SPEED_THRESHOLD:
                        # 速度慢 → 球停在手上 (跟隨食指)
                        # 計算目標位置 (食指前方一點)
                        target_x = finger_x
                        target_y = finger_y - BALL_RADIUS - 10  # 球在食指上方

                        # 阻尼跟隨
                        ball_x += (target_x - ball_x) * HOLD_DAMPING
                        ball_y += (target_y - ball_y) * HOLD_DAMPING
                        ball_vx *= 0.5  # 快速減速
                        ball_vy *= 0.5
                    else:
                        # 速度快 → 踢球
                        dx = ball_x - finger_x
                        dy = ball_y - finger_y
                        dist = math.sqrt(dx * dx + dy * dy)

                        if dist > 0:
                            # 正規化方向
                            dx /= dist
                            dy /= dist

                            # 計算踢球力道
                            kick = max(KICK_POWER, finger_speed * 0.8)

                            ball_vx = dx * kick + finger_vx * 0.5
                            ball_vy = dy * kick + finger_vy * 0.5

                            # 把球推出碰撞範圍
                            ball_x = finger_x + dx * (BALL_RADIUS + 20)
                            ball_y = finger_y + dy * (BALL_RADIUS + 20)

                # 更新前一幀位置
                prev_finger_x = finger_x
                prev_finger_y = finger_y

            # 繪製球 (有速度時加上運動模糊效果)
            speed = math.sqrt(ball_vx ** 2 + ball_vy ** 2)
            if speed > 5:
                # 運動軌跡
                trail_x = int(ball_x - ball_vx * 2)
                trail_y = int(ball_y - ball_vy * 2)
                cv2.line(frame, (trail_x, trail_y), (int(ball_x), int(ball_y)), (0, 50, 150), 8)

            cv2.circle(frame, (int(ball_x), int(ball_y)), BALL_RADIUS, BALL_COLOR, -1)
            cv2.circle(frame, (int(ball_x), int(ball_y)), BALL_RADIUS, (0, 50, 150), 3)

            # 顯示狀態
            status = "Finger Detected" if finger_detected else "No Hand"
            color = (0, 255, 0) if finger_detected else (128, 128, 128)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Speed: {speed:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Finger Ball - Kick it!', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()