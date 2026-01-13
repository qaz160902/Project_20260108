import cv2
import mediapipe as mp
import time
import os
import random
import math
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= 遊戲參數設定 =================
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 障礙物設定
OBSTACLE_SPEED = 9          
OBSTACLE_FREQUENCY = 65     
OBSTACLE_WIDTH = 80
GAP_SIZE = 280              

# 物理與球設定
BALL_RADIUS = 25
BALL_COLOR = (0, 140, 255)  
GRAVITY = 0.6               

# 【調整】物理參數修改區
# 減弱拍擊力道 (原本 -22 -> 改為 -17)
KICK_FORCE = -17            
# 降低地板反彈係數 (原本 0.7 -> 改為 0.5)
BOUNCE_DAMPING = 0.5        
MAX_FALL_SPEED = 25         

# 長方形板子設定
PADDLE_WIDTH = 180          
PADDLE_HEIGHT = 40          
PADDLE_COLOR = (255, 255, 0)

# ================= MediaPipe 設定 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'gesture_recognizer.task')

# ================= 遊戲類別 =================
class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.game_over = False
        self.score = 0
        self.obstacles = []
        self.frame_count = 0
        
        # 物理變數
        self.ball_x = 250            
        self.ball_y = WINDOW_HEIGHT // 2
        self.ball_vel_y = 0          

    def update_physics(self, paddle_cx, paddle_cy, paddle_active):
        """處理重力與長方形板子互動"""
        if self.game_over or not self.active:
            return

        # 1. 施加重力
        self.ball_vel_y += GRAVITY
        if self.ball_vel_y > MAX_FALL_SPEED:
            self.ball_vel_y = MAX_FALL_SPEED

        # 2. 更新球的位置
        self.ball_y += self.ball_vel_y

        # 3. 檢測長方形碰撞 (AABB 碰撞)
        if paddle_active:
            # 計算板子的四個邊界
            rect_left = paddle_cx - (PADDLE_WIDTH // 2)
            rect_right = paddle_cx + (PADDLE_WIDTH // 2)
            rect_top = paddle_cy - (PADDLE_HEIGHT // 2)
            rect_bottom = paddle_cy + (PADDLE_HEIGHT // 2)

            # 碰撞檢測
            is_x_overlap = (self.ball_x + BALL_RADIUS > rect_left) and (self.ball_x - BALL_RADIUS < rect_right)
            is_y_overlap = (self.ball_y + BALL_RADIUS > rect_top) and (self.ball_y - BALL_RADIUS < rect_bottom)

            if is_x_overlap and is_y_overlap:
                # 只有當球是「往下掉」的時候才反彈
                if self.ball_vel_y > 0:
                    self.ball_vel_y = KICK_FORCE
                    # 強制修正位置到板子正上方
                    self.ball_y = rect_top - BALL_RADIUS - 2

        # 4. 邊界檢查
        # 地板
        if self.ball_y >= WINDOW_HEIGHT - BALL_RADIUS:
            self.ball_y = WINDOW_HEIGHT - BALL_RADIUS
            # 反彈力道減弱
            self.ball_vel_y *= -BOUNCE_DAMPING
            if abs(self.ball_vel_y) < 2: self.ball_vel_y = 0

        # 天花板
        if self.ball_y <= BALL_RADIUS:
            self.ball_y = BALL_RADIUS
            self.ball_vel_y *= -0.5 

    def update_obstacles(self):
        if self.game_over or not self.active:
            return

        self.frame_count += 1
        
        # 生成障礙物
        if self.frame_count % OBSTACLE_FREQUENCY == 0:
            min_gap_y = GAP_SIZE // 2 + 50
            max_gap_y = WINDOW_HEIGHT - (GAP_SIZE // 2) - 50
            gap_center = random.randint(min_gap_y, max_gap_y)
            self.obstacles.append({'x': WINDOW_WIDTH, 'gap_y': gap_center, 'passed': False})

        # 移動障礙物
        for obs in self.obstacles:
            obs['x'] -= OBSTACLE_SPEED

        # 移除與計分
        if self.obstacles:
            if self.obstacles[0]['x'] < -OBSTACLE_WIDTH:
                self.obstacles.pop(0)
            
            front_obs = self.obstacles[0]
            if not front_obs['passed'] and front_obs['x'] + OBSTACLE_WIDTH < self.ball_x - BALL_RADIUS:
                self.score += 1
                front_obs['passed'] = True

    def check_collision(self):
        for obs in self.obstacles:
            if obs['x'] < self.ball_x + BALL_RADIUS and obs['x'] + OBSTACLE_WIDTH > self.ball_x - BALL_RADIUS:
                top_pipe_bottom = obs['gap_y'] - (GAP_SIZE // 2)
                bottom_pipe_top = obs['gap_y'] + (GAP_SIZE // 2)

                if (self.ball_y - BALL_RADIUS < top_pipe_bottom) or \
                   (self.ball_y + BALL_RADIUS > bottom_pipe_top):
                    return True
        return False

# 全域變數
game = GameState()
recognition_result = None

def save_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global recognition_result
    recognition_result = result

def main():
    global recognition_result

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=save_result
    )

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)

    print("=== 穩重版手掌板子遊戲啟動 ===")
    print("參數已調整：彈力減弱，操控更穩定")
    print("R: 開始/重來, Q: 離開")

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            recognizer.recognize_async(mp_image, int(time.time() * 1000))

            # --- 1. 取得手掌中心 ---
            paddle_cx, paddle_cy = 0, 0
            paddle_active = False

            if recognition_result and recognition_result.hand_landmarks:
                if len(recognition_result.hand_landmarks) > 0:
                    hand = recognition_result.hand_landmarks[0]
                    
                    paddle_cx = int(hand[9].x * WINDOW_WIDTH)
                    paddle_cy = int(hand[9].y * WINDOW_HEIGHT)
                    paddle_active = True

                    # 繪製長方形板子
                    rect_x1 = paddle_cx - (PADDLE_WIDTH // 2)
                    rect_y1 = paddle_cy - (PADDLE_HEIGHT // 2)
                    rect_x2 = paddle_cx + (PADDLE_WIDTH // 2)
                    rect_y2 = paddle_cy + (PADDLE_HEIGHT // 2)

                    # 畫板子
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), PADDLE_COLOR, -1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), PADDLE_COLOR, 3)
                    cv2.circle(frame, (paddle_cx, paddle_cy), 5, (255, 255, 255), -1)

            # --- 2. 遊戲邏輯 ---
            if game.active and not game.game_over:
                game.update_physics(paddle_cx, paddle_cy, paddle_active)
                game.update_obstacles()

                if game.check_collision():
                    game.game_over = True

                # 繪製球
                cv2.circle(frame, (int(game.ball_x), int(game.ball_y)), BALL_RADIUS, BALL_COLOR, -1)
                cv2.circle(frame, (int(game.ball_x)-8, int(game.ball_y)-8), 8, (255, 255, 255), -1)

                # 繪製障礙物
                for obs in game.obstacles:
                    top_h = obs['gap_y'] - (GAP_SIZE // 2)
                    bottom_y = obs['gap_y'] + (GAP_SIZE // 2)
                    
                    cv2.rectangle(frame, (obs['x'], 0), (obs['x']+OBSTACLE_WIDTH, top_h), (0, 255, 0), -1)
                    cv2.rectangle(frame, (obs['x'], 0), (obs['x']+OBSTACLE_WIDTH, top_h), (0, 100, 0), 3)
                    
                    cv2.rectangle(frame, (obs['x'], bottom_y), (obs['x']+OBSTACLE_WIDTH, WINDOW_HEIGHT), (0, 255, 0), -1)
                    cv2.rectangle(frame, (obs['x'], bottom_y), (obs['x']+OBSTACLE_WIDTH, WINDOW_HEIGHT), (0, 100, 0), 3)

                # 分數
                cv2.putText(frame, str(game.score), (WINDOW_WIDTH//2, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 3)

            elif game.game_over:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                
                cv2.putText(frame, "GAME OVER", (WINDOW_WIDTH//2 - 180, WINDOW_HEIGHT//2), 
                            cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255), 4)
                cv2.putText(frame, f"Score: {game.score}", (WINDOW_WIDTH//2 - 80, WINDOW_HEIGHT//2 + 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'R' to Restart", (WINDOW_WIDTH//2 - 140, WINDOW_HEIGHT//2 + 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 1)

            else:
                # 待機畫面
                cv2.putText(frame, "PADDLE GAME", (WINDOW_WIDTH//2 - 200, WINDOW_HEIGHT//2 - 50), 
                            cv2.FONT_HERSHEY_TRIPLEX, 2.5, (0, 165, 255), 3)
                cv2.putText(frame, "Hit the ball with the Rectangle!", (WINDOW_WIDTH//2 - 230, WINDOW_HEIGHT//2 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                dummy_y = int(WINDOW_HEIGHT//2 + math.sin(time.time()*5)*30)
                cv2.circle(frame, (WINDOW_WIDTH//2, dummy_y + 100), BALL_RADIUS, BALL_COLOR, -1)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                game.reset()
                game.active = True

            cv2.imshow('Rect Paddle Game (Stable)', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()