import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter, ImageTk, ImageDraw
import numpy as np
import tkinter as tk
from tkinter import messagebox
import os
import cv2
import mediapipe as mp
from collections import deque

# =================設定區=================
MODEL_PATH = r"D:\AWORKSPACE\Github\Project_20260108\AIProject_KE\mnist_cnn.pth"
# =======================================

# --- 1. 模型架構 (不變) ---
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. 影像處理邏輯 (支援多筆畫) ---
def preprocess_points_to_tensor(points_list, device):
    """
    將座標點清單轉換為 Tensor，支援斷點 (None)
    """
    # 濾除全空的狀況
    valid_points = [p for p in points_list if p is not None]
    if len(valid_points) < 2:
        return None, None

    canvas_size = 400
    pil_image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(pil_image)
    
    # --- 關鍵修改：分段繪圖 ---
    # 我們將 points_list 根據 None 切割成很多小段 (strokes)
    strokes = []
    current_stroke = []
    
    for p in points_list:
        if p is None:
            if len(current_stroke) > 1:
                strokes.append(current_stroke)
            current_stroke = []
        else:
            current_stroke.append(p)
    # 把最後一段也加進去
    if len(current_stroke) > 1:
        strokes.append(current_stroke)

    # 畫出每一段
    for stroke in strokes:
        draw.line(stroke, fill=255, width=15, joint="curve")

    # --- 接下來的步驟與之前相同 (裁切、置中) ---
    bbox = pil_image.getbbox()
    if bbox:
        pil_image = pil_image.crop(bbox)
    else:
        return None, None

    target_size = 28
    digit_len = 20
    w, h = pil_image.size
    if max(w, h) == 0: return None, None

    ratio = digit_len / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    final_img = Image.new('L', (target_size, target_size), 0)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    final_img.paste(img_resized, (paste_x, paste_y))
    
    img_normal = final_img
    img_thick = img_normal.filter(ImageFilter.MaxFilter(3)) 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor_normal = transform(img_normal).unsqueeze(0).to(device)
    tensor_thick = transform(img_thick).unsqueeze(0).to(device)
    
    display_thumb = img_thick.resize((140, 140), Image.Resampling.NEAREST)
    return [tensor_normal, tensor_thick], display_thumb

# --- 3. GUI 應用程式類別 ---
class CameraDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 手勢隔空書寫辨識系統 (多筆畫支援版)")
        self.root.geometry("900x600")
        self.root.resizable(False, False)

        # --- MediaPipe 初始化 ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # --- OpenCV 初始化 ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("錯誤", "無法開啟攝影機")
            root.destroy()
            return
        
        # --- 繪圖狀態變數 ---
        self.draw_points = []
        self.is_drawing_active = False 
        self.gesture_status_text = "等待手勢..."
        
        # --- 防抖動 (Debouncing) 變數 ---
        self.history_len = 7
        self.gesture_buffer = deque(maxlen=self.history_len)

        # --- 模型載入 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvNet().to(self.device)
        self.load_model_weights()

        self.create_widgets()
        self.update_video_stream()

    def load_model_weights(self):
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}")
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
            except TypeError:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()
            print("模型載入成功！")
        except Exception as e:
            messagebox.showerror("錯誤", f"模型載入失敗：\n{str(e)}")

    def create_widgets(self):
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_frame = tk.Frame(main_container, width=640, height=480, bg="black")
        video_frame.pack(side=tk.LEFT, padx=10)
        self.video_label = tk.Label(video_frame)
        self.video_label.pack()

        control_panel = tk.Frame(main_container)
        control_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        tk.Label(control_panel, text="操作說明", font=("微軟正黑體", 14, "bold")).pack(pady=(0, 10))
        instruction = "1. 食指向上 = 落筆 (開始寫)\n2. 握拳 = 提筆 (斷開筆畫)\n3. 再次食指向上 = 寫下一筆"
        tk.Label(control_panel, text=instruction, font=("微軟正黑體", 12), justify=tk.LEFT, bg="#f0f0f0", padx=10, pady=10).pack(fill=tk.X)

        self.status_label_var = tk.StringVar(value="狀態: 等待手勢...")
        tk.Label(control_panel, textvariable=self.status_label_var, font=("微軟正黑體", 12, "bold"), fg="purple", pady=10).pack()

        tk.Label(control_panel, text="[AI 看到的影像]").pack(pady=(20,0))
        self.panel_processed = tk.Label(control_panel, text="等待輸入...", width=20, height=10, relief="sunken", bg="#ddd")
        self.panel_processed.pack()

        self.result_label = tk.Label(control_panel, text="預測：?", font=("Arial", 30, "bold"), fg="blue")
        self.result_label.pack(pady=10)
        self.confidence_label = tk.Label(control_panel, text="信心：--%", font=("Arial", 12))
        self.confidence_label.pack()

        btn_frame = tk.Frame(control_panel)
        btn_frame.pack(side=tk.BOTTOM, pady=20, fill=tk.X)
        tk.Button(btn_frame, text="清除軌跡", command=self.clear_canvas, bg="#ffcccc", font=("Arial", 12), height=2).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(btn_frame, text="開始辨識", command=self.predict, bg="#ccffcc", font=("Arial", 12, "bold"), height=2).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

    def get_current_gesture(self, lm_list):
        if not lm_list: return "NONE"

        index_tip_y = lm_list[8][2]
        index_pip_y = lm_list[6][2]
        
        middle_tip_y = lm_list[12][2]
        middle_pip_y = lm_list[10][2]
        
        ring_tip_y = lm_list[16][2]
        ring_pip_y = lm_list[14][2]
        
        pinky_tip_y = lm_list[20][2]
        pinky_pip_y = lm_list[18][2]

        index_up = index_tip_y < index_pip_y 
        
        margin = 0 
        middle_down = middle_tip_y > (middle_pip_y - margin)
        ring_down = ring_tip_y > (ring_pip_y - margin)
        pinky_down = pinky_tip_y > (pinky_pip_y - margin)

        if index_up and middle_down and ring_down and pinky_down:
            return "DRAW"
        
        index_down = index_tip_y > index_pip_y
        if index_down and middle_down and ring_down and pinky_down:
            return "STOP"
            
        return "UNKNOWN"

    def update_video_stream(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(img_rgb)
            
            current_frame_gesture = "NONE"
            index_finger_tip = None

            if results.multi_hand_landmarks:
                my_hand = results.multi_hand_landmarks[0]
                lm_list = []
                for id, lm in enumerate(my_hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])

                self.mp_draw.draw_landmarks(frame, my_hand, self.mp_hands.HAND_CONNECTIONS)
                current_frame_gesture = self.get_current_gesture(lm_list)
                index_finger_tip = (lm_list[8][1], lm_list[8][2])

            self.gesture_buffer.append(current_frame_gesture)
            
            most_common = max(set(self.gesture_buffer), key=self.gesture_buffer.count)
            occurrence = self.gesture_buffer.count(most_common)

            if occurrence >= 5:
                if most_common == "DRAW":
                    self.is_drawing_active = True
                    self.gesture_status_text = "狀態: ✍️ 繪圖中"
                elif most_common == "STOP":
                    self.is_drawing_active = False
                    self.gesture_status_text = "狀態: ✋ 提筆 (停止)"
                
            # --- 關鍵修正：軌跡記錄邏輯 ---
            if self.is_drawing_active and index_finger_tip:
                self.draw_points.append(index_finger_tip)
                cv2.circle(frame, index_finger_tip, 15, (0, 255, 0), cv2.FILLED)
            else:
                # 當「不」在畫圖時，檢查清單最後一個是不是 None
                # 如果不是，就補一個 None 當作斷點
                if self.draw_points and self.draw_points[-1] is not None:
                    self.draw_points.append(None)

            # --- 關鍵修正：螢幕繪製邏輯 (支援斷點) ---
            if len(self.draw_points) > 1:
                for i in range(1, len(self.draw_points)):
                    p1 = self.draw_points[i-1]
                    p2 = self.draw_points[i]
                    
                    # 如果任一點是斷點 (None)，就不要連線
                    if p1 is None or p2 is None:
                        continue
                        
                    cv2.line(frame, p1, p2, (0, 0, 255), 6)

            self.status_label_var.set(self.gesture_status_text)
            
            img_final = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img_final)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video_stream)

    def clear_canvas(self):
        self.draw_points = []
        self.result_label.config(text="預測：?")
        self.confidence_label.config(text="信心：--%", fg="black")
        self.panel_processed.config(image='', text="等待輸入...", width=20, height=10)

    def predict(self):
        # 檢查是否全空 (考慮到可能充滿了 None)
        valid_points = [p for p in self.draw_points if p is not None]
        if not valid_points:
            messagebox.showwarning("提示", "尚未繪製任何軌跡！")
            return

        tensors, display_thumb = preprocess_points_to_tensor(self.draw_points, self.device)

        if tensors is None:
             messagebox.showwarning("提示", "軌跡無效。")
             return

        best_confidence = -1
        best_prediction = None

        with torch.no_grad():
            for tensor in tensors:
                output = self.model(tensor)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
                conf = probs[0][pred].item() * 100
                
                if conf > best_confidence:
                    best_confidence = conf
                    best_prediction = pred

        photo_thumb = ImageTk.PhotoImage(display_thumb)
        self.panel_processed.config(image=photo_thumb, width=140, height=140)
        self.panel_processed.image = photo_thumb

        self.result_label.config(text=f"預測：{best_prediction}")
        color = "green" if best_confidence > 80 else "red"
        self.confidence_label.config(text=f"信心：{best_confidence:.2f}%", fg=color)

    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = CameraDigitApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"程式發生錯誤: {e}")
        if 'app' in locals() and hasattr(app, 'cap') and app.cap.isOpened():
            app.cap.release()