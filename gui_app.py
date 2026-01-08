import os
# 強制使用 Legacy Keras (Keras 2) 以相容 Teachable Machine 模型
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# 關閉 OneDNN 最佳化訊息，減少干擾
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from tensorflow.keras.models import load_model
except ImportError:
    raise ImportError("無法匯入 keras。請確認已安裝 'tf_keras' 套件：pip install tf-keras")
import threading

# 設定外觀模式
ctk.set_appearance_mode("Light")  # 使用亮色模式以符合您的白底設計
ctk.set_default_color_theme("blue")

class AIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- 視窗設定 ---
        self.title("AI 辨識應用程式")
        self.geometry("900x600")
        
        # 設定 Grid 權重，讓畫面可以縮放但保持比例
        self.grid_columnconfigure(0, weight=3) # 左邊畫面區 (較寬)
        self.grid_columnconfigure(1, weight=1) # 右邊控制區 (較窄)
        self.grid_rowconfigure(0, weight=1)    # 標題區 (較矮)
        self.grid_rowconfigure(1, weight=5)    # 主要內容區 (較高)

        # --- 1. UI 標題 (藍色) ---
        # 對應圖中上方的藍色長條
        self.header_frame = ctk.CTkFrame(self, fg_color="#4267B2", corner_radius=20)
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=20, pady=(20, 10))
        
        self.header_label = ctk.CTkLabel(self.header_frame, text="UI 標題", font=("Microsoft JhengHei", 24, "bold"), text_color="white")
        self.header_label.place(relx=0.5, rely=0.5, anchor="center")

        # --- 2. 畫面 Canva (綠色 -> 攝影機顯示) ---
        # 對應圖中左方的綠色方塊
        # 這裡我們使用 Label 來顯示影像，初始顏色設為綠色以符合您的設計圖
        self.camera_frame = ctk.CTkLabel(self, text="畫面\nCanva(畫布)\n(攝影機啟動中...)", 
                                         fg_color="#7CB342", 
                                         text_color="white",
                                         font=("Microsoft JhengHei", 20),
                                         corner_radius=0) # 直角
        self.camera_frame.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(10, 20))

        # --- 右側容器 (用來包裝按鈕和結果) ---
        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(10, 20))
        
        # 讓右側容器內部也分上下
        self.right_panel.grid_rowconfigure(0, weight=0) # 按鈕區
        self.right_panel.grid_rowconfigure(1, weight=1) # 結果區 (佔據剩餘空間)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # --- 3. 辨識按鈕 (灰色) ---
        # 對應圖中右上的灰色按鈕
        self.predict_btn = ctk.CTkButton(self.right_panel, 
                                         text="辨識(BTN)", 
                                         font=("Microsoft JhengHei", 18),
                                         fg_color="#9E9E9E", 
                                         hover_color="#757575",
                                         height=50,
                                         command=self.predict_current_frame)
        self.predict_btn.grid(row=0, column=0, sticky="ew", pady=(0, 20))

        # --- 4. AI 結果 (橘色) ---
        # 對應圖中右下的橘色方塊
        self.result_label = ctk.CTkLabel(self.right_panel, 
                                         text="AI結果\n(Label)", 
                                         font=("Microsoft JhengHei", 24, "bold"),
                                         fg_color="#EF6C00", 
                                         text_color="white",
                                         corner_radius=20)
        self.result_label.grid(row=1, column=0, sticky="nsew")

        # --- AI 模型初始化 ---
        self.model = None
        self.class_names = []
        self.load_ai_model()

        # --- 攝影機初始化 ---
        self.cap = None
        self.current_image = None # 儲存當前畫面供辨識用
        self.start_camera()

    def load_ai_model(self):
        try:
            # 載入模型
            print("Loading model...")
            self.model = load_model("model/keras_model.h5", compile=False)
            
            # 載入標籤
            with open("model/labels.txt", "r", encoding="utf-8") as f: # 加入 utf-8 避免中文亂碼
                self.class_names = f.readlines()
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.result_label.configure(text=f"模型載入失敗:\n{e}")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.update_camera()

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            # 1. 轉成 RGB (OpenCV 預設是 BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. 儲存原始影像供 AI 辨識使用 (不需縮放顯示的這份，保留原尺寸較好，或是同步處理)
            self.current_frame_for_ai = frame_rgb.copy()

            # 3. 轉換成 CTk 可顯示的格式
            # 為了效能，我們可以將顯示的圖片縮放到符合 UI 的大小 (選擇性)
            img = Image.fromarray(frame_rgb)
            
            # 取得目前 Label 的大小來動態調整圖片 (保持 Cover 模式或 Fit 模式)
            # 這裡我們做簡單的 Resize 以符合視窗
            w = self.camera_frame.winfo_width()
            h = self.camera_frame.winfo_height()
            
            if w > 10 and h > 10: # 避免初始化時尺寸為 1
                 # 保持比例縮放
                img_ratio = img.width / img.height
                target_ratio = w / h
                
                if target_ratio > img_ratio:
                    new_h = h
                    new_w = int(h * img_ratio)
                else:
                    new_w = w
                    new_h = int(w / img_ratio)
                    
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
            
            # 更新 UI
            self.camera_frame.configure(image=ctk_img, text="") # 清空文字，顯示圖片
            
        # 每 10 毫秒呼叫一次自己，形成迴圈
        self.after(10, self.update_camera)

    def predict_current_frame(self):
        if self.model is None or self.current_frame_for_ai is None:
            return

        # 1. 影像前處理 (參照 opencv_tm.py)
        # Resize to 224x224
        img = cv2.resize(self.current_frame_for_ai, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Make the image a numpy array and reshape it to the models input shape.
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        img = (img / 127.5) - 1

        # 2. 預測
        prediction = self.model.predict(img, verbose=0)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]

        # 3. 處理文字 (移除換行符號和前綴編號)
        class_text = class_name.strip()
        # 假設 labels.txt 格式是 "0 ClassName"，我們試圖去掉前面的數字
        if " " in class_text:
            class_text = class_text.split(" ", 1)[1]

        # 4. 更新橘色 Label
        result_text = f"類別: {class_text}\n信心度: {int(confidence_score * 100)}%"
        self.result_label.configure(text=result_text)

    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = AIApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
