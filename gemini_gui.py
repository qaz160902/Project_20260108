import os
import threading
import time
from datetime import datetime
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

# --- 環境設定 ---
# 強制使用 Legacy Keras (Keras 2) 以相容 Teachable Machine 模型
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# 關閉 OneDNN 最佳化訊息
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("警告: 未安裝 tf_keras，模型可能無法載入。")

# --- GUI 設定 ---
ctk.set_appearance_mode("Dark")  # 預設深色模式
ctk.set_default_color_theme("dark-blue")  # 主題色

class GeminiVisionPro(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. 視窗基礎設定
        self.title("Gemini Vision Pro - AI 辨識系統")
        self.geometry("1100x700")
        
        # 2. 變數初始化
        self.model = None
        self.class_names = []
        self.cap = None
        self.is_running = True
        self.current_frame = None
        self.auto_predict_mode = False # 自動辨識模式狀態
        self.last_predict_time = 0
        self.model_load_error = None
        
        # 3. 介面佈局 (Grid Layout)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- 左側側邊欄 (Sidebar) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Gemini Vision", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_label_1 = ctk.CTkLabel(self.sidebar_frame, text="控制面板", anchor="w")
        self.sidebar_label_1.grid(row=1, column=0, padx=20, pady=(10, 0))

        # 自動辨識開關
        self.auto_predict_switch = ctk.CTkSwitch(self.sidebar_frame, text="即時自動辨識", command=self.toggle_auto_predict)
        self.auto_predict_switch.grid(row=2, column=0, padx=20, pady=10)

        # 手動辨識按鈕
        self.manual_btn = ctk.CTkButton(self.sidebar_frame, text="單次快照辨識", command=self.predict_once)
        self.manual_btn.grid(row=3, column=0, padx=20, pady=10)

        # 外觀模式選單
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="外觀主題:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))

        # --- 右側主畫面區 ---
        self.main_panel = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_panel.grid(row=0, column=1, sticky="nsew")
        self.main_panel.grid_rowconfigure(0, weight=3) # 影像區佔比大
        self.main_panel.grid_rowconfigure(1, weight=1) # 儀表板佔比小
        self.main_panel.grid_columnconfigure(0, weight=1)

        # A. 視訊顯示區
        self.video_frame = ctk.CTkFrame(self.main_panel, fg_color="#1a1a1a", corner_radius=15)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_rowconfigure(0, weight=1)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="正在啟動攝影機...", font=("Arial", 16))
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # B. 資訊儀表板 (Dashboard)
        self.dashboard_frame = ctk.CTkFrame(self.main_panel, corner_radius=15, fg_color=("gray85", "gray20"))
        self.dashboard_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.dashboard_frame.grid_columnconfigure(0, weight=1) # 結果
        self.dashboard_frame.grid_columnconfigure(1, weight=1) # 信心度
        self.dashboard_frame.grid_columnconfigure(2, weight=1) # 歷史紀錄

        # B-1. 辨識結果卡片
        self.result_card = ctk.CTkFrame(self.dashboard_frame, fg_color="transparent")
        self.result_card.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.lbl_result_title = ctk.CTkLabel(self.result_card, text="當前辨識結果", font=("Microsoft JhengHei", 14))
        self.lbl_result_title.pack(pady=(10, 0))
        
        self.lbl_class_name = ctk.CTkLabel(self.result_card, text="等待中...", font=("Microsoft JhengHei", 36, "bold"), text_color="#3B8ED0")
        self.lbl_class_name.pack(pady=10)

        # B-2. 信心度卡片
        self.confidence_card = ctk.CTkFrame(self.dashboard_frame, fg_color="transparent")
        self.confidence_card.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.lbl_conf_title = ctk.CTkLabel(self.confidence_card, text="AI 信心指數", font=("Microsoft JhengHei", 14))
        self.lbl_conf_title.pack(pady=(10, 5))
        
        self.progress_bar = ctk.CTkProgressBar(self.confidence_card, orientation="horizontal", height=15)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10, fill="x", padx=20)
        
        self.lbl_conf_val = ctk.CTkLabel(self.confidence_card, text="0%", font=("Arial", 16))
        self.lbl_conf_val.pack()

        # B-3. 歷史紀錄 (Scrollable)
        self.history_frame = ctk.CTkScrollableFrame(self.dashboard_frame, label_text="近期紀錄", height=100)
        self.history_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)

        # 4. 啟動後台任務
        # 使用執行緒載入模型，避免卡住 UI
        self.lbl_class_name.configure(text="模型載入中...")
        threading.Thread(target=self.load_model_task, daemon=True).start()
        self.check_model_load_status()
        
        # 啟動攝影機
        self.start_camera()

    def load_model_task(self):
        """後台載入模型 (不操作 UI)"""
        try:
            # 模擬一點延遲讓使用者看到載入狀態 (可選)
            time.sleep(0.5) 
            
            _model = load_model("model/keras_model.h5", compile=False)
            
            with open("model/labels.txt", "r", encoding="utf-8") as f:
                self.class_names = f.readlines()
            
            self.model = _model
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error: {e}")
            self.model_load_error = str(e)

    def check_model_load_status(self):
        """定期檢查模型載入狀態 (主執行緒)"""
        if self.model is not None:
            self.lbl_class_name.configure(text="準備就緒", text_color="white")
        elif self.model_load_error is not None:
            self.lbl_class_name.configure(text="載入失敗", text_color="red")
        else:
            self.after(100, self.check_model_load_status)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.update_camera()

    def update_camera(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            # 1. 影像處理
            frame = cv2.flip(frame, 1) # 左右鏡像，像照鏡子一樣
            self.current_frame = frame # 保存原始 BGR 影像供辨識
            
            # 2. 轉為 RGB 供顯示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # 3. 智慧縮放 (保持比例填滿視窗)
            # 取得當前顯示區域的大小
            display_w = self.video_frame.winfo_width()
            display_h = self.video_frame.winfo_height()
            
            if display_w > 10 and display_h > 10:
                # 計算縮放比例，使用 "Cover" 模式 (填滿) 或 "Contain" 模式 (完整顯示)
                # 這裡使用 Contain 模式確保畫面完整
                img_ratio = img.width / img.height
                target_ratio = display_w / display_h
                
                if target_ratio > img_ratio:
                    new_h = display_h
                    new_w = int(display_h * img_ratio)
                else:
                    new_w = display_w
                    new_h = int(display_w / img_ratio)
                
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
            self.video_label.configure(image=ctk_img, text="")

            # 4. 自動辨識邏輯
            if self.auto_predict_mode and self.model:
                # 限制辨識頻率，例如每 0.1 秒一次，避免過度消耗資源
                if time.time() - self.last_predict_time > 0.1: 
                    self.perform_prediction()
                    self.last_predict_time = time.time()

        self.after(10, self.update_camera)

    def toggle_auto_predict(self):
        self.auto_predict_mode = self.auto_predict_switch.get()
        if self.auto_predict_mode:
            self.manual_btn.configure(state="disabled", text="自動模式中...")
        else:
            self.manual_btn.configure(state="normal", text="單次快照辨識")

    def predict_once(self):
        if self.model:
            self.perform_prediction()

    def perform_prediction(self):
        if self.current_frame is None: 
            return

        # 影像前處理
        img = cv2.resize(self.current_frame, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1

        # 預測
        prediction = self.model.predict(img, verbose=0)
        index = np.argmax(prediction)
        class_name = self.class_names[index].strip()
        confidence_score = prediction[0][index]

        # 清理文字 (去除前面的數字 "0 ", "1 " 等)
        display_name = class_name
        if " " in display_name:
            display_name = display_name.split(" ", 1)[1]

        # 更新 UI
        self.update_dashboard(display_name, confidence_score)

    def update_dashboard(self, class_name, confidence):
        # 1. 更新結果文字
        self.lbl_class_name.configure(text=class_name)
        
        # 2. 更新信心度條與顏色
        self.progress_bar.set(confidence)
        self.lbl_conf_val.configure(text=f"{int(confidence * 100)}%")
        
        # 根據信心度改變文字顏色 (視覺回饋)
        if confidence > 0.8:
            self.lbl_class_name.configure(text_color="#2CC985") # 綠色
        elif confidence > 0.5:
            self.lbl_class_name.configure(text_color="#F9AA33") # 黃色
        else:
            self.lbl_class_name.configure(text_color="#E53935") # 紅色

        # 3. 新增歷史紀錄 (只在信心度高時記錄，避免雜訊)
        if confidence > 0.7:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_text = f"[{timestamp}] {class_name} ({int(confidence*100)}%)"
            # 插入到最上方
            log_label = ctk.CTkLabel(self.history_frame, text=log_text, anchor="w")
            log_label.pack(fill="x", pady=2)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

if __name__ == "__main__":
    app = GeminiVisionPro()
    app.mainloop()