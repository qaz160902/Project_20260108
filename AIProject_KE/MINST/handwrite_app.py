import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter, ImageTk, ImageDraw
import numpy as np
import tkinter as tk
from tkinter import messagebox
import os

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

# --- 2. 影像處理邏輯 (不變) ---
def preprocess_for_canvas(pil_image):
    img = pil_image.convert('L')
    img = ImageOps.invert(img)
    thresh = 50 
    fn = lambda x : 255 if x > thresh else 0
    img = img.point(fn, mode='L')

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    else:
        return None 

    target_size = 28
    digit_len = 20
    w, h = img.size
    
    if max(w, h) == 0: return None

    ratio = digit_len / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    new_img = Image.new('L', (target_size, target_size), 0)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    return new_img

# --- 3. GUI 應用程式類別 ---
class HandwriteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手寫數字辨識系統 (繪圖板版)")
        self.root.geometry("700x550")
        self.root.resizable(False, False)

        self.brush_size = 15  
        self.last_x, self.last_y = None, None
        
        self.canvas_width = 300
        self.canvas_height = 300
        self.image_in_memory = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image_in_memory)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvNet().to(self.device)
        self.load_model_weights()

        self.create_widgets()

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
        tk.Label(self.root, text="請在左側畫出 0-9 的數字", font=("微軟正黑體", 16)).pack(pady=10)

        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=10)

        # 左側
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(left_frame, text="[繪圖區]", font=("Arial", 10)).pack()
        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, bg="white", relief="sunken", bd=2)
        self.canvas.pack()
        
        self.canvas.bind("<Button-1>", self.activate_paint)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coords)

        # 右側
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(right_frame, text="[AI 視角]", font=("Arial", 10)).pack()
        
        # 這裡設定初始大小 (文字單位)
        self.panel_processed = tk.Label(right_frame, text="等待輸入...", width=20, height=10, relief="sunken", bg="#f0f0f0")
        self.panel_processed.pack(pady=5)

        self.result_label = tk.Label(right_frame, text="預測：?", font=("Arial", 30, "bold"), fg="blue")
        self.result_label.pack(pady=10)
        self.confidence_label = tk.Label(right_frame, text="信心：--%", font=("Arial", 12))
        self.confidence_label.pack()

        # 按鈕
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20, fill=tk.X)
        tk.Button(btn_frame, text="清除重寫", command=self.clear_canvas, bg="#ffcccc", font=("Arial", 12), width=12).pack(side=tk.LEFT, padx=50)
        tk.Button(btn_frame, text="開始辨識", command=self.predict, bg="#ccffcc", font=("Arial", 12, "bold"), width=12).pack(side=tk.RIGHT, padx=50)

    def activate_paint(self, event):
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line((self.last_x, self.last_y, x, y), 
                                    width=self.brush_size, fill='black', 
                                    capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line((self.last_x, self.last_y, x, y), fill=0, width=self.brush_size, joint="curve")
        self.last_x, self.last_y = x, y

    def reset_coords(self, event):
        self.last_x, self.last_y = None, None

    # --- 修正後的清除功能 ---
    def clear_canvas(self):
        # 清除螢幕畫布
        self.canvas.delete("all")
        
        # 清除記憶體畫布
        self.image_in_memory = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image_in_memory)
        
        # 重設結果文字
        self.result_label.config(text="預測：?")
        self.confidence_label.config(text="信心：--%", fg="black")
        
        # 【關鍵修正】
        # 移除圖片 (image='') 的同時，強制重設 width=20 (文字單位), height=10 (文字單位)
        # 這樣它才不會變成 140 個字寬的巨型標籤
        self.panel_processed.config(image='', text="等待輸入...", width=20, height=10)

    def predict(self):
        processed_img = preprocess_for_canvas(self.image_in_memory)
        
        if processed_img is None:
            messagebox.showwarning("提示", "畫布是空白的，請先寫個數字！")
            return

        img_normal = processed_img
        img_thick = img_normal.filter(ImageFilter.MaxFilter(3))

        candidates = [
            {"name": "一般", "img": img_normal},
            {"name": "加粗", "img": img_thick}
        ]
        
        best_confidence = -1
        best_result = None

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        for candidate in candidates:
            img_tensor = transform(candidate["img"]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(img_tensor)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
                conf = probs[0][pred].item() * 100
            
            if conf > best_confidence:
                best_confidence = conf
                best_result = {
                    "pred": pred,
                    "conf": conf,
                    "img": candidate["img"],
                    "type": candidate["name"]
                }

        # 顯示圖片 (這時 width/height 會變成像素單位)
        display_img = best_result["img"].resize((140, 140), Image.Resampling.NEAREST)
        photo_processed = ImageTk.PhotoImage(display_img)
        
        # 設定為像素尺寸 (140px)
        self.panel_processed.config(image=photo_processed, width=140, height=140)
        self.panel_processed.image = photo_processed

        self.result_label.config(text=f"預測：{best_result['pred']}")
        color = "green" if best_result['conf'] > 80 else "red"
        self.confidence_label.config(
            text=f"信心：{best_result['conf']:.2f}% ({best_result['type']})", 
            fg=color
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = HandwriteApp(root)
    root.mainloop()