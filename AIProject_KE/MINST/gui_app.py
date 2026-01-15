import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# =================設定區=================
# 請確認這裡的模型路徑是正確的
MODEL_PATH = r"D:\AWORKSPACE\Github\Project_20260108\AIProject_KE\mnist_cnn.pth"
# =======================================

# --- 1. 模型架構 (必須與訓練時一致) ---
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

# --- 2. 影像處理邏輯 ---
def preprocess_standard(pil_image):
    """ 基礎預處理：轉灰階、反轉、二值化、裁切、置中 """
    img = pil_image.convert('L')
    
    # 反轉 (白底黑字 -> 黑底白字)
    if np.mean(img) > 127: 
        img = ImageOps.invert(img)
    
    # 二值化
    thresh = 50 
    fn = lambda x : 255 if x > thresh else 0
    img = img.point(fn, mode='L')

    # 裁切數字
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # 置中與縮放
    target_size = 28
    digit_len = 20
    w, h = img.size
    ratio = digit_len / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    new_img = Image.new('L', (target_size, target_size), 0)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    return new_img

# --- 3. GUI 應用程式類別 ---
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手寫數字辨識系統 (AI Project)")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        # 載入模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvNet().to(self.device)
        self.load_model_weights()

        # 初始化介面元件
        self.create_widgets()

    def load_model_weights(self):
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}")
            
            # 嘗試載入權重
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
            except TypeError:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            
            self.model.eval()
            print("模型載入成功！")
        except Exception as e:
            messagebox.showerror("錯誤", f"模型載入失敗：\n{str(e)}")

    def create_widgets(self):
        # 標題
        title_label = tk.Label(self.root, text="請選擇圖片進行辨識", font=("Arial", 16))
        title_label.pack(pady=10)

        # 按鈕區
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        
        select_btn = tk.Button(btn_frame, text="開啟圖片", command=self.open_image, font=("Arial", 12), bg="#ddd", width=15)
        select_btn.pack(side=tk.LEFT, padx=10)

        # 圖片顯示區 (使用 Canvas 或 Label)
        img_frame = tk.Frame(self.root)
        img_frame.pack(pady=20)

        # 左邊：原圖
        self.panel_original = tk.Label(img_frame, text="[原始圖片]", width=25, height=12, relief="sunken", bg="#f0f0f0")
        self.panel_original.pack(side=tk.LEFT, padx=20)

        # 右邊：模型看到的圖
        self.panel_processed = tk.Label(img_frame, text="[模型視角]", width=25, height=12, relief="sunken", bg="#f0f0f0")
        self.panel_processed.pack(side=tk.LEFT, padx=20)

        # 結果顯示區
        self.result_label = tk.Label(self.root, text="預測結果：--", font=("Arial", 20, "bold"), fg="blue")
        self.result_label.pack(pady=10)

        self.confidence_label = tk.Label(self.root, text="信心水準：--%", font=("Arial", 12))
        self.confidence_label.pack()

        # 狀態列
        self.status_label = tk.Label(self.root, text=f"執行裝置: {self.device}", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not file_path:
            return

        try:
            # 1. 顯示原始圖片 (縮放到適合顯示的大小)
            original_img = Image.open(file_path)
            display_img = original_img.copy()
            display_img.thumbnail((200, 200)) # 縮圖供介面顯示
            photo_original = ImageTk.PhotoImage(display_img)
            
            self.panel_original.config(image=photo_original, width=200, height=200)
            self.panel_original.image = photo_original # 保持引用避免被回收

            # 2. 執行辨識
            self.predict_logic(file_path)

        except Exception as e:
            messagebox.showerror("錯誤", f"無法開啟圖片：\n{str(e)}")

    def predict_logic(self, image_path):
        # 準備雙重驗證 (原始處理 vs 加粗處理)
        pil_image = Image.open(image_path)
        img_normal = preprocess_standard(pil_image)
        img_thick = img_normal.filter(ImageFilter.MaxFilter(3)) # 加粗

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        candidates = [
            {"name": "一般", "img": img_normal},
            {"name": "加粗", "img": img_thick}
        ]
        
        best_confidence = -1
        best_result = None

        # 進行推論
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

        # 3. 更新介面顯示
        # 顯示最終採用的「模型視角圖」
        display_processed = best_result["img"].resize((200, 200), Image.Resampling.NEAREST)
        photo_processed = ImageTk.PhotoImage(display_processed)
        self.panel_processed.config(image=photo_processed, width=200, height=200)
        self.panel_processed.image = photo_processed

        # 更新文字結果
        self.result_label.config(text=f"預測結果：{best_result['pred']}")
        
        # 根據信心水準變更顏色
        color = "green" if best_result['conf'] > 80 else "red"
        self.confidence_label.config(
            text=f"信心水準：{best_result['conf']:.2f}% (採用{best_result['type']}模式)", 
            fg=color
        )

# --- 啟動程式 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()