import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os

# --- 設定路徑 ---
MODEL_PATH = r"D:\AWORKSPACE\Github\Project_20260108\AIProject_KE\mnist_cnn.pth"
IMAGE_PATH = r"D:\AWORKSPACE\Github\Project_20260108\AIProject_KE\MINST\0.jpg"

# --- 模型架構 ---
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

# --- 影像處理工具 ---
def get_centered_image(img):
    """ 負責基礎的二值化、裁切與置中 """
    # 1. 轉灰階
    img = img.convert('L')
    
    # 2. 自動反轉 (確保是黑底白字)
    if np.mean(img) > 127: 
        img = ImageOps.invert(img)
    
    # 3. 二值化 (拉高對比)
    thresh = 50 
    fn = lambda x : 255 if x > thresh else 0
    img = img.point(fn, mode='L')

    # 4. 裁切
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # 5. 置中與縮放
    target_size = 28
    digit_len = 20 # 數字本體大小
    
    w, h = img.size
    ratio = digit_len / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    new_img = Image.new('L', (target_size, target_size), 0)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    return new_img

def predict_batch(model, device, transform, images, labels):
    print(f"\n{'-'*10} 開始粗細測試 {'-'*10}")
    
    for i, img in enumerate(images):
        # 存檔供檢查
        filename = f"debug_thick_{i}_{labels[i]}.png"
        img.save(filename)
        
        # 預測
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            conf = probs[0][pred].item() * 100
            
        print(f"版本 [{labels[i]}]:")
        print(f"  -> 預測數字: {pred}")
        print(f"  -> 信心水準: {conf:.2f}%")
        if conf < 50:
            print("     (信心過低，模型看不懂)")
        print(f"  -> 影像已存為: {filename}")
        print("-" * 30)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 載入模型
    model = ConvNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. 準備原始圖片
    original_img = Image.open(IMAGE_PATH)
    base_img = get_centered_image(original_img) # 這是標準處理後的圖

    # 3. 製作不同粗細版本
    # 版本 A: 原始 (可能太細)
    img_normal = base_img
    
    # 版本 B: 加粗 (使用 MaxFilter 模擬膨脹效果)
    # 對於黑底白字，MaxFilter 會讓白色區域變大
    img_thick = base_img.filter(ImageFilter.MaxFilter(3)) 
    
    # 版本 C: 特粗 (再加粗一次)
    img_thicker = img_thick.filter(ImageFilter.MaxFilter(3))

    # 4. 準備進行預測
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    images = [img_normal, img_thick, img_thicker]
    labels = ["原始粗細", "加粗(Level 1)", "特粗(Level 2)"]
    
    predict_batch(model, device, transform, images, labels)

if __name__ == "__main__":
    main()