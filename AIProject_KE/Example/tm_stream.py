import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

def main():
    # 1. 設定路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "keras_model.h5")
    labels_path = os.path.join(project_root, "labels.txt")

    # 2. 載入模型與標籤
    print(f"正在載入模型: {model_path} ...")
    model = load_model(model_path, compile=False)
    
    print(f"正在載入標籤: {labels_path} ...")
    with open(labels_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    # 3. 開啟攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟攝像頭")
        return

    print("開始影像辨識，按 'q' 鍵退出...")

    # 預先分配記憶體給模型輸入
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法接收影像")
            break

        # --- 影像預處理 (為了符合 Teachable Machine 的訓練格式) ---
        
        # 1. OpenCV 讀取的是 BGR，需轉為 RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. 轉為 PIL Image 物件 (方便使用 ImageOps)
        image = Image.fromarray(image_rgb)
        
        # 3. 調整大小並裁切 (保持長寬比，裁切中間)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # 4. 轉回 Numpy array 並標準化 (-1 到 1 之間)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # 5. 填入輸入資料
        data[0] = normalized_image_array

        # --- 預測 ---
        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # --- 顯示結果 ---
        # 處理文字顯示 (標籤內容可能包含編號，如 "0 ClassName")
        display_text = f"{class_name}: {confidence_score:.2%}"
        
        # 在畫面上印出文字 (顏色: 綠色, 字體大小: 1)
        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Teachable Machine Real-time', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
