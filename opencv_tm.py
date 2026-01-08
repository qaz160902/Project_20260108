import os
# 強制使用 Legacy Keras (Keras 2) 以相容 Teachable Machine 模型
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# 關閉 OneDNN 最佳化訊息，減少干擾
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    from tensorflow.keras.models import load_model
except ImportError:
    raise ImportError("無法匯入 keras。請確認已安裝 'tf_keras' 套件：pip install tf-keras")

import cv2  # 請安裝 opencv-python
import numpy as np

# 禁用科學記號以提高清晰度
np.set_printoptions(suppress=True)

# 載入模型
model = load_model("model/keras_model.h5", compile=False)

# 載入標籤
class_names = open("model/labels.txt", "r", encoding="utf-8").readlines()

# CAMERA 可以是 0 或 1，取決於您電腦的預設攝影機
camera = cv2.VideoCapture(0)

while True:
    # 擷取網路攝影機的影像
    ret, image = camera.read()

    # 將原始影像調整大小為 (224-高, 224-寬) 像素
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # 在視窗中顯示影像
    cv2.imshow("Webcam Image", image)

    # 將影像轉換為 numpy 陣列並重塑為模型的輸入形狀
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # 正規化影像陣列
    image = (image / 127.5) - 1

    # 進行模型預測
    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # 列印預測結果和信心分數
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # 監聽鍵盤輸入
    keyboard_input = cv2.waitKey(1)

    # 27 是鍵盤上 Esc 鍵的 ASCII 碼
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
