import cv2

def main():
    # 顯示 OpenCV 版本以確認安裝成功
    print(f"OpenCV Version: {cv2.__version__}")

    # 初始化攝影機，參數 0 通常代表系統預設的第一個攝影機
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("錯誤：無法開啟攝影機")
        return

    print("攝影機已開啟，按 'q' 鍵退出...")

    while True:
        # 逐幀讀取影像
        ret, frame = cap.read()

        # 如果讀取失敗（例如攝影機斷線），則退出迴圈
        if not ret:
            print("錯誤：無法接收影像幀 (Stream end?)")
            break

        # 顯示影像視窗
        cv2.imshow('Camera Stream', frame)

        # 等待按鍵輸入，若按下 'q' 則退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放攝影機資源並關閉視窗
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()