import cv2

def main():
    # 打開預設攝像頭 (通常是 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("無法開啟攝像頭")
        return

    print("正在開啟攝影機串流，按 'q' 鍵退出...")

    while True:
        # 逐幀捕獲
        ret, frame = cap.read()

        if not ret:
            print("無法接收串流影像 (stream end?)。正在退出...")
            break

        # 顯示影像
        cv2.imshow('Camera Stream', frame)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
