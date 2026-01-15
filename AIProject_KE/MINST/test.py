import mediapipe as mp

try:
    print(f"MediaPipe 版本: {mp.__version__}")
    # 嘗試存取 solutions
    mp_hands = mp.solutions.hands
    print("✅ 成功！solutions 屬性現在可以正常使用了。")
except AttributeError:
    print("❌ 失敗... 仍然找不到 solutions。")
except Exception as e:
    print(f"發生其他錯誤: {e}")