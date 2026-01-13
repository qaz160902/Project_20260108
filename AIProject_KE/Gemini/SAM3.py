from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# 建立 Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("=== 正在讀取 Google 伺服器上的模型清單 ===")

try:
    # 直接列出所有模型，不做過濾，避免屬性錯誤
    for model in client.models.list():
        # 新版 SDK 的模型物件通常有 .name 屬性
        print(f"模型 ID: {model.name}")
        
except Exception as e:
    print(f"發生錯誤: {e}")