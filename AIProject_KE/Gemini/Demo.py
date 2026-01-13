from google import genai
from dotenv import load_dotenv

load_dotenv() # 這會讀取 .env 檔案並設定為環境變數
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client( )

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="請用繁體中文跟我自我介紹"
)
print(response.text)