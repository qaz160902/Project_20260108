"""
Google 日曆 AI Agent
使用 LangGraph + Gemini 來查詢和新增行程
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 設定 Google API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

from langchain_google_community import CalendarToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 取得腳本所在目錄並切換過去 (CalendarToolkit 需要)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
CREDENTIALS_PATH = os.path.join(SCRIPT_DIR, "credentials.json")
TOKEN_PATH = os.path.join(SCRIPT_DIR, "token.json")


def get_current_time_str():
    """取得當前時間的格式化字串"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]
    return now.strftime(f"%Y年%m月%d日 {weekday} %H:%M:%S")


def create_calendar_agent():
    """建立日曆 Agent"""

    # 1. 初始化日曆工具
    toolkit = CalendarToolkit(
        credentials_path=CREDENTIALS_PATH,
        token_path=TOKEN_PATH,
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    tools = toolkit.get_tools()

    print(f"已載入 {len(tools)} 個日曆工具:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")

    # 2. 設定 Gemini 模型
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # 使用 1.5 版本，配額較寬鬆
        temperature=0.1
    )

    # 3. 建立系統提示詞
    current_time = get_current_time_str()
    system_prompt = f"""你是一個專業的日曆助理 Agent。

當前時間：{current_time}
時區：Asia/Taipei (台北時間 UTC+8)

規則：
1. 請使用這個時間作為參考來理解使用者說的「今天」、「明天」、「下週」等相對時間
2. 所有行程預設時區為 Asia/Taipei
3. 如果使用者沒有指定活動時長，預設為 1 小時
4. 請用繁體中文回答使用者的問題
5. 新增行程時，請確認行程的標題、開始時間、結束時間"""

    # 4. 使用 LangGraph 建立 Agent
    agent = create_react_agent(llm, tools)

    return agent, system_prompt


def main():
    print("=" * 60)
    print("Google 日曆 AI Agent")
    print("=" * 60)
    print(f"當前時間：{get_current_time_str()}")
    print("對話記憶：保留最近 5 輪對話")
    print("-" * 60)
    print("你可以問我關於行程的問題，例如：")
    print("  - 我明天有什麼行程？")
    print("  - 幫我在下週一下午 3 點新增一個會議")
    print("  - 這週有哪些活動？")
    print("輸入 'quit' 或 'exit' 離開")
    print("=" * 60)

    # 建立 Agent
    agent, system_prompt = create_calendar_agent()

    # 對話歷史 (保留 5 輪)
    chat_history = [SystemMessage(content=system_prompt)]
    MAX_HISTORY = 5  # 最多保留 5 輪對話

    while True:
        try:
            user_input = input("\n你: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("再見！")
                break

            # 加入使用者訊息
            chat_history.append(HumanMessage(content=user_input))

            # 執行 Agent
            result = agent.invoke({"messages": chat_history})

            # 取得回應
            response_messages = result["messages"]
            ai_response = response_messages[-1].content

            print(f"\nAgent: {ai_response}")

            # 加入 AI 回應到歷史
            chat_history.append(AIMessage(content=ai_response))

            # 保持最多 5 輪對話 (1 system + 10 human/ai)
            if len(chat_history) > 1 + MAX_HISTORY * 2:
                # 保留 system message + 最近的對話
                chat_history = [chat_history[0]] + chat_history[-(MAX_HISTORY * 2):]

        except KeyboardInterrupt:
            print("\n再見！")
            break
        except Exception as e:
            print(f"\n錯誤: {e}")


if __name__ == "__main__":
    main()
