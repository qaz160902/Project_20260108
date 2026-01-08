# Todo List App 專案開發計畫書

## 1. 專案動機與目的 (Motivation & Purpose)
本專案旨在為個人使用者打造一款結合 **待辦事項 (Todo)** 與 **日曆行程 (Calendar)** 的管理工具。
- **自用需求:** 提供一個專屬的數位空間，用於整合零散的日常行程。
- **行程管理:** 視覺化呈現何時上課、開會或處理公務，避免行程衝突。
- **工作記錄:** 詳細記錄工作內容與備忘錄，提升工作效率與回顧便利性。

---

## 2. 專案概述 (Project Overview)
- **前端 (Frontend):** Vue.js 3 (使用 Vite 建置)，整合日曆套件 (如 FullCalendar 或 V-Calendar)。
- **後端 (Backend):** Python Flask
- **資料庫 (Database):** SQLite (透過 SQLAlchemy ORM 操作)
- **通訊協定:** HTTP RESTful API

---

## 3. 專案目錄結構 (Project Structure)
```text
todo-project/
├── backend/                # 後端 Flask 專案
│   ├── app.py              # API 路由與邏輯
│   ├── models.py           # 資料庫模型定義
│   ├── database.db         # SQLite 資料庫
│   └── requirements.txt    # 依賴套件
│
└── frontend/               # 前端 Vue 專案
    ├── src/
    │   ├── components/
    │   │   ├── CalendarView.vue  # 日曆視圖組件
    │   │   ├── TodoList.vue      # 清單視圖組件
    │   │   └── EventModal.vue    # 新增/編輯行程的彈窗
    │   ├── App.vue
    │   └── main.js
    └── ...
```

---

## 4. 資料庫設計 (Database Schema)
更新資料表 `todos` 以支援行程管理。

**Table Name:** `todos`

| 欄位名稱 (Field) | 資料型態 (Type) | 說明 (Description) |
| :--- | :--- | :--- |
| `id` | Integer (PK) | 唯一識別碼 |
| `title` | String(100) | 標題 (例如: "行銷會議", "Python 課程") |
| `description` | Text | 詳細工作內容或備註 |
| `category` | String(50) | 分類 (例如: "工作", "上課", "開會", "私人") |
| `start_time` | DateTime | 開始時間 (若為純待辦事項可為空) |
| `end_time` | DateTime | 結束時間 |
| `is_all_day` | Boolean | 是否為全天行程 |
| `completed` | Boolean | 完成狀態 |

---

## 5. 後端 API 規格 (Backend API Specification)
新增支援日期區間篩選的 API。

| HTTP Method | Endpoint | 功能 | 參數範例 |
| :--- | :--- | :--- | :--- |
| **GET** | `/api/todos` | 取得所有事項 (支援日期篩選) | `?start=2023-10-01&end=2023-10-31` |
| **POST** | `/api/todos` | 新增行程/待辦 | `{"title": "開會", "start_time": "2023-10-20 14:00"}` |
| **PUT** | `/api/todos/<id>` | 更新內容 | `{"completed": true}` |
| **DELETE** | `/api/todos/<id>` | 刪除 | 無 |

---

## 6. 前端規劃 (Frontend Architecture)

### 6.1 主要功能模組
1.  **日曆視圖 (Calendar View):**
    - 以月/週/日為單位顯示行程。
    - 點擊日期可新增行程。
    - 拖過時間區段可建立特定時段的會議。
2.  **清單視圖 (List View):**
    - 傳統條列式顯示，適合快速勾選完成項目。
3.  **分類標籤:**
    - 使用不同顏色區分 "工作", "上課" 等類別。

### 6.2 技術選型
- **核心:** Vue 3 + Vite
- **日曆套件:** FullCalendar (功能強大) 或 V-Calendar (輕量美觀)
- **狀態管理:** Pinia (管理行程資料流)

---

## 7. 開發步驟 (Development Roadmap)

### Phase 1: 後端核心 (Backend Core)
1.  定義新的資料庫模型 (加入時間與分類欄位)。
2.  實作支援日期範圍查詢的 API。

### Phase 2: 前端日曆整合 (Frontend Calendar)
1.  建立 Vue 專案。
2.  整合 FullCalendar 套件。
3.  實作從後端撈取資料並渲染在日曆上。

### Phase 3: 互動與優化 (Interaction)
1.  實作點擊日曆新增行程的彈窗 (Modal)。
2.  拖拉移動行程功能 (Drag & Drop)。

