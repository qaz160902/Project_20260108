# 個人日程管理系統開發計畫 (Vue.js + Flask + SQLite)

這是一份為個人使用量身打造的日程與工作管理系統開發藍圖。目標是解決「事情太多容易忘記」的問題，並提供清晰的時間視圖（今天、本週、下週）與日曆功能。

## 1. 專案目標 (Project Goals)

*   **核心目的：** 個人行程記錄與工作提醒。
*   **解決痛點：** 防止遺忘會議時間或工作內容，解決事務繁雜導致的混亂。
*   **主要視圖：**
    *   **看板儀表板 (Dashboard)：** 自動分類顯示「今天」、「本週」、「下週」的待辦事項。
    *   **日曆視圖 (Calendar View)：** 以月曆形式查看行程分布。

## 2. 技術堆疊 (Tech Stack)

*   **前端 (Frontend):**
    *   Vue.js 3 (Composition API)
    *   Vite
    *   Axios
    *   **UI 框架:** Element Plus 或 Tailwind CSS (用於快速建構美觀的看板佈局)
    *   **日曆套件:** FullCalendar (Vue adapter) 或 V-Calendar
*   **後端 (Backend):**
    *   Python 3.x
    *   Flask
    *   Flask-SQLAlchemy
    *   Flask-CORS
*   **資料庫 (Database):**
    *   SQLite (單機檔案儲存，方便備份與遷移)

## 3. 專案目錄結構 (Project Structure)

```
schedule-app/
├── backend/
│   ├── app.py
│   ├── models.py
│   ├── routes.py           # 建議將路由獨立出來
│   ├── instance/
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── TaskDashboard.vue  # 看板視圖 (今日/本週/下週)
    │   │   ├── CalendarView.vue   # 日曆視圖
    │   │   └── TaskForm.vue       # 新增/編輯任務表單
    │   ├── views/
    │   ├── App.vue
    │   └── services/
    └── package.json
```

## 4. 資料庫設計 (Database Schema)

更新資料表設計以支援日程安排與詳細描述。

**Table: `Task`**

| 欄位名稱      | 類型        | 描述 |
| ------------- | ----------- | ---- |
| `id`          | Integer     | 主鍵 |
| `title`       | String(100) | 標題 (例如：跟 A 客戶開會) |
| `description` | Text        | 詳細內容 (例如：準備報價單、會議重點) |
| `due_date`    | DateTime    | **截止/執行時間** (核心欄位，用於排序與顯示) |
| `completed`   | Boolean     | 是否完成 |
| `created_at`  | DateTime    | 建立時間 |

## 5. API 介面設計 (RESTful API)

| 方法   | 路徑 | 功能 | 參數/備註 |
| ------ | ---- | ---- | --------- |
| GET    | `/api/tasks` | 獲取任務列表 | 支援 Query Params: `start_date`, `end_date` (用於日曆與看板篩選) |
| POST   | `/api/tasks` | 新增任務 | 需包含 `due_date` |
| PUT    | `/api/tasks/<id>` | 更新任務 | 修改時間、內容或狀態 |
| DELETE | `/api/tasks/<id>` | 刪除任務 | |

## 6. 功能模組詳細規劃

### A. 儀表板 (Dashboard)
這是應用程式的首頁。系統會自動根據 `due_date` 將未完成的任務分類到以下三個區塊：
1.  **今天 (Today):** `due_date` 為當天的所有任務。
2.  **本週 (This Week):** `due_date` 在本週剩餘時間的任務。
3.  **下週 (Next Week):** `due_date` 在下週一至下週日的任務。
*   *過期任務提醒：* 醒目顯示 `due_date` 小於今天且未完成的任務。

### B. 日曆 (Calendar)
*   顯示整個月的視圖。
*   點擊日期可以查看當天的詳細行程。
*   (進階) 支援拖拉任務來改變日期。

### C. 任務管理
*   新增時必須選擇日期與時間（針對會議）。
*   支援簡單的標記（如：重要、待確認）。

## 7. 開發步驟修訂

### 階段一：後端核心 (Backend Core)
1.  建立 Flask 專案。
2.  實作包含 `due_date` 和 `description` 的資料庫模型。
3.  開發支援日期範圍篩選的 API (`GET /api/tasks?start=...&end=...`)。

### 階段二：前端基礎與 API 串接
1.  建立 Vue 3 專案。
2.  設定路由 (Router) 以切換「看板」與「日曆」模式。
3.  實作 API 服務層。

### 階段三：看板視圖開發 (Dashboard)
1.  開發三個列表元件：Today, This Week, Next Week。
2.  實作前端邏輯：從後端撈取資料後，根據日期分配到對應列表。

### 階段四：日曆與優化
1.  整合日曆套件。
2.  優化 UI 樣式 (卡片式設計，區分緊急程度)。