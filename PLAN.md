# AutoTabular — Implementation Plan

---

## Phase 1 ✅ COMPLETED — Auth, File Upload, Preprocessing

### 完成內容

**認證系統**
- 使用者註冊 (`POST /api/auth/register`)、登入 (`POST /api/auth/login`)、查詢自身資訊 (`GET /api/auth/me`)
- JWT token 驗證，密碼以 bcrypt 雜湊儲存（passlib + bcrypt==4.0.1）
- `verify_token` dependency 將 JWT sub 轉型為 int，解決與 ORM user_id 型別不符的問題

**檔案上傳**
- `POST /api/files/upload` — 儲存 CSV 至 `uploads/` 目錄並在 DB 寫入 metadata
- `File` ORM model（UUID 主鍵、user_id FK、filepath、upload_time）
- `verify_file_ownership()` helper — 403 / 404 防護

**資料預處理**
- `GET /api/data/summary/{file_id}` — 回傳 shape、column dtype、缺失值、重複列等統計
- `POST /api/data/preprocess/{file_id}` — 支援 handle_missing、handle_outliers、encode_categorical、scale_features、remove_duplicates
- 其餘細項端點：`/missing-values`、`/outliers`、`/encode`、`/scale`、`/duplicates`、`/save`、`/reset`

**測試基礎建設**
- `tests/conftest.py`：使用 SQLite in-memory + StaticPool，完全隔離，不產生 `test.db`
- Fixtures：`db_setup`、`client`、`registered_user`、`auth_token`
- 測試通過：20 個

**關鍵 Bug 修正**
- bcrypt 5.x 與 passlib 不相容 → 降版至 bcrypt==4.0.1
- JWT sub 字串 vs int 比較錯誤 → `int(payload.get("sub"))` 強制轉型
- `load_csv` 的 `df.empty` 檢查誤被 `except Exception` 捕捉 → 移至 try 區塊外
- `load_csv` 重構後缺少 `return df` → 補上

---

## Phase 2 ✅ COMPLETED — ML Training Pipeline

### 完成內容

**TrainingJob DB Model**
- 資料表欄位：`id`（UUID）、`file_id`、`user_id`、`task_type`、`target_column`、`algorithm`、`hyperparameters`（JSON string）、`test_size`、`random_state`、`metrics`（JSON string）、`model_filepath`、`status`、`error_message`、`training_duration_seconds`、`created_at`、`completed_at`

**ModelTrainer 服務**
- 支援演算法：LightGBM、XGBoost、Random Forest（分類與回歸皆支援）
- Pipeline：載入 CSV → 驗證資料 → 切分 train/test → 訓練 → 評估 → 儲存 artifact
- 分類指標：accuracy、precision、recall、f1_score、confusion_matrix
- 回歸指標：mse、rmse、mae、r2_score
- Model artifact 以 joblib 儲存為 `{MODEL_DIR}/{job_id}.joblib`，格式：`{"model": <fitted_model>, "feature_columns": [...]}`（含 feature_columns 以利推論）
- 靜態方法 `load_model_artifact()` 含 legacy 格式回退機制

**API 端點**
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/training/train` | 啟動訓練任務 |
| GET | `/api/training/jobs` | 列出使用者的訓練任務（分頁） |
| GET | `/api/training/jobs/{job_id}` | 取得任務詳情與指標 |
| DELETE | `/api/training/jobs/{job_id}` | 刪除任務及 model artifact |

**測試通過：19 個**（含 3 種演算法 × 2 種任務、自訂超參數、非數值特徵、缺少目標欄位、檔案權限等）

**關鍵 Bug 修正**
- numpy 型別無法 JSON 序列化 → 評估結果全部轉為 Python 原生型別
- 訓練樣本過少時 lightgbm 報錯 → 最少列數驗證（30 列）
- `has_categorical.csv` fixture 修正（30 列）確保非數值特徵測試在樣本數檢查前觸發

---

## Phase 3 ✅ COMPLETED — Model Inference

### 完成內容

**ModelPredictor 服務**
- 載入訓練任務 → 驗證任務所有權與狀態（需為 `completed`）
- 從 artifact 讀取 `feature_columns`；若為舊格式則回退至原始訓練 CSV 推導
- 輸入 CSV 驗證：
  - 缺少必要欄位 → 400（列出缺失欄位）
  - 多餘欄位 → 自動忽略
  - 非數值欄位 → 400
  - 含 NaN → 400
- 分類結果轉為 `int`，回歸結果轉為 `float`
- 回傳格式：包含 predictions 陣列與每列附加 `prediction` 欄的 `data_with_predictions`

**API 端點**
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/inference/predict` | 對指定 file 執行推論，回傳預測結果 |

**測試通過：15 個**（3 種演算法 × 2 種任務 happy path、欄位不符、多餘欄位、非數值輸入、空 CSV、找不到任務/檔案、未認證、跨使用者 job/file 權限）

**關鍵 Bug 修正**
- 所有 numpy 數值轉為 Python 原生型別，避免 JSON 序列化失敗

---

## Phase 4 ✅ COMPLETED — React Frontend

### 完成內容

**後端新增端點**
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/files/` | 列出目前使用者上傳的所有檔案 |

**前端架構**
- 技術棧：React + Vite + TypeScript + Tailwind CSS v3
- 路徑：`frontend/`
- HTTP client：Axios（自動附加 Bearer token）
- 路由：React Router v6
- 認證：Context API + localStorage

**頁面與流程（4A → 4G）**

| 頁面 | Route | 說明 |
|------|-------|------|
| 4A Login/Register | `/` | 登入 / 註冊切換，錯誤提示 |
| 4B Upload | `/upload` | Drag & Drop CSV 上傳 |
| 4C Data Preview | `/preview/:fileId` | 資料統計（shape、dtype、missing、min/max/mean）|
| 4D Preprocess | `/preprocess/:fileId` | 勾選丟棄欄位、缺失值策略、正規化 |
| 4E Train | `/train/:fileId` | 選擇目標欄位、任務類型、演算法 |
| 4F Metrics | `/metrics/:jobId` | 指標卡片（Accuracy/F1/RMSE/R²）+ Confusion Matrix |
| 4G Inference | `/inference/:jobId` | 上傳推論 CSV、顯示預測結果、下載 CSV |
| Dashboard | `/dashboard` | 訓練歷史列表，含 Metrics / Infer 快捷連結 |

**主要檔案**
```
frontend/src/
├── api/client.ts              # Axios instance（自動附加 token）
├── context/AuthContext.tsx    # 全域認證狀態
├── components/
│   └── ProtectedRoute.tsx     # 未登入自動導向 /
└── pages/
    ├── LoginPage.tsx
    ├── UploadPage.tsx
    ├── DataPreviewPage.tsx
    ├── PreprocessPage.tsx
    ├── TrainPage.tsx
    ├── MetricsPage.tsx
    ├── DashboardPage.tsx
    └── InferencePage.tsx
```

**啟動方式**
```bash
# 後端
uvicorn app.main:app --reload

# 前端
cd frontend && npm run dev
```

---

## 整體測試覆蓋率

| Phase | 測試數 | 覆蓋率 |
|-------|--------|--------|
| Phase 1 | 20 | — |
| Phase 2 | +19 | — |
| Phase 3 | +15 | — |
| **Total** | **56** | **77.69%**（門檻 70%）|
