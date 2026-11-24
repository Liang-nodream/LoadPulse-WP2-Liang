# LoadPulse · WP2｜電力負載預測與時間模型失效分析

> **目前狀態：研究開發／初步 Pilot。** 本 Repository 存放 LoadPulse 建模工作、每週交付文件及文獻整理。既有 Ridge／GAM 實驗僅屬工程與探索性測試，**尚未證實 Robust Temporal Failure**。

## 專案目標

以時間與日曆資訊建立逐時電力需求的基準預測，比較不同模型在相同目標時段的誤差，進而研究跨模型共同失效候選及可能的成因。

## 資料夾導覽

| 位置 | 用途 |
|---|---|
| [`baseline_2/`](baseline_2/) | 目前的 Python 模型與實驗主程式（持續開發） |
| [`deliverables/`](deliverables/) | 每週研究交付；第一週為 [`LoadPulse_W1_v025/`](deliverables/LoadPulse_W1_v025/)（若已改名為 `W1`，請同步修改連結） |
| `paper/` | 文獻與閱讀資料；該資料夾需 Commit／Push 後才會出現在 GitHub |

> AEMO 原始資料與執行產生的大量輸出保留在本機，不隨 Repository 上傳；相關目錄由 `.gitignore` 排除。正式模型資料將與資料組的 `data/processed/` 規格對接。

## 目前已有的模型（初步 Pilot）

| 模型 | 方法 | 目前使用資訊 |
|---|---|---|
| Model A | Ridge Regression | 小時、星期、假日與年度週期 |
| Model B | GAM（廣義加性模型） | 同類日曆資訊，以平滑函數描述非線性週期 |

目前兩模型皆為 **Calendar-only**；未加入歷史負載或氣象。此版本的 GAM 並非所參考論文的完整 GAM + ETS + AR 系統。

### 主要程式

| 檔案（位於 `baseline_2/`） | 功能 |
|---|---|
| `main.py` | 控制資料來源、訓練、預測、指標計算與跨模型比較 |
| `model_a.py` | Ridge 訓練與預測 |
| `gam_model.py` | GAM 訓練與預測 |
| `feature_engineering.py` | 日曆特徵建構 |
| `failure_detector.py` | 使用校準門檻標記大誤差候選時段 |
| `aemo_loader.py` | 讀取 AEMO 原始半小時 CSV，轉為逐時平均需求（MW） |
| `data_generator.py` | 產生合成資料，供流程測試 |

## 快速執行

在 Repository 根目錄開啟終端機：

```powershell
cd baseline_2
python -m pip install numpy pandas scikit-learn pygam matplotlib holidays
python main.py --source synthetic
```

要執行澳洲 QLD1 Pilot，請先將 `AEMO_NEM_QLD1_RAW_2021H1.csv` 放在本機的 `baseline_2/Australia_AEMO/`：

```powershell
python main.py --source aemo --region QLD1 --models both
```

其他參數：`python main.py --help`。輸出預設存放於 `baseline_2/outputs/`（已設定為 Git 忽略）。

> **資料版本注意：**上述 `aemo_loader.py` 使用 AEMO 原始半小時 CSV；尚未切換至資料組清洗後的逐時寬表／Full QC CSV，兩個版本的實驗數據不可混用。

## QLD1 初步實驗紀錄

資料期間：2021 年 1～6 月，1～4 月訓練、5 月校準、6 月 Pilot（720 小時）。

| 指標 | Ridge | GAM |
|---|---:|---:|
| RMSE | 1,371.91 MW | 566.35 MW |
| MAE | 1,263.82 MW | 466.61 MW |
| 超過失效門檻的候選時段 | 191 | 46 |

**兩模型共同超過各自門檻：46 小時。** 候選門檻為 `絕對誤差 > 2.5 × 校準期 RMSE`，尚未完成共同失效的穩健性與統計顯著性檢驗。

## 第一週交付與新版研究規劃

第一週依 **LoadPulse v0.25** 規格完成三份設計文件：

1. `model_ladder_spec.md`：M0 → M3+ 資訊階梯與公平模型比較原則。
2. `forecast_protocol.md`：預測發布時點、時距、資料切分、rolling-origin 與資訊可得性。
3. `prediction_schema.md`：統一預測欄位、配對鍵與品質旗標。

下一步包含 M0、seasonal-naive 對照、樣本外預測與資料品質旗標的整合。**既有 Ridge／GAM Pilot 不代表新版 M0～M3+ 階梯已完成。**

