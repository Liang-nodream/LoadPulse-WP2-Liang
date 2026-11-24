# LoadPulse · WP2｜電力負載預測與時間模型失效分析

> **狀態：初步 Pilot（開發中）**。本專案負責 LoadPulse 的 WP2 建模工作；目前結果僅供工程測試與探索，不代表已證實 Robust Temporal Failure。

## 研究目標

利用僅含**時間／日曆資訊**的模型預測逐時電力需求，比較不同模型在何時出現明顯誤差，再進一步研究共同失效的時間特徵，以及後續可能的氣象關聯。

## 目前模型

| 模型 | 方法 | 使用資訊 |
|---|---|---|
| Model A | Ridge Regression | 小時、星期、假日、年度週期 |
| Model B | GAM（廣義加性模型） | 相同日曆資訊，以平滑函數描述非線性週期 |

目前都**不使用實際溫度或歷史負載**。GAM 是單階段 Calendar-only 版本，並非完整的 GAM + ETS + AR 論文模型。

## 程式檔案

| 檔案 | 用途 |
|---|---|
| `main.py` | 切換資料來源、執行模型、比較並輸出結果 |
| `model_a.py` / `gam_model.py` | Ridge / GAM 訓練與預測 |
| `feature_engineering.py` | 日曆特徵處理 |
| `failure_detector.py` | 以校準門檻偵測大誤差候選 |
| `aemo_loader.py` | AEMO 原始半小時 CSV 轉逐時資料 |
| `data_generator.py` | 合成資料測試 |
| `requirements.txt` | Python 相依套件 |

## 快速執行

安裝套件：

```bash
python -m pip install -r requirements.txt
```

先測合成資料（不需要下載 CSV）：

```bash
python main.py --source synthetic
```

測試澳洲昆士蘭 QLD1：先自行將 `AEMO_NEM_QLD1_RAW_2021H1.csv` 放在本機 `Australia_AEMO/`，再執行：

```bash
python main.py --source aemo --region QLD1 --models both
```

更多參數：`python main.py --help`。預測結果產生在 `outputs/aemo/QLD1/` 或 `outputs/synthetic/`。

> 資料注意：目前的 `aemo_loader.py` 讀取的是 **raw 半小時 CSV**，尚未改成直接讀取資料組的 `data/processed/` 標準化檔案。這兩類資料的 Pilot 結果不可直接混用。

## QLD1 初步實驗結果

資料：AEMO QLD1 **2021 年上半年原始資料**，模型端轉成逐時需求；1–4 月訓練、5 月校準、6 月測試（720 小時）。訓練資料另受預測起點可取得資訊限制。

| 指標 | Ridge | GAM |
|---|---:|---:|
| RMSE | 1,371.91 MW | 566.35 MW |
| MAE | 1,263.82 MW | 466.61 MW |
| 超過門檻的時段 | 191 | 46 |

兩模型**共同超過各自校準門檻：46 小時**。目前門檻為「絕對誤差 > 2.5 × 校準 RMSE」；這些只是**共同失效候選**，尚未進行正式穩健性／統計顯著性驗證。

## 後續工作

依 `C_modeling_v025.md` 的新版研究規劃，逐步建立 **M0 → M1 → M2 → M3 → M3+** 的資訊階梯，完成 forecast protocol、統一 prediction schema、rolling-origin、seasonal-naive 對照，並與資料組對齊 processed 資料與品質旗標。**目前 Model A／B 是舊版兩個 Pilot 模型，不等於新版資訊階梯已完成。**

## 參考文獻

Zimmermann & Ziel (2025), *Efficient Mid-term Forecasting of Hourly Electricity Load Using Generalized Additive Models*。
