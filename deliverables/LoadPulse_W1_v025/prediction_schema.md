# LoadPulse v0.25｜預測輸出與配對損失規格

- 交付：C 組 W1（11/19–11/22）；修訂日期：2026-11-24（Revision 1：COVID-19 PHEIC 排除）
- 狀態：候選共同介面，待資料組與統計組確認；範例為 mock，非實驗結果。
- 依據：`C_modeling_v025.md`、`C_modeling.md`、現有 `main.py`。

## 1. 檔案與欄位

下週預計輸出 `oos_predictions_pilot.csv`。採 UTF-8、逗號分隔、首列欄位名、long format（每列是一個模型對一個 origin/target 的預測）。時間以 ISO 8601 UTC（Z 或 +00:00；輸出統一 Z）表示，缺失數值為空白，不以 0 代替。

| 欄位 | 型別 / 單位 | 定義 |
|---|---|---|
| timestamp | datetime UTC | 目標區間起始 |
| origin_time | datetime UTC | 預測發布時點 |
| horizon | numeric / hour | timestamp − origin_time 的實際小時數 |
| split | string | development / calibration / confirmation / pilot |
| model_id | string | M0、M1、M2、M3、M3+、M_AR、seasonal_naive 等；形式變體用獨立 ID 如 M0_int |
| model_version | string | 可追溯的模型規格與參數版本，對應 metadata |
| actual | float / MW | 目標實際負載，發布後才填 |
| prediction | float / MW | origin 時凍結的預測 |
| residual | float / MW | actual − prediction；正值表示低估 |
| loss | float / MW | 候選主要損失 abs(residual)；無效列留空 |
| quality_flag | string | 品質狀態；見下表 |
| is_valid | boolean | 是否符合該實驗的評估條件 |
| run_id | string | 實驗批次識別，所有比較模型共用同批次 |
| region_id | string | 資料區域識別（不得混用不同區域） |
| data_version | string | 負載口徑與清洗版本 |
| information_mode | string | operational / retrospective；無天氣模型可按相同實驗模式建立對照 |
| data_source | string | synthetic / aemo / taiwan 或 mock |
| fit_id | string | 指向實際擬合紀錄，重訓時更新 |
| squared_loss | float / MW² | residual²，輔助 RMSE |

前 11 欄為工作單列出的核心欄位；其餘為建議擴充，確保可追溯與配對。資料版本須包含負載定義與時間粒度。`information_mode=operational` 表示依可得資訊設計，不代表 mock 或合成資料已證明可部署。

## 1.1 PHEIC 排除與目標網格

依 `forecast_protocol.md`，正式真實資料實驗排除 2020-01-30 至 2023-05-05（含兩日）；當地日曆時間左閉右開區間為 `[2020-01-30 00:00, 2023-05-06 00:00)`。排除期間不產生正式預測列或損失，亦不納入 n_expected；保留原始時間戳，不重新編造連續序列。

在獨立排除紀錄（建議 `data_exclusion_audit.csv`，非 W1 額外交付）記錄被排除的目標時間、區域、資料版本、規則版本與 `exclusion_reason=pheic_period`；metadata 記錄排除起訖、日曆時區、排除前後筆數及各 split 的有效範圍。`data_version` 應可追溯至此遮罩版本。若保留的目標因 lag 或其他觀測特徵落在排除期而不可用，仍保留該模型的預測列，標記 `unavailable_feature`、`is_valid=false`，損失留空；詳細原因記入 audit log。此情況不可與預先排除的目標混淆。合成及 mock 僅供工程與格式測試，可依測試目的單獨套用遮罩。

### 排除紀錄與 metadata 欄位

排除稽核列可使用 `quality_flag=excluded_covid` 與 `is_valid=false`；僅存在於獨立稽核表，不混入正式預測 CSV，不計算 residual 或 loss。`exclusion_reason` 固定為 `pheic_period`（COVID-19 PHEIC）。

| Metadata 欄位 | 定義 / 值 |
|---|---|
| exclusion_policy_version | 候選識別 `pheic_v1` |
| exclusion_applied | true / false；合成測試須明記 |
| exclusion_start | `2020-01-30`（含） |
| exclusion_end_exclusive | `2023-05-06`（不含） |
| exclusion_timezone | 已確認的區域日曆時區 |
| exclusion_reason | `pheic_period` |
| origin_exclusion_policy | 跨界 origin 的凍結規則 |
| n_before_exclusion | 排除前標準化目標觀測列數 |
| n_excluded_covid | 其中因 PHEIC 遮罩排除的列數，不含其他品質排除 |
| n_after_exclusion | 前述兩數之差，尚未扣除其他品質問題 |

另按區域與 split 記錄實際時間範圍及有效筆數；若另建完整目標網格，應分開記錄觀測列與網格列的數量，不混用分母。

## 2. 品質與缺失處理

| quality_flag | 含義 | is_valid |
|---|---|---|
| ok | 真實資料且通過該模式檢查 | true |
| synthetic | 合成資料工程測試，通過檢查 | true（僅工程評估） |
| mock | 人工示意，非實際預測 | true（僅格式測試，不納入實證） |
| missing_actual | 目標觀測缺失或尚未發布 | false |
| missing_prediction | 預測未能產生 | false |
| unavailable_feature | origin 時資訊不可得 | false |
| invalid_observation | 負載品質不合格 | false |
| timestamp_error | 時間或時距不符合協議 | false |

多種錯誤同時發生時，候選採固定優先順序 timestamp_error → invalid_observation → missing_actual → unavailable_feature → missing_prediction；完整原因另存 audit log。保留原始已知 actual/prediction，不以零補缺；無效列 residual/loss/squared_loss 留空。對每個模型保留共同排除規則後的完整預期目標網格，報告各原因筆數。

## 3. 配對鍵與唯一性

共同配對鍵 K：

```text
(run_id, data_source, region_id, data_version, information_mode,
 split, timestamp, origin_time, horizon)
```

預測列唯一鍵：`K + model_id + model_version`。每次比較先固定每個 model_id 的指定版本；若同一模型版本在同一 K 有多筆，視為錯誤，不取平均掩蓋重複。fit_id 隨預定重訓變動，不作跨模型配對鍵。

- 先以共同排除規則後的 schedule 建立各模型的完整列，再以 K 外連接檢查缺漏；不可先各自 dropna 再比較。
- 驗證兩模型的 actual 一致，且兩列均有效才計算配對損失；不一致視為資料錯誤並列入報告。
- 候選 P3 定義 `delta_loss = loss_reference - loss_candidate`，正值表示候選模型誤差較小；累積比較 M0 vs Mk，增量比較 Mk-1 vs Mk。
- 每組報告 n_expected、n_valid_ref、n_valid_candidate、n_paired 及缺失原因；如需跨全部模型排序，另報所有比較模型共同有效集合，避免不同樣本造成不公平。
- MAE 為有效 loss 的平均；RMSE 為有效 squared_loss 平均後開根號。配對比較的指標須在同一有效集合計算。
- 不跨 information_mode、horizon、split 或資料版本混配。P3 統計推論方法由團隊確認，C 組只交付可配對資料。

## 4. Mock CSV 範例

以下假設合成/人工測試使用 Brisbane 時區、前日當地 12:00 發布；所有數字僅示意。第三、四列刻意呈現缺值保留。這不是本週另需交付的實際 CSV。

```csv
timestamp,origin_time,horizon,split,model_id,model_version,actual,prediction,residual,loss,quality_flag,is_valid,run_id,region_id,data_version,information_mode,data_source,fit_id,squared_loss
2023-11-01T14:00:00Z,2023-11-01T02:00:00Z,12,pilot,M0,m0_v001,3100,3000,100,100,mock,true,mock_w1,DEMO,mock_v1,operational,mock,fit_m0,10000
2023-11-01T14:00:00Z,2023-11-01T02:00:00Z,12,pilot,seasonal_naive,sn_v001,3100,3050,50,50,mock,true,mock_w1,DEMO,mock_v1,operational,mock,fit_naive,2500
2023-11-01T15:00:00Z,2023-11-01T02:00:00Z,13,pilot,M0,m0_v001,,3020,,,missing_actual,false,mock_w1,DEMO,mock_v1,operational,mock,fit_m0,
2023-11-01T15:00:00Z,2023-11-01T02:00:00Z,13,pilot,seasonal_naive,sn_v001,,3040,,,missing_actual,false,mock_w1,DEMO,mock_v1,operational,mock,fit_naive,
```

第一個目標以 seasonal_naive 為候選、M0 為參照時，delta_loss=100−50=50 MW；第二個目標不得計算配對損失。整個 mock 不用於科學推論。

## 5. 舊程式欄位映射

| 現有 `predictions_model_a.csv` | 新欄位 / 處理 |
|---|---|
| timestamp_utc | timestamp |
| forecast_origin_utc | origin_time |
| horizon_hours | horizon |
| y_true | actual |
| y_pred | prediction |
| model_id | model_id；舊 model_a_ridge_v1 保留歷史身份，不直接改標 M0 |
| is_valid、quality_flag | 保留並擴充可得性與預測有效性檢查 |
| 無 | 新增 split、model_version、residual、loss、squared_loss、run_id、region_id、data_version、information_mode、data_source、fit_id |

`main.py` 已輸出 `model_a_experiment.json`，可沿用為 metadata 基礎；新增每次擬合 cutoff、版本、套件環境、資料來源版本、模式、更新規則與可得性紀錄。不得將 `failure_detector.py` 的絕對誤差 Error 當作有方向的 residual。

## 6. 驗收檢查

- [ ] 必要欄位齊全、UTC 可解析、origin 不晚於 target、horizon 計算一致。
- [ ] 真實資料預測表無 PHEIC 排除期間的目標；排除紀錄及前後筆數可重現，跨缺口特徵不可用時保留無效列。
- [ ] 唯一鍵無重複，每個模型的預期網格完整，actual 跨模型一致。
- [ ] 預測使用的資訊與訓練標籤在相應時點可得；版本及 fit_id 可追溯。
- [ ] 無效列保留且 loss 為空；所有有效列數值有限、residual/loss 計算一致。
- [ ] 配對集合與排除數可重現；回顧與部署模式分開；合成及 mock 不進入正式推論。

以上為下一階段驗收條件，不代表目前已產出正式樣本外預測。
