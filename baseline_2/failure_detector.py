import math

import numpy as np
import pandas as pd


def detect_temporal_failures(
    result_df: pd.DataFrame,
    rmse_baseline: float | None = None,
    multiplier: float = 2.5,
    *,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Return valid rows whose absolute error exceeds a fixed threshold in MW.

    Pass a threshold determined before the scan, or pass a baseline RMSE to
    multiply by the legacy factor. For a held-out scan, use calibration RMSE.
    """
    if (threshold is None) == (rmse_baseline is None):
        raise ValueError("Provide exactly one of threshold or rmse_baseline.")

    if threshold is None:
        if not math.isfinite(rmse_baseline) or rmse_baseline < 0:
            raise ValueError("rmse_baseline must be finite and non-negative.")
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("multiplier must be finite and positive.")
        threshold = rmse_baseline * multiplier
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative.")

    if {"y_true", "y_pred"}.issubset(result_df.columns):
        absolute_error = (result_df["y_true"] - result_df["y_pred"]).abs()
    else:
        absolute_error = result_df["Error"]

    eligible = np.isfinite(absolute_error) & (absolute_error > threshold)
    if "is_valid" in result_df:
        validity = result_df["is_valid"]
        if not validity.dropna().isin([True, False]).all():
            raise ValueError("is_valid must contain boolean flags.")
        eligible &= validity.eq(True)

    failures = result_df.loc[eligible].copy()
    failures["Error"] = absolute_error.loc[eligible]
    print(f"[模組 4] 失效門檻：絕對誤差大於 {threshold:.2f} MW")
    print(f"共偵測到 {len(failures)} 筆候選失效時段。")
    return failures