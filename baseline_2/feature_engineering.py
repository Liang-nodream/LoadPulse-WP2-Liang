from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd


def build_calendar_features(
    timestamps: pd.Series | pd.DatetimeIndex,
    *,
    local_timezone: str,
    holiday_dates: Iterable[date],
) -> pd.DataFrame:
    """Build Model A features from timezone-aware target times and local holiday dates.

    Preserve the input Series index, or use a RangeIndex for a DatetimeIndex.
    """
    times = pd.DatetimeIndex(timestamps)
    if times.tz is None:
        raise ValueError("Target timestamps must include a timezone.")
    if times.hasnans:
        raise ValueError("Target timestamps must not contain NaT.")

    local_times = times.tz_convert(local_timezone)
    days_in_year = np.where(local_times.is_leap_year, 366, 365)
    annual_angle = 2 * np.pi * (local_times.dayofyear - 1) / days_in_year
    holidays = pd.Index(local_times.date).isin(list(holiday_dates))

    return pd.DataFrame(
        {
            "Hour_Of_Day": local_times.hour,
            "Day_Of_Week": local_times.dayofweek,
            "Is_Weekend": (local_times.dayofweek >= 5).astype("int8"),
            "Is_Holiday": holidays.astype("int8"),
            "Year_Sin": np.sin(annual_angle),
            "Year_Cos": np.cos(annual_angle),
        },
        index=timestamps.index if isinstance(timestamps, pd.Series) else None,
    )


def preprocess_features(df, time_col='Timestamp', target_col='Actual_Load'):
    print("[模組 2] 正在進行特徵萃取 (HoD, HoW, HoY, 歷史慣性)...")
    df = df.copy()
    
    df['HoD'] = df[time_col].dt.hour
    df['HoW'] = df[time_col].dt.dayofweek * 24 + df['HoD']
    df['HoY'] = df[time_col].dt.dayofyear * 24 + df['HoD']
    
    # 萃取昨天同一時間的用電量
    df['Load_Lag_24h'] = df[target_col].shift(24)
    
    # 清除因為 shift 產生的空值
    df = df.dropna().reset_index(drop=True)
    return df