import numpy as np
import pandas as pd


def get_load_data(*, local_timezone: str | None = None) -> pd.DataFrame:
    """Generate reproducible 2023 test data, optionally on a local hourly grid."""
    print("[模組 1] 正在產生合成負載資料")
    timestamps = pd.date_range(
        start="2023-01-01",
        end="2024-01-01",
        freq="h",
        inclusive="left",
        tz=local_timezone,
    )

    rng = np.random.RandomState(42)
    daily_cycle = 500 * np.sin(np.pi * timestamps.hour / 12)
    noise = rng.normal(0, 100, len(timestamps))
    return pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Actual_Load": 3000 + daily_cycle + noise,
            "Is_Holiday": (timestamps.dayofyear % 40 == 0).astype(int),
        }
    )
