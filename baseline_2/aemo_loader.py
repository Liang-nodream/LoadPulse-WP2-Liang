"""Read AEMO NEM 2021H1 regional demand CSVs and aggregate to hourly MW.

AEMO SETTLEMENTDATE is the *END* of the 30-minute trading interval in
Australian Eastern Standard Time (AEST, UTC+10; no daylight saving time).
For each half-hourly row, interval start = SETTLEMENTDATE - 30 minutes.
Two valid half-hours are averaged (MW, NOT summed) for each hourly target.

Inputs: AEMO_NEM_{REGION}_RAW_*.csv, with REGION, SETTLEMENTDATE,
        TOTALDEMAND, RRP, PERIODTYPE.
Outputs: Timestamp (timezone-aware hour start), Actual_Load (MW), REGION,
         quality_flag and n_valid_halfhours.
"""

from pathlib import Path

import numpy as np
import pandas as pd

AEST = "Australia/Brisbane"  # AEMO market clock: UTC+10 throughout the year.
REGION_SUBDIVISIONS = {
    "NSW1": "NSW", "QLD1": "QLD", "VIC1": "VIC", "SA1": "SA", "TAS1": "TAS",
}
REQUIRED_COLUMNS = {"REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP", "PERIODTYPE"}


def get_aemo_load_data(aemo_dir: str | Path, *, region: str = "QLD1") -> pd.DataFrame:
    """Read one region's raw AEMO CSV(s); return a regular AEST hourly grid.

    Incomplete/invalid hours are preserved with NaN and a quality flag so
    evaluation can exclude them without silently shifting timestamps.
    """
    region = region.upper()
    if region not in REGION_SUBDIVISIONS:
        raise ValueError(f"Unknown AEMO region {region!r}. Choose {list(REGION_SUBDIVISIONS)}")

    folder = Path(aemo_dir).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(
            f"AEMO data folder not found: {folder}\n"
            "Download/sync the CSV files from Australia_AEMO to this local folder, "
            "or set --aemo-dir to their actual location."
        )
    paths = sorted(folder.glob(f"AEMO_NEM_{region}_RAW_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No AEMO CSV for {region} in {folder}. "
            f"Expected a file like AEMO_NEM_{region}_RAW_2021H1.csv"
        )

    frames = []
    for path in paths:
        data = pd.read_csv(path, encoding="utf-8-sig")
        missing = REQUIRED_COLUMNS.difference(data.columns)
        if missing:
            raise ValueError(f"{path.name}: missing CSV columns {sorted(missing)}")
        if not data["REGION"].eq(region).all():
            raise ValueError(f"{path.name}: REGION contains values other than {region}")
        if not data["PERIODTYPE"].eq("TRADE").all():
            raise ValueError(f"{path.name}: expected only TRADE half-hour intervals")
        frames.append(data.loc[:, ["SETTLEMENTDATE", "TOTALDEMAND"]].copy())

    df = pd.concat(frames, ignore_index=True)
    # Deliberately do not parse the local clock as a region's DST-adjusted wall time.
    try:
        period_end = pd.to_datetime(
            df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S", errors="raise"
        ).dt.tz_localize(AEST)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid AEMO SETTLEMENTDATE format") from exc
    if period_end.isna().any() or (period_end.dt.minute % 30 != 0).any() or period_end.dt.second.ne(0).any():
        raise ValueError("AEMO timestamps must be valid half-hour boundaries")
    if period_end.duplicated().any():
        raise ValueError("Duplicate settlement timestamps; overlapping CSV files or bad data")

    interval_start = period_end - pd.Timedelta(minutes=30)
    demand = pd.to_numeric(df["TOTALDEMAND"], errors="coerce")
    demand = demand.where(np.isfinite(demand) & demand.gt(0))
    halfhour = pd.Series(demand.to_numpy(dtype=float), index=pd.DatetimeIndex(interval_start))
    halfhour = halfhour.sort_index()

    # For the 2021H1 files, each hour must comprise TWO distinct 30-min rows.
    # resample produces NaN/0-count hours for gaps without dropping them.
    grouped = halfhour.resample("h").agg(["mean", "count"])
    grouped.columns = ["Actual_Load", "n_valid_halfhours"]
    if grouped.empty:
        raise ValueError("AEMO CSV contains no observations")
    grouped["Actual_Load"] = grouped["Actual_Load"].where(grouped["n_valid_halfhours"].eq(2))
    grouped["quality_flag"] = np.where(
        grouped["n_valid_halfhours"].eq(2), "aemo_observed", "missing_or_invalid_halfhour"
    )
    result = grouped.reset_index(names="Timestamp")
    result["REGION"] = region
    result = result[["Timestamp", "Actual_Load", "REGION", "n_valid_halfhours", "quality_flag"]]

    n_good = int(result["Actual_Load"].notna().sum())
    print(
        f"[AEMO] {region}: {len(df)} 筆 30 分鐘資料 → "
        f"{len(result)} 筆每小時資料（有效 {n_good}，不完整 {len(result)-n_good}）"
    )
    return result


def get_australian_holiday_dates(region: str, years: set[int]) -> set:
    """State-specific public holidays (dates known before forecasting).

    Requires `pip install holidays`. This is separate from the raw demand data.
    AEMO model clock is fixed AEST: state DST/holiday clock alignment is an
    approximation outside QLD and needs care for a production study.
    """
    region = region.upper()
    if region not in REGION_SUBDIVISIONS:
        raise ValueError(f"Unknown AEMO region {region!r}")
    try:
        import holidays
    except ImportError as exc:
        raise ImportError(
            "Real AEMO runs need state public holidays. Install: pip install holidays\n"
            "For a temporary calendar-without-holidays pilot, pass --no-holidays "
            "(and document this limitation)."
        ) from exc
    calendar = holidays.country_holidays("AU", subdiv=REGION_SUBDIVISIONS[region], years=sorted(years))
    return set(calendar.keys())
