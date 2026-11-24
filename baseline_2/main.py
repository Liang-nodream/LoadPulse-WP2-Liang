"""LoadPulse two-model baseline: choose synthetic data or real Australia AEMO CSVs.

Default: QLD1 2021H1 AEMO data; Jan-Apr training, May calibration, June pilot.
Run `python main.py --help` for options. Both models are calendar-only.
"""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from aemo_loader import AEST, REGION_SUBDIVISIONS, get_aemo_load_data, get_australian_holiday_dates
from data_generator import get_load_data
from failure_detector import detect_temporal_failures
from feature_engineering import build_calendar_features
from gam_model import predict_gam_out_of_sample, train_gam_model
from model_a import predict_model_a, train_model_a

BASE_DIR = Path(__file__).resolve().parent
LOCAL_TIMEZONE = AEST
ORIGIN_HOUR = 12  # target day forecast generated at previous day's 12:00 AEST
INTERVAL_DURATION = pd.Timedelta(hours=1)
RELEASE_DELAY = pd.Timedelta(0)  # optimistic assumption; AEMO publication delay not verified
RIDGE_ALPHA = 1.0
GAM_LAM = 10.0
THRESHOLD_MULTIPLIER = 2.5
PREDICTION_COLUMNS = [
    "timestamp_utc", "forecast_origin_utc", "model_id", "y_true", "y_pred",
    "horizon_hours", "is_valid", "quality_flag",
]
PERIODS = {
    "synthetic": {
        "train_start": "2023-01-01", "calibration_start": "2023-10-01",
        "pilot_start": "2023-11-01", "pilot_end": "2023-12-01",
    },
    "aemo": {
        "train_start": "2021-01-01", "calibration_start": "2021-05-01",
        "pilot_start": "2021-06-01", "pilot_end": "2021-07-01",
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare Ridge and GAM on synthetic or AEMO demand")
    parser.add_argument("--source", choices=("aemo", "synthetic"), default="aemo")
    parser.add_argument("--region", choices=[*REGION_SUBDIVISIONS, "all"], default="QLD1")
    parser.add_argument("--aemo-dir", type=Path, default=BASE_DIR / "Australia_AEMO",
                        help="Local folder containing downloaded AEMO_NEM_*_RAW_*.csv files")
    parser.add_argument("--models", choices=("both", "ridge", "gam"), default="both")
    parser.add_argument("--no-holidays", action="store_true",
                        help="AEMO pilot without state holiday indicators (temporary only)")
    for key in ("train_start", "calibration_start", "pilot_start", "pilot_end"):
        parser.add_argument("--" + key.replace("_", "-"), default=None,
                            help=f"Override {key} in YYYY-MM-DD format")
    return parser.parse_args(argv)


def build_forecast_schedule(start: str, end: str, local_timezone: str) -> pd.DataFrame:
    """Predict hourly targets at the previous AEST day's noon."""
    targets = pd.date_range(start, end, freq="h", inclusive="left", tz=local_timezone)
    local_day = targets.tz_localize(None).normalize()
    origins = (local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=ORIGIN_HOUR)).tz_localize(local_timezone)
    target_utc = targets.tz_convert("UTC")
    origin_utc = origins.tz_convert("UTC")
    return pd.DataFrame({
        "timestamp_utc": target_utc,
        "forecast_origin_utc": origin_utc,
        "horizon_hours": (target_utc - origin_utc) / pd.Timedelta(hours=1),
    })


def make_temporal_split(observations, calibration_schedule, periods):
    """Fit only from targets already observed at first calibration forecast origin."""
    local_times = observations["timestamp_utc"].dt.tz_convert(LOCAL_TIMEZONE)
    train_start = pd.Timestamp(periods["train_start"], tz=LOCAL_TIMEZONE)
    cal_start = pd.Timestamp(periods["calibration_start"], tz=LOCAL_TIMEZONE)
    first_calibration_origin = calibration_schedule["forecast_origin_utc"].min()
    train = observations.loc[
        (local_times >= train_start)
        & (local_times < cal_start)
        & (observations["available_at_utc"] <= first_calibration_origin)
        & np.isfinite(observations["y_true"])
    ].copy()
    return train


def build_predictions(model, predict_fn, model_id, schedule, observations, feature_options):
    features = build_calendar_features(schedule["timestamp_utc"], **feature_options)
    predictions = schedule.assign(model_id=model_id, y_pred=predict_fn(model, features)).merge(
        observations, on="timestamp_utc", how="left", validate="one_to_one"
    )
    predictions["is_valid"] = np.isfinite(predictions["y_true"])
    predictions["quality_flag"] = predictions["source_quality_flag"].fillna("missing_observation")
    return predictions.drop(columns=["source_quality_flag", "available_at_utc"])


def calculate_metrics(predictions: pd.DataFrame) -> dict:
    valid = predictions.loc[predictions["is_valid"]]
    if valid.empty:
        raise ValueError("No valid observations are available for evaluation. Check CSV dates and split settings.")
    error = valid["y_true"] - valid["y_pred"]
    return {
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "mae_mw": float(np.mean(error.abs())),
        "n_valid": len(valid),
    }


def export_predictions(predictions, output_dir, *, stem, display_name, source, region):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions[PREDICTION_COLUMNS].to_csv(output_dir / f"predictions_{stem}.csv", index=False)
    local_times = predictions["timestamp_utc"].dt.tz_convert(LOCAL_TIMEZONE)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(local_times, predictions["y_true"].where(predictions["is_valid"]), label="Actual")
    ax.plot(local_times, predictions["y_pred"], label=display_name, alpha=0.8)
    ax.set(xlabel=f"AEMO market time ({LOCAL_TIMEZONE})", ylabel="Demand (MW)",
           title=f"{display_name} | {source.upper()} {region} pilot")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(output_dir / f"{stem}_pilot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_source(args, region):
    if args.source == "synthetic":
        raw = get_load_data(local_timezone=LOCAL_TIMEZONE)
        holidays_set = set(raw.loc[raw["Is_Holiday"].eq(1), "Timestamp"].dt.date)
        source_files = []
    else:
        folder = args.aemo_dir if args.aemo_dir.is_absolute() else BASE_DIR / args.aemo_dir
        raw = get_aemo_load_data(folder, region=region)
        source_files = [p.name for p in sorted(folder.glob(f"AEMO_NEM_{region}_RAW_*.csv"))]
        years = set(raw["Timestamp"].dt.year.unique().tolist())
        if args.no_holidays:
            print("[注意] AEMO 假日特徵全部設為 0（僅供暫時測試）。")
            holidays_set = set()
        else:
            holidays_set = get_australian_holiday_dates(region, years)
    return raw, holidays_set, source_files


def run_one(args, region):
    periods = {key: getattr(args, key) or value for key, value in PERIODS[args.source].items()}
    if not (periods["train_start"] < periods["calibration_start"] < periods["pilot_start"] < periods["pilot_end"]):
        raise ValueError("Date split must satisfy train_start < calibration_start < pilot_start < pilot_end")

    raw, holiday_dates, source_files = _load_source(args, region)
    if raw.empty:
        raise ValueError("Source data is empty")
    if raw["Timestamp"].dt.tz is None:
        raise ValueError("Input timestamps must be timezone-aware")
    observations = pd.DataFrame({
        "timestamp_utc": raw["Timestamp"].dt.tz_convert("UTC"),
        "y_true": raw["Actual_Load"],
        "source_quality_flag": raw["quality_flag"] if args.source == "aemo" else "synthetic",
    })
    observations["available_at_utc"] = observations["timestamp_utc"] + INTERVAL_DURATION + RELEASE_DELAY
    feature_options = {"local_timezone": LOCAL_TIMEZONE, "holiday_dates": holiday_dates}
    cal_schedule = build_forecast_schedule(periods["calibration_start"], periods["pilot_start"], LOCAL_TIMEZONE)
    pilot_schedule = build_forecast_schedule(periods["pilot_start"], periods["pilot_end"], LOCAL_TIMEZONE)
    train = make_temporal_split(observations, cal_schedule, periods)
    if len(train) < 24:
        raise ValueError(f"Only {len(train)} training hours. Check file dates and --train-start etc.")
    first_pilot_origin = pilot_schedule["forecast_origin_utc"].min()
    training_features = build_calendar_features(train["timestamp_utc"], **feature_options)

    output_dir = BASE_DIR / "outputs" / ("aemo" if args.source == "aemo" else "synthetic")
    if args.source == "aemo":
        output_dir = output_dir / region
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "aemo":
        raw.to_csv(output_dir / f"aemo_hourly_{region}.csv", index=False)

    models = []
    if args.models in ("both", "ridge"):
        print(f"[{region} | Model A] 訓練 Ridge Regression...")
        ridge = train_model_a(training_features, train["y_true"], alpha=RIDGE_ALPHA)
        models.append(("model_a", "model_a_ridge_v1", "Model A (Ridge)", ridge, predict_model_a))
    if args.models in ("both", "gam"):
        print(f"[{region} | Model B] 訓練 GAM...")
        gam = train_gam_model(training_features, train["y_true"], lam=GAM_LAM)
        models.append(("gam", "model_b_gam_v1", "Model B (GAM)", gam, predict_gam_out_of_sample))

    comparison = pilot_schedule[["timestamp_utc", "forecast_origin_utc"]].copy()
    summary_rows = []
    masks = []
    for stem, model_id, display_name, model, predict_fn in models:
        cal = build_predictions(model, predict_fn, model_id, cal_schedule, observations, feature_options)
        cal = cal.loc[cal["timestamp_utc"] + INTERVAL_DURATION + RELEASE_DELAY <= first_pilot_origin]
        cal_metrics = calculate_metrics(cal)
        threshold = cal_metrics["rmse_mw"] * THRESHOLD_MULTIPLIER

        pilot = build_predictions(model, predict_fn, model_id, pilot_schedule, observations, feature_options)
        pilot_metrics = calculate_metrics(pilot)
        failures = detect_temporal_failures(pilot, threshold=threshold)
        export_predictions(pilot, output_dir, stem=stem, display_name=display_name,
                           source=args.source, region=region)

        valid = pilot["is_valid"].to_numpy(dtype=bool)
        absolute_error = (pilot["y_true"] - pilot["y_pred"]).abs().to_numpy()
        failure_mask = valid & (absolute_error > threshold)
        masks.append(failure_mask)
        comparison[f"y_pred_{stem}"] = pilot["y_pred"].to_numpy()
        comparison[f"abs_error_{stem}_mw"] = absolute_error
        comparison[f"failure_{stem}"] = failure_mask
        comparison["y_true"] = pilot["y_true"].to_numpy()
        comparison["is_valid"] = valid
        comparison["quality_flag"] = pilot["quality_flag"].to_numpy()

        metadata = {
            "data_source": args.source, "region": region, "source_files": source_files,
            "model_id": model_id, "commit": git_commit(),
            "hyperparameters": {"alpha": RIDGE_ALPHA} if stem == "model_a" else {"lam": GAM_LAM},
            "calendar_only": True, "holiday_calendar": "none" if args.no_holidays else (
                "synthetic" if args.source == "synthetic" else f"AU/{REGION_SUBDIVISIONS[region]}"),
            "timezone": LOCAL_TIMEZONE, "timezone_meaning": "AEMO fixed AEST market clock (not DST-adjusted state wall time)",
            "origin_local_hour": ORIGIN_HOUR, "target_timestamp": "interval_start",
            "release_delay_after_interval_end_hours": RELEASE_DELAY.total_seconds() / 3600,
            "refit_during_calibration_or_pilot": False,
            "configured_local_periods_end_exclusive": {
                "train": [periods["train_start"], periods["calibration_start"]],
                "calibration": [periods["calibration_start"], periods["pilot_start"]],
                "pilot": [periods["pilot_start"], periods["pilot_end"]],
            },
            "training_rows": len(train), "first_training_target_utc": train["timestamp_utc"].min().isoformat(),
            "last_training_target_utc": train["timestamp_utc"].max().isoformat(),
            "last_calibration_target_utc": cal["timestamp_utc"].max().isoformat(),
            "calibration": cal_metrics, "pilot": pilot_metrics,
            "threshold_multiplier": THRESHOLD_MULTIPLIER,
            "failure_threshold_mw": threshold, "candidate_failure_count": len(failures),
        }
        (output_dir / f"{stem}_experiment.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        summary_rows.append({
            "source": args.source, "region": region, "model_id": model_id,
            "train_rows": len(train), "calibration_valid_rows": cal_metrics["n_valid"],
            "calibration_rmse_mw": cal_metrics["rmse_mw"], "calibration_mae_mw": cal_metrics["mae_mw"],
            "pilot_rmse_mw": pilot_metrics["rmse_mw"], "pilot_mae_mw": pilot_metrics["mae_mw"],
            "pilot_valid_rows": pilot_metrics["n_valid"], "failure_threshold_mw": threshold,
            "candidate_failure_count": len(failures),
        })
        print(
            f"[{region} | {display_name} | {args.source.upper()} Pilot] "
            f"RMSE {pilot_metrics['rmse_mw']:.2f} MW, "
            f"MAE {pilot_metrics['mae_mw']:.2f} MW, "
            f"有效筆數 {pilot_metrics['n_valid']}"
        )

    if len(masks) == 2:
        comparison["common_failure"] = np.logical_and.reduce(masks)
        comparison.loc[comparison["common_failure"]].to_csv(
            output_dir / "common_failure_candidates.csv", index=False
        )
        print(f"[{region} | 跨模型比對] 共同失效候選：{int(comparison['common_failure'].sum())} 筆")
    comparison.to_csv(output_dir / "pilot_model_comparison.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "model_metrics_comparison.csv", index=False)
    print(f"[{region} | 完成] 結果儲存至：{output_dir}")
    return summary_rows


def main(argv=None):
    args = parse_args(argv)
    regions = ([*REGION_SUBDIVISIONS] if args.region == "all" and args.source == "aemo"
               else ["SYNTHETIC" if args.source == "synthetic" else args.region])
    all_rows = []
    for region in regions:
        all_rows.extend(run_one(args, region))
    if args.source == "aemo" and len(regions) > 1:
        combined_dir = BASE_DIR / "outputs" / "aemo"
        pd.DataFrame(all_rows).to_csv(combined_dir / "all_regions_metrics.csv", index=False)
        print(f"[AEMO | all] 跨區域指標已儲存：{combined_dir / 'all_regions_metrics.csv'}")


if __name__ == "__main__":
    main()
