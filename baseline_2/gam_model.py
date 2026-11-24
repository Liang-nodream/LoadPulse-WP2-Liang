"""Model B: calendar-only generalized additive model (GAM) using pyGAM.

Ridge and GAM receive the same *information*: hour, weekday, holiday flag,
and annual position.  GAM additionally learns nonlinear periodic effects, in
particular an hour-of-week profile.  Historical load lags are intentionally
excluded to prevent using observations unavailable at the forecast origin.
"""

import numpy as np
import pandas as pd


# Values are shared between fitting and forecasting through this conversion.
FEATURE_COLUMNS = [
    "Hour_Of_Day", "Day_Of_Week", "Is_Holiday", "Year_Sin", "Year_Cos"
]


def _gam_features(calendar_features: pd.DataFrame) -> np.ndarray:
    """Convert public calendar columns to daily, weekly and yearly phases."""
    missing = set(FEATURE_COLUMNS).difference(calendar_features.columns)
    if missing:
        raise ValueError(f"Missing GAM calendar columns: {sorted(missing)}")

    hour = calendar_features["Hour_Of_Day"].to_numpy(dtype=float)
    weekday = calendar_features["Day_Of_Week"].to_numpy(dtype=float)
    holiday = calendar_features["Is_Holiday"].to_numpy(dtype=float)
    annual_sin = calendar_features["Year_Sin"].to_numpy(dtype=float)
    annual_cos = calendar_features["Year_Cos"].to_numpy(dtype=float)

    hour_of_week = weekday * 24.0 + hour
    year_phase = np.mod(np.arctan2(annual_sin, annual_cos), 2 * np.pi) / (2 * np.pi)
    matrix = np.column_stack([hour, hour_of_week, year_phase, holiday])
    if not np.isfinite(matrix).all():
        raise ValueError("GAM calendar features contain missing or infinite values.")
    return matrix


def train_gam_model(
    calendar_features: pd.DataFrame,
    actual_load: pd.Series,
    *,
    lam: float = 10.0,
):
    """Fit a periodic additive model on the same training rows as Model A."""
    try:
        from pygam import LinearGAM, f, s
    except ImportError as exc:
        raise ImportError("GAM requires pyGAM. Install it with: pip install pygam") from exc

    if not calendar_features.index.equals(actual_load.index):
        raise ValueError("Calendar features and actual load must have matching indices.")

    X = _gam_features(calendar_features)
    y = actual_load.to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Training load contains missing or infinite values.")

    # cp = cyclic P-spline: period boundaries connect smoothly.
    # HoD: 0..24, HoW: 0..168, annual phase: 0..1.
    terms = (
        s(0, n_splines=24, basis="cp", edge_knots=[0, 24])
        + s(1, n_splines=20, basis="cp", edge_knots=[0, 168])
        + s(2, n_splines=12, basis="cp", edge_knots=[0, 1])
        + f(3)
    )
    model = LinearGAM(terms, lam=lam)
    return model.fit(X, y)


def predict_gam_out_of_sample(model, calendar_features: pd.DataFrame) -> pd.Series:
    """Generate GAM forecasts without using observed target-period loads."""
    return pd.Series(
        model.predict(_gam_features(calendar_features)),
        index=calendar_features.index,
        name="y_pred",
    )
