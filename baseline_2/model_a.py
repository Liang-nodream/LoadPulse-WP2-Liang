"""Model A: linear calendar-only load baseline (Ridge regression)."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_model_a(
    calendar_features: pd.DataFrame,
    actual_load: pd.Series,
    *,
    alpha: float = 1.0,
) -> Pipeline:
    """Fit Model A using only calendar information available before forecasting."""
    if not calendar_features.index.equals(actual_load.index):
        raise ValueError("Calendar features and actual load must have matching indices.")

    preprocessing = ColumnTransformer(
        [
            (
                "hour_and_weekday",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Hour_Of_Day", "Day_Of_Week"],
            ),
            (
                "holiday_and_year",
                "passthrough",
                ["Is_Holiday", "Year_Sin", "Year_Cos"],
            ),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [("calendar_features", preprocessing), ("ridge", Ridge(alpha=alpha))]
    )
    return model.fit(calendar_features, actual_load)


def predict_model_a(
    model: Pipeline,
    calendar_features: pd.DataFrame,
) -> pd.Series:
    """Generate out-of-sample Model A predictions, preserving the row index."""
    return pd.Series(
        model.predict(calendar_features),
        index=calendar_features.index,
        name="y_pred",
    )
