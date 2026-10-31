"""Leakage-aware, per-engine trailing time-series feature engineering."""
import pandas as pd
from .data import ID_COL, TIME_COL, SETTING_COLS, SENSOR_COLS
from .config import ROLLING_WINDOWS

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {ID_COL, TIME_COL, *SETTING_COLS, *SENSOR_COLS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ordered = df.sort_values([ID_COL, TIME_COL]).copy()
    out = ordered[[ID_COL, TIME_COL, *SETTING_COLS]].copy()

    for sensor in SENSOR_COLS:
        out[f"{sensor}_current"] = ordered[sensor].astype(float)
        grouped = ordered.groupby(ID_COL, sort=False)[sensor]
        for window in ROLLING_WINDOWS:
            out[f"{sensor}_mean_{window}"] = grouped.transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
            out[f"{sensor}_std_{window}"] = grouped.transform(
                lambda s: s.rolling(window=window, min_periods=1).std().fillna(0.0)
            )

    return out.replace([float("inf"), float("-inf")], 0).fillna(0.0)
