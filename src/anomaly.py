"""Simple exploratory sensor-deviation monitor."""
import pandas as pd
from .data import ID_COL, TIME_COL, SENSOR_COLS

def add_baseline_anomaly_score(
    df: pd.DataFrame, baseline_cycles: int = 10, threshold: float = 3.0
) -> pd.DataFrame:
    """
    Calculate a per-row maximum absolute sensor z-score against each engine's
    first `baseline_cycles` observations.

    This is an exploratory deviation flag, not a validated fault detector.
    """
    if baseline_cycles < 2:
        raise ValueError("baseline_cycles must be at least 2")

    ordered = df.sort_values([ID_COL, TIME_COL]).copy()
    result = ordered[[ID_COL, TIME_COL]].copy()
    scores = []

    for unit_id, group in ordered.groupby(ID_COL, sort=False):
        baseline = group.head(baseline_cycles)[SENSOR_COLS]
        means = baseline.mean()
        stds = baseline.std(ddof=0).replace(0, 1e-8)
        z = (group[SENSOR_COLS] - means) / stds
        score = z.abs().max(axis=1)
        scores.append(pd.DataFrame({
            ID_COL: unit_id,
            TIME_COL: group[TIME_COL].to_numpy(),
            "anomaly_score": score.to_numpy(),
            "anomaly_flag": (score.to_numpy() >= threshold),
        }))
    if not scores:
        return pd.DataFrame(columns=[ID_COL, TIME_COL, "anomaly_score", "anomaly_flag"])
    return pd.concat(scores, ignore_index=True)
