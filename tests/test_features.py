import pandas as pd
from src.features import build_features
from src.data import SETTING_COLS, SENSOR_COLS

def tiny_frame():
    rows = []
    for unit in [1, 2]:
        for cycle in [1, 2, 3]:
            row = {"unit": unit, "cycle": cycle}
            row.update({c: 0.1 for c in SETTING_COLS})
            row.update({c: float(cycle) for c in SENSOR_COLS})
            rows.append(row)
    return pd.DataFrame(rows)

def test_features_keep_one_row_per_observation():
    raw = tiny_frame()
    result = build_features(raw)
    assert len(result) == len(raw)
    assert not result.isna().any().any()

def test_rolling_features_are_engine_specific():
    raw = tiny_frame()
    result = build_features(raw)
    # First observation for each engine has a rolling mean equal to its own value.
    first = result[result["cycle"] == 1]
    assert (first["sensor_1_mean_5"] == 1.0).all()
