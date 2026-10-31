from tests.test_features import tiny_frame
from src.anomaly import add_baseline_anomaly_score

def test_anomaly_monitor_returns_expected_columns():
    result = add_baseline_anomaly_score(tiny_frame(), baseline_cycles=2)
    assert {"unit", "cycle", "anomaly_score", "anomaly_flag"}.issubset(result.columns)
    assert len(result) == 6
