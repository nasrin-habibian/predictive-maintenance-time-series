"""Predict RUL for the final observed cycle of each test engine."""
import json
import joblib
import pandas as pd
from .config import MODEL_DIR, REPORT_DIR
from .data import load_test, ID_COL, TIME_COL
from .features import build_features
from .evaluate import regression_metrics

def main() -> None:
    model_path = MODEL_DIR / "rul_random_forest.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Train the model first: python -m src.train")

    raw, true_rul = load_test()
    features = build_features(raw)
    last_rows = features.groupby(ID_COL, sort=False).tail(1).copy()
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    X = last_rows[metadata["feature_columns"]]
    predictions = model.predict(X).clip(0, metadata["rul_cap"])

    output = pd.DataFrame({
        "unit": last_rows[ID_COL].to_numpy(),
        "last_observed_cycle": last_rows[TIME_COL].to_numpy(),
        "predicted_rul_cycles": predictions,
        "actual_rul_cycles": true_rul,
    })
    output.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    print("Test-set metrics:")
    print(json.dumps(regression_metrics(true_rul, predictions), indent=2))
    print(f"Predictions saved to {REPORT_DIR / 'test_predictions.csv'}")

if __name__ == "__main__":
    main()
