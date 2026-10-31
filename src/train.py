"""Train, validate and save the RUL regression model."""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from .config import (
    MODEL_DIR, REPORT_DIR, FIGURE_DIR, RANDOM_STATE, VALIDATION_SIZE, N_ESTIMATORS
)
from .data import load_train, ID_COL
from .features import build_features
from .evaluate import (
    regression_metrics, save_json, plot_rul_distribution,
    plot_engine_trajectories, plot_predictions
)
from .anomaly import add_baseline_anomaly_score

def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_train()
    features = build_features(raw)
    target = raw["rul"].astype(float)

    # Split by engine, not by individual rows, to avoid engine-level leakage.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(splitter.split(features, target, groups=raw[ID_COL]))

    drop_cols = [ID_COL]
    X = features.drop(columns=drop_cols)
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_val).clip(0, 125)

    metrics = regression_metrics(y_val, predictions)
    metrics["validation_engine_count"] = int(raw.iloc[val_idx][ID_COL].nunique())
    metrics["model"] = "RandomForestRegressor"
    metrics["target"] = "RUL capped at 125 cycles"
    save_json(metrics, REPORT_DIR / "metrics.json")

    val_output = pd.DataFrame({
        "unit": raw.iloc[val_idx][ID_COL].to_numpy(),
        "cycle": raw.iloc[val_idx]["cycle"].to_numpy(),
        "actual_rul": y_val.to_numpy(),
        "predicted_rul": predictions,
    })
    val_output.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    joblib.dump(model, MODEL_DIR / "rul_random_forest.joblib")
    metadata = {
        "model": "RandomForestRegressor",
        "feature_columns": list(X.columns),
        "rul_cap": 125,
        "random_state": RANDOM_STATE,
        "training_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "note": "NASA C-MAPSS FD001 benchmark; research use only."
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    plot_rul_distribution(target, FIGURE_DIR / "rul_distribution.png")
    plot_engine_trajectories(raw, FIGURE_DIR / "engine_sensor_trajectories.png")
    plot_predictions(y_val, predictions, FIGURE_DIR / "validation_predictions.png")

    anomaly = add_baseline_anomaly_score(raw)
    anomaly.to_csv(REPORT_DIR / "training_anomaly_scores.csv", index=False)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved under: {REPORT_DIR} and {MODEL_DIR}")

if __name__ == "__main__":
    main()
