"""Metrics and plotting helpers."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae_cycles": float(mean_absolute_error(y_true, y_pred)),
        "rmse_cycles": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }

def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def plot_rul_distribution(y: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.hist(y, bins=30, edgecolor="black")
    plt.xlabel("Capped remaining useful life (cycles)")
    plt.ylabel("Number of observations")
    plt.title("Training target distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_engine_trajectories(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sensor = "sensor_2"
    plt.figure(figsize=(9, 5))
    for unit_id, group in df.groupby("unit"):
        if unit_id <= 8:
            plt.plot(group["cycle"], group[sensor], alpha=0.75, linewidth=1)
    plt.xlabel("Operating cycle")
    plt.ylabel(sensor)
    plt.title(f"Example engine trajectories: {sensor}")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_predictions(y_true, y_pred, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.35)
    limit = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([0, limit], [0, limit], linestyle="--")
    plt.xlabel("Actual RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title("Validation: actual vs predicted RUL")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
