"""Dataset loading and target construction for NASA C-MAPSS FD001."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import RAW_DIR, SUBSET, RUL_CAP

ID_COL = "unit"
TIME_COL = "cycle"
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = [ID_COL, TIME_COL, *SETTING_COLS, *SENSOR_COLS]

def load_train(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_DIR / f"train_{SUBSET}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m src.download_data` first."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = ALL_COLS
    final_cycles = df.groupby(ID_COL)[TIME_COL].max().rename("final_cycle")
    df = df.join(final_cycles, on=ID_COL)
    df["rul"] = (df["final_cycle"] - df[TIME_COL]).clip(upper=RUL_CAP)
    return df.drop(columns="final_cycle")

def load_test(path: Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    path = path or RAW_DIR / f"test_{SUBSET}.txt"
    rul_path = RAW_DIR / f"RUL_{SUBSET}.txt"
    if not path.exists() or not rul_path.exists():
        raise FileNotFoundError(
            "Test files are missing. Run `python -m src.download_data` first."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None).dropna(axis=1, how="all")
    df.columns = ALL_COLS
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    return df, true_rul
