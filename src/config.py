from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

# NASA C-MAPSS FD001 benchmark settings
SUBSET = "FD001"
RUL_CAP = 125
ROLLING_WINDOWS = (5, 10)
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
N_ESTIMATORS = 350
