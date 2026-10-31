# Predictive Maintenance with Time-Series Sensor Data

![CI](https://github.com/nasrin-habibian/predictive-maintenance-time-series/actions/workflows/ci.yml/badge.svg)

**End-to-end machine-learning portfolio project**  
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib · pytest

## Project overview

This repository demonstrates an end-to-end predictive-maintenance workflow using the **NASA C-MAPSS turbofan engine degradation benchmark (FD001)**.

The objective is to estimate an engine's **remaining useful life (RUL)** from multivariate sensor measurements collected over successive operating cycles. The project includes:

1. Downloading and loading the public NASA C-MAPSS dataset.
2. Exploratory analysis of engine run-to-failure trajectories.
3. Time-series feature engineering using trailing rolling windows.
4. A leakage-aware train/validation split by engine ID.
5. A Random Forest regression baseline for RUL estimation.
6. Evaluation with MAE, RMSE and R².
7. An anomaly-monitoring example based on sensor deviations from each engine's early-life baseline.
8. Saving a trained model and metadata for reuse.
9. A command-line prediction script and automated tests.

> **Important scope note:** C-MAPSS is a simulated turbofan engine degradation benchmark. It is useful for demonstrating predictive-maintenance methods, but it is not real Finnair aircraft data and does not establish operational readiness or aviation safety suitability. Model outputs are research estimates, not maintenance instructions.

## Dataset

The project uses the FD001 subset from NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset.

- Training file: `train_FD001.txt`
- Test file: `test_FD001.txt`
- RUL file: `RUL_FD001.txt`
- Each row represents one engine at one operating cycle.
- The dataset contains engine identifiers, cycle number, operating settings and 21 sensor channels.

The downloader retrieves the original dataset archive from NASA's Prognostics Center of Excellence repository. If the download URL changes, download the C-MAPSS data manually from the NASA repository and place the three FD001 files in `data/raw/`.

## Quick start

### 1. Create an environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

```bash
python -m src.download_data
```

If NASA's server is temporarily unavailable, download C-MAPSS manually and put `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in `data/raw/`.

### 4. Run the complete pipeline

```bash
python -m src.train
```

The pipeline creates:
- `reports/metrics.json`
- `reports/validation_predictions.csv`
- `reports/feature_importance.csv`
- `reports/figures/rul_distribution.png`
- `reports/figures/engine_sensor_trajectories.png`
- `reports/figures/validation_predictions.png`
- `models/rul_random_forest.joblib`
- `models/model_metadata.json`

### 5. Predict RUL for the test engines

```bash
python -m src.predict
```

This writes `reports/test_predictions.csv`.

### 6. Run tests

```bash
pytest -q
```

## Methodology

### Target construction

For each training engine, the final observed cycle is treated as the failure cycle. RUL at cycle `t` is:

`RUL = final_cycle_for_engine - t`

To reduce the influence of very large early-life RUL values, the target is capped at 125 cycles, a common benchmark convention:

`RUL_capped = min(RUL, 125)`

### Time-series feature engineering

For each sensor channel, the pipeline calculates trailing rolling mean and standard deviation over windows of 5 and 10 cycles. It also includes the current sensor reading and the engine's cycle count.

All rolling windows are calculated separately for each engine and use only current and past observations. This avoids using future sensor readings to predict the present.

### Validation design

Engine IDs are split into training and validation groups. No engine appears in both groups. This is more appropriate than randomly splitting individual rows, which can leak engine-specific patterns between training and validation.

### Model

A `RandomForestRegressor` is used as a robust, interpretable baseline. The project reports:
- **MAE:** average absolute RUL prediction error, in cycles.
- **RMSE:** square-root mean squared error, in cycles; penalizes larger errors more strongly.
- **R²:** proportion of target variance explained by the model.

Results depend on the dataset version, split and software environment. The pipeline writes the measured results to `reports/metrics.json`; no performance figure is claimed in advance.

### Anomaly monitoring

The example anomaly monitor standardizes sensor values relative to each engine's first 10 observed cycles and flags unusually large deviations. This is an exploratory monitoring heuristic, not a validated fault classifier. It is deliberately kept separate from the RUL model.

## Repository structure

```text
predictive-maintenance-time-series/
├── data/
│   └── raw/                  # NASA data files are downloaded here
├── models/                   # Trained model and metadata
├── reports/
│   └── figures/              # Generated charts
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── download_data.py
│   ├── features.py
│   ├── evaluate.py
│   ├── anomaly.py
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── test_features.py
│   └── test_anomaly.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Reproducibility

The random seed and modelling parameters are defined in `src/config.py`. Generated artifacts are excluded from Git by default; you can commit selected report images if you want to display results on your GitHub repository.

## Limitations and next steps

- FD001 is simulated benchmark data with one operating condition and one fault mode.
- A single benchmark model cannot be transferred directly to a real aircraft fleet.
- The anomaly score is a simple baseline and requires domain-specific validation.
- Further work for an operational system would require representative fleet data, robust drift monitoring, uncertainty estimation, human review, safety assurance and approval under applicable aviation processes.

## Suggested GitHub description

> End-to-end predictive-maintenance project using NASA C-MAPSS multivariate engine sensor time series. Includes leakage-aware feature engineering, remaining-useful-life regression, anomaly monitoring, evaluation, reproducible training and tests.
