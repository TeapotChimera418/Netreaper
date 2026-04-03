# Netreaper Stage A (Clean Baseline)

This workspace is cleaned to a minimal Stage A setup for binary intrusion detection on NSL-KDD.

## Kept Files

- `stage_a_baseline.py`: end-to-end Stage A baseline pipeline.
- `KDDTrain_with_headers.csv`: dataset with headers.
- `requirements.txt`: runtime dependencies.
- `context.md`: project context and roadmap.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python stage_a_baseline.py
```

## Stage B (Person 3) - Attack Simulation

Run with dummy data (parallel-friendly while data pipeline is still in progress):

```powershell
python run_stage_b_attack.py
```

Run with real NSL-KDD split using Stage A preprocessing:

```powershell
python run_stage_b_attack.py --real-data --csv-path KDDTrain_with_headers.csv --model-type rf
```

The script runs both Random Forest and XGBoost baselines, prints metrics, and renders:

- confusion matrices
- feature importance plots
- ROC curves
- comparison summary table

