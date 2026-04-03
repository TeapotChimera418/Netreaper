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

The script runs both Random Forest and XGBoost baselines, prints metrics, and renders:

- confusion matrices
- feature importance plots
- ROC curves
- comparison summary table

