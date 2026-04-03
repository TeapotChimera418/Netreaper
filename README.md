# Netreaper - MLShark Hackathon Project

Adversarial-resilient intrusion detection on NSL-KDD.

## Current Scope

- Stage A: Binary intrusion classification baseline (`stage_a_baseline.py`)
- Stage B: Adversarial attack simulation for Person 3 (`attack_simulation.py`, `run_stage_b_attack.py`)
- Stage C: Anomaly detection (planned)

## Dataset

- File: `KDDTrain_with_headers.csv`
- Target column: `label`
- Stage A binary mapping:
	- `normal` -> `0`
	- all attack labels -> `1`

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run Stage A Baseline

```bash
python stage_a_baseline.py
```

Stage A runs Random Forest + XGBoost and reports:

- Accuracy, Precision, Recall, F1, ROC-AUC
- confusion matrix plots
- feature importance plots
- ROC curves
- comparison summary table

## Run Stage B (Person 3) Attack Simulation

### 1) Dummy mode (parallel-friendly)

```bash
python run_stage_b_attack.py
```

Use this while data preprocessing is still in progress.

### 2) Real data mode (plug-in after data pipeline)

```bash
python run_stage_b_attack.py --real-data --csv-path KDDTrain_with_headers.csv --model-type rf
```

Optional model type:

- `--model-type rf`
- `--model-type xgb`

Stage B reports:

- clean accuracy
- adversarial accuracy
- accuracy drop
- attack success rate
- average L2 perturbation
- average L-infinity perturbation

## Integration Notes

- Person 3 (this module) provides adversarial robustness metrics.
- Person 5 can combine classifier output + anomaly output using:
	- `(clf_pred == 1) OR (anomaly_pred == -1)`

See `handover.md` and `context.md` for teammate-facing handoff details.

