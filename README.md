# Netreaper - ML Security Hackathon Project

Adversarial-resilient intrusion detection on NSL-KDD.

## Current Scope

- Stage A: Binary intrusion classification baseline (`stage_a_baseline.py`)
- Stage B: Adversarial attack simulation for Person 3 (`attack_simulation.py`, `run_stage_b_attack.py`)
- Stage C: Anomaly detection (`python stage_c_anomaly.py`,`python stage_c_test.py`)

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

## Stage C (Person 4) - Anomaly Detection (Safety Net)
Stage C implements an unsupervised "Safety Net" designed to catch adversarial samples that successfully bypass the Stage A baseline.

### Key Features

- Unsupervised Learning: Utilizes an IsolationForest trained exclusively on "normal" traffic to establish a secure behavioral baseline.
- Explainability: Integrated SHAP (SHapley Additive exPlanations) to provide transparency into why specific traffic is flagged.
- Adversarial Resilience: Designed to detect "outlier" patterns in adversarial samples that appear normal to supervised classifiers.

Run Anomaly Training:
To train the Isolation Forest and generate the SHAP explainability plot:
```bash
python stage_c_anomaly.py
```

Run Adversarial Verification:
To test the Safety Net against the stealth attacks generated in Stage B:
```bash
python stage_c_test.py
```

Stage C runs Isolation Forest and reports: 
- Adversarial Detection Rate: Reports the percentage of Stage B attacks successfully flagged as anomalies.
- SHAP Summary Plot: Visualizes feature-level contributions to anomaly scores (Explainable AI).
- Model Artifact: stage_c_isolation_forest.pkl for final system integration.
- Anomaly Metrics: Total Attacks Tested vs. Attacks Caught.
- Explainability: Feature importance visualization saved to stage_c_shap_summary.png.

## Integration Notes

- Person 3 (this module) provides adversarial robustness metrics.
- Person 5 can combine classifier output + anomaly output using:
	- `(clf_pred == 1) OR (anomaly_pred == -1)`

See `handover.md` and `context.md` for teammate-facing handoff details.

