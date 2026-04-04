# Handover Notes

## Project Context

- Project: MLShark - Adversarial Resilient Intrusion Detection System
- Current repo style: minimal script-based layout (no `src/` package tree)
- Delivery status:
	- Stage A (classification): implemented and runnable
	- Stage B (adversarial attack simulation): implemented for Person 3
	- Stage C (anomaly detection): pending

## What Is Implemented Now

### Stage A (baseline classification)

- File: `stage_a_baseline.py`
- Key functions available for reuse:
	- `preprocess_data(df)`
	- `train_model(x_train, y_train, model_type)`
	- `evaluate_model(model, x_test, y_test, model_name)`
	- `main(df)`
- Binary target mapping:
	- `label == "normal"` -> `0`
	- any other label -> `1`

### Stage B (Person 3 attack track)

- New file: `attack_simulation.py`
	- `AttackEvaluation` dataclass
	- `StageBAttackSimulator` class
		- `generate_adversarial(x_test)` using Gaussian noise
		- `evaluate(model, x_test, y_test)` returning:
			- `clean_accuracy`
			- `adversarial_accuracy`
			- `accuracy_drop`
			- `attack_success_rate`
			- `avg_l2_perturbation`
			- `avg_linf_perturbation`
- New file: `run_stage_b_attack.py`
	- Dummy mode (parallel work):
		- `python run_stage_b_attack.py`
	- Real-data mode (after Person 1):
		- `python run_stage_b_attack.py --real-data --csv-path KDDTrain_with_headers.csv --model-type rf`

### Stage C (Person 4 - Anomaly Detection)

- File: `stage_c_anomaly.py`
	- Trains an IsolationForest on "normal" baseline traffic.
 	- Generates SHAP summary plots for model explainability (XAI).

- File: `stage_c_test.py`
  	- Validates the safety net against Stage B adversarial samples.
  	- Cleans prefixed column names (e.g., num__) to align adversarial data with the model's expected 42 numeric features.

- Model Artifact: `stage_c_isolation_forest.pkl`
  	- The trained unsupervised model ready for integration.

- Visualization: `stage_c_shap_summary.png`
  	- Illustrates the key features (e.g., src_bytes, count) that trigger the anomaly detection safety net.

## Validation Performed

- Stage B dummy flow executed successfully in venv:
	- Command: `python run_stage_b_attack.py`
	- Output included all expected robustness metrics.
- Stage B real-data path is wired to Stage A preprocessing/training and is ready once the dataset path is available.

- Stage C Verification executed successfully:
    - Command: python stage_c_test.py
    - Adversarial Detection Rate: 99.65%
    - Result: Flagged 25,108 out of 25,195 adversarial samples as anomalies (-1).

## Current Workspace Files (relevant)

- `KDDTrain_with_headers.csv`
- `stage_a_baseline.py`
- `attack_simulation.py`
- `run_stage_b_attack.py`
- `README.md`
- `requirements.txt`
- `context.md`
- `handover.md`
- `stage_c_anomaly.py`
- `stage_c_test.py`
- `stage_c_isolation_forest.pkl`
- `stage_c_shap_summary.png`
- `X_adversarial_samples.csv`

## Integration Contract For Teammates

### Person 1 (Data)

- Provide clean train/test features compatible with Stage A preprocessing output.
- Keep target column as `label` in raw CSV/DataFrame if using Stage A preprocessing directly.

### Person 4 (Anomaly)

- Keep anomaly output convention explicit for integration:
	- `-1` = anomaly
	- `1` = normal

### Person 5 (Integration)

- Current decision rule expected by team plan:
	- `final_decision = (clf_pred == 1) OR (anomaly_pred == -1)`
- Person 3 currently contributes robustness metrics (not final label votes).
- Recommended demo fields from Person 3:
	- clean vs adversarial accuracy
	- attack success rate
	- perturbation norms

## Next Recommended Steps

- Commit and push Stage B files with README/handover/context updates.
- Person 5 can immediately consume printed Stage B metrics for plotting.
- Optional follow-up: add CSV export for Stage B metrics to simplify dashboard/demo wiring.

