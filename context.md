This is a hackathon project called "MLShark – Adversarial Resilient Intrusion Detection System".

Current implementation status:
- Stage A: Classification baseline implemented in `stage_a_baseline.py`
- Stage B: Adversarial attack simulation implemented for Person 3 in:
	- `attack_simulation.py`
	- `run_stage_b_attack.py`
- Stage C: Anomaly detection (Isolation Forest) pending

Stage A details:
- Dataset: NSL-KDD (`KDDTrain_with_headers.csv` in repo root)
- Target column: `label`
- Binary mapping: `normal` -> 0, all attack labels -> 1

Stage B details:
- Dummy mode available for parallel development:
	- `python run_stage_b_attack.py`
- Real-data mode (after data pipeline is finalized):
	- `python run_stage_b_attack.py --real-data --csv-path KDDTrain_with_headers.csv --model-type rf`
- Metrics emitted:
	- clean/adversarial accuracy
	- accuracy drop
	- attack success rate
	- average L2 and L-inf perturbation

Integration expectations:
- Person 3 supplies robustness metrics and adversarial simulation behavior.
- Person 4 should keep anomaly prediction convention explicit (`-1` anomaly, `1` normal).
- Person 5 combines outputs with final decision logic:
	- `(clf_pred == 1) OR (anomaly_pred == -1)`

Code should remain clean, modular, and easy to plug across stages.