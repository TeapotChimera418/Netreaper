from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from attack_simulation import StageBAttackSimulator
from stage_a_baseline import preprocess_data, train_model

# This function creates dummy data at random ranges
def _build_dummy_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.random((1000, 20))
    y_train = rng.integers(0, 2, 1000)
    x_test = rng.random((200, 20))
    y_test = rng.integers(0, 2, 200)
    return x_train, y_train, x_test, y_test

# This runs the model on the dummy data we generated
def run_dummy_mode(noise_std: float, model_trees: int) -> None:
    x_train, y_train, x_test, y_test = _build_dummy_data()
    model = RandomForestClassifier(n_estimators=model_trees, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    simulator = StageBAttackSimulator(noise_std=noise_std, clip_min=0.0, clip_max=1.0)
    result = simulator.evaluate(model, x_test, y_test)

    print("\nStage B Attack Simulation (Dummy Data)")
    print(f"clean_accuracy       : {result.clean_accuracy:.4f}")
    print(f"adversarial_accuracy : {result.adversarial_accuracy:.4f}")
    print(f"accuracy_drop        : {result.accuracy_drop:.4f}")
    print(f"attack_success_rate  : {result.attack_success_rate:.4f}")
    print(f"avg_l2_perturbation  : {result.avg_l2_perturbation:.4f}")
    print(f"avg_linf_perturbation: {result.avg_linf_perturbation:.4f}")

# This runs the model on the actual _adversial csv dataset
def run_real_data_mode(csv_path: Path, noise_std: float, model_type: str) -> None:
    if not csv_path.exists():
        raise SystemExit(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    x_train, x_test, y_train, y_test = preprocess_data(df)
    model = train_model(x_train, y_train, model_type=model_type)

    simulator = StageBAttackSimulator(noise_std=noise_std)
    result = simulator.evaluate(model, x_test.to_numpy(dtype=float), y_test.to_numpy())

    print("\nStage B Attack Simulation (Real NSL-KDD Split)")
    print(f"model_type           : {model_type}")
    print(f"clean_accuracy       : {result.clean_accuracy:.4f}")
    print(f"adversarial_accuracy : {result.adversarial_accuracy:.4f}")
    print(f"accuracy_drop        : {result.accuracy_drop:.4f}")
    print(f"attack_success_rate  : {result.attack_success_rate:.4f}")
    print(f"avg_l2_perturbation  : {result.avg_l2_perturbation:.4f}")
    print(f"avg_linf_perturbation: {result.avg_linf_perturbation:.4f}")

# It lets you customise how the script runs in the command line without changing the code itself [Ex: python run_script.py [customised arguments]]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B adversarial attack simulation runner")
    parser.add_argument("--real-data", action="store_true", help="Use NSL-KDD CSV + Stage A preprocess")
    parser.add_argument("--csv-path", type=Path, default=Path("KDDTrain_with_headers.csv"))
    parser.add_argument("--model-type", choices=["rf", "xgb"], default="rf")
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--dummy-trees", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.real_data:
        run_real_data_mode(args.csv_path, args.noise_std, args.model_type)
    else:
        run_dummy_mode(args.noise_std, args.dummy_trees)


if __name__ == "__main__":
    main()
