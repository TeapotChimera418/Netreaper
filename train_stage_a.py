from __future__ import annotations

import argparse
from pathlib import Path

from mlshark.stage_a import StageAClassifier, StageAConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage A classifier for MLShark intrusion detection."
    )
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Classifier type",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="artifacts/stage_a",
        help="Directory for trained model and metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = StageAConfig(
        data_path=Path(args.data),
        target_column=args.target,
        model_name=args.model,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
        output_dir=Path(args.output_dir),
    )

    stage_a = StageAClassifier(config)
    artifacts, report = stage_a.train()

    print("Training complete")
    print(f"Model path: {artifacts.model_path}")
    print(f"Metrics path: {artifacts.metrics_path}")
    print(f"Accuracy: {report.accuracy:.4f}")
    print(f"F1 (weighted): {report.f1_weighted:.4f}")


if __name__ == "__main__":
    main()
