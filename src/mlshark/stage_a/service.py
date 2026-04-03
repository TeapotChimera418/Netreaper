from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .config import StageAConfig
from .data import build_splits, load_dataset
from .pipeline import build_training_pipeline
from .types import EvaluationReport, TrainArtifacts


class StageAClassifier:
    def __init__(self, config: StageAConfig) -> None:
        self.config = config
        self.config.validate()
        self.pipeline: Pipeline | None = None
        self.labels: list[Any] = []

    def train(self) -> tuple[TrainArtifacts, EvaluationReport]:
        x, y = load_dataset(str(self.config.data_path), self.config.target_column)
        splits = build_splits(
            x=x,
            y=y,
            test_size=self.config.test_size,
            validation_size=self.config.validation_size,
            random_state=self.config.random_state,
        )

        self.pipeline = build_training_pipeline(self.config, splits.x_train)
        self.pipeline.fit(splits.x_train, splits.y_train)

        report = self.evaluate(splits.x_test, splits.y_test)
        artifacts = self.save_artifacts(report)
        return artifacts, report

    def evaluate(self, x: pd.DataFrame, y: pd.Series) -> EvaluationReport:
        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been trained or loaded")

        predictions = self.pipeline.predict(x)
        self.labels = sorted(pd.Series(y).unique().tolist())

        report = EvaluationReport(
            accuracy=float(accuracy_score(y, predictions)),
            precision_weighted=float(
                precision_score(y, predictions, average="weighted", zero_division=0)
            ),
            recall_weighted=float(
                recall_score(y, predictions, average="weighted", zero_division=0)
            ),
            f1_weighted=float(
                f1_score(y, predictions, average="weighted", zero_division=0)
            ),
            confusion_matrix=confusion_matrix(
                y,
                predictions,
                labels=self.labels,
            ).tolist(),
            labels=self.labels,
        )
        return report

    def predict(self, records: pd.DataFrame) -> list[Any]:
        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been trained or loaded")
        return self.pipeline.predict(records).tolist()

    def predict_proba(self, records: pd.DataFrame) -> list[list[float]]:
        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been trained or loaded")

        classifier = self.pipeline.named_steps["classifier"]
        if not hasattr(classifier, "predict_proba"):
            raise RuntimeError("Model does not support probability predictions")

        probabilities = self.pipeline.predict_proba(records)
        return probabilities.tolist()

    def save_artifacts(self, report: EvaluationReport) -> TrainArtifacts:
        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been trained or loaded")

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "classifier.joblib"
        metrics_path = output_dir / "metrics.json"
        labels_path = output_dir / "labels.json"

        joblib.dump(self.pipeline, model_path)

        metrics_payload = {
            "accuracy": report.accuracy,
            "precision_weighted": report.precision_weighted,
            "recall_weighted": report.recall_weighted,
            "f1_weighted": report.f1_weighted,
            "confusion_matrix": report.confusion_matrix,
            "labels": report.labels,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        labels_path.write_text(json.dumps(report.labels, indent=2), encoding="utf-8")

        return TrainArtifacts(
            model_path=model_path,
            metrics_path=metrics_path,
            labels_path=labels_path,
        )

    def load_model(self, model_path: str | Path) -> None:
        self.pipeline = joblib.load(model_path)
