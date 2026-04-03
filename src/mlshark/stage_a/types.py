from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_matrix: list[list[int]]
    labels: list[Any]


@dataclass(frozen=True)
class TrainArtifacts:
    model_path: Path
    metrics_path: Path
    labels_path: Path
