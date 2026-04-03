from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from .config import StageAConfig
from .model import build_classifier
from .preprocessing import build_preprocessor


def build_training_pipeline(config: StageAConfig, x_train: pd.DataFrame) -> Pipeline:
    numeric_columns = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        c for c in x_train.columns.tolist() if c not in set(numeric_columns)
    ]

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    classifier = build_classifier(
        model_name=config.model_name,
        random_state=config.random_state,
        **config.model_params,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
