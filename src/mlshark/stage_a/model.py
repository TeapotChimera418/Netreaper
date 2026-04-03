from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_classifier(model_name: str, random_state: int, **params: object):
    name = model_name.lower().strip()

    if name == "random_forest":
        defaults = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": random_state,
        }
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    if name == "logistic_regression":
        defaults = {
            "max_iter": 2000,
            "n_jobs": -1,
            "random_state": random_state,
        }
        defaults.update(params)
        return LogisticRegression(**defaults)

    raise ValueError(
        "Unsupported model_name. Use 'random_forest' or 'logistic_regression'."
    )
