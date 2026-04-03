from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocess input data and return stratified 80/20 train-test split."""
    if "label" not in df.columns:
        raise ValueError("Expected target column 'label' in input DataFrame")

    data = df.copy()

    # Binary target for Stage A baseline: normal -> 0, all attacks -> 1.
    label_as_text = data["label"].astype(str).str.strip().str.lower()
    y = (label_as_text != "normal").astype(int)
    x = data.drop(columns=["label"])

    categorical_cols = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in x.columns if col not in categorical_cols]

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # We impute missing values instead of dropping rows to preserve scarce attack examples.
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    x_train_arr = preprocessor.fit_transform(x_train_raw)
    x_test_arr = preprocessor.transform(x_test_raw)

    feature_names = preprocessor.get_feature_names_out()
    x_train = pd.DataFrame(x_train_arr, columns=feature_names, index=x_train_raw.index)
    x_test = pd.DataFrame(x_test_arr, columns=feature_names, index=x_test_raw.index)

    # Guardrail: model input must be fully numeric.
    if not all(np.issubdtype(dtype, np.number) for dtype in x_train.dtypes):
        raise TypeError("Preprocessing failed: non-numeric columns remain in training features")

    return x_train, x_test, y_train, y_test


def train_model(x_train: pd.DataFrame, y_train: pd.Series, model_type: str) -> Any:
    """Train and return a RandomForest or XGBoost classifier."""
    model_key = model_type.strip().lower()

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    if model_key == "rf":
        model = RandomForestClassifier(
            n_estimators=300,  # robust baseline size for tabular intrusion data
            min_samples_leaf=2,  # light regularization to reduce noisy splits
            class_weight="balanced",  # compensates normal/attack class imbalance
            n_jobs=-1,
            random_state=42,
        )
    elif model_key == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for model_type='xgb'. Install with: pip install xgboost"
            ) from exc

        model = XGBClassifier(
            n_estimators=350,  # strong baseline without heavy tuning overhead
            max_depth=6,  # enough capacity while limiting overfitting
            learning_rate=0.05,  # stable gradient boosting updates
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,  # class imbalance-aware boosting
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError("model_type must be either 'rf' or 'xgb'")

    model.fit(x_train, y_train)
    return model


def _predict_scores(model: Any, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return class predictions and positive-class scores/probabilities."""
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_score = model.decision_function(x_test)
        y_score = (raw_score - raw_score.min()) / max(raw_score.max() - raw_score.min(), 1e-9)
    else:
        y_score = y_pred.astype(float)

    return y_pred, y_score


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute required binary classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }


def evaluate_model(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Print evaluation table for one model and return predictions and scores."""
    y_pred, y_score = _predict_scores(model, x_test)
    metrics = _compute_metrics(y_test, y_pred, y_score)

    metric_df = pd.DataFrame([metrics], index=[model_name])
    print(f"\n{model_name} - Evaluation Metrics")
    print(metric_df.to_string(float_format=lambda v: f"{v:.4f}"))

    return y_pred, y_score


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    """Plot confusion matrix heatmap for a model."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (0)", "Attack (1)"],
        yticklabels=["Normal (0)", "Attack (1)"],
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str,
    top_n: int | None = None,
) -> None:
    """Plot top-N feature importance for supported models."""
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef[0] if coef.ndim > 1 else coef)
    else:
        print(f"{model_name}: feature importance is unavailable for this estimator")
        return

    if top_n is None:
        top_n = min(20, len(feature_names))
    top_n = max(1, min(top_n, len(feature_names)))

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    top_df = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature"], top_df["importance"], color="teal")
    plt.title(f"{model_name} - Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: pd.Series, y_score: np.ndarray, model_name: str) -> None:
    """Plot ROC curve with AUC for a model."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}", color="darkorange", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def print_comparison_table(metrics_by_model: dict[str, dict[str, float]]) -> None:
    """Print side-by-side summary for evaluated models."""
    comparison_df = pd.DataFrame(metrics_by_model).T[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    print("\nModel Comparison Summary")
    print(comparison_df.to_string(float_format=lambda v: f"{v:.4f}"))


def main(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Run full Stage A baseline pipeline and return predictions for both models."""
    x_train, x_test, y_train, y_test = preprocess_data(df)
    feature_names = x_train.columns.tolist()

    metrics_summary: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}

    for model_key, model_name in (("rf", "Random Forest"), ("xgb", "XGBoost")):
        model = train_model(x_train, y_train, model_key)
        y_pred, y_score = evaluate_model(model, x_test, y_test, model_name)

        metrics_summary[model_name] = _compute_metrics(y_test, y_pred, y_score)
        predictions[model_name] = y_pred

        plot_confusion_matrix(y_test, y_pred, model_name)
        plot_feature_importance(model, feature_names, model_name)
        plot_roc_curve(y_test, y_score, model_name)

    print_comparison_table(metrics_summary)
    return predictions


if __name__ == "__main__":
    default_path = Path("KDDTrain_with_headers.csv")
    if not default_path.exists():
        raise SystemExit(
            "Provide a pandas DataFrame to main(df), or place KDDTrain_with_headers.csv in the project root."
        )

    dataframe = pd.read_csv(default_path)
    _ = main(dataframe)
