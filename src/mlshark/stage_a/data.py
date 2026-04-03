from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplits:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_dataset(data_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    y = df[target_column]
    x = df.drop(columns=[target_column])
    return x, y


def build_splits(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> DataSplits:
    stratify_target = y if y.nunique() > 1 else None

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    if validation_size == 0:
        return DataSplits(
            x_train=x_train_val,
            x_val=x_train_val.iloc[0:0],
            x_test=x_test,
            y_train=y_train_val,
            y_val=y_train_val.iloc[0:0],
            y_test=y_test,
        )

    adjusted_val_size = validation_size / (1.0 - test_size)
    stratify_train_val = y_train_val if y_train_val.nunique() > 1 else None

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    return DataSplits(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
