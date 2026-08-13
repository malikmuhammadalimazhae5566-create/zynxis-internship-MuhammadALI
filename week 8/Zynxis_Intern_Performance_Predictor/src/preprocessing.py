from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "domain",
    "education_level",
    "coding_score",
    "communication_score",
    "teamwork_score",
    "discipline_score",
    "attendance_pct",
    "hours_logged",
    "project_quality",
    "prior_experience",
    "mentor_rating",
]

TARGET_COLUMN = "performance_label"


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [
        col for col in X.columns if X[col].dtype == "object" or X[col].nunique() <= 10
    ]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")
