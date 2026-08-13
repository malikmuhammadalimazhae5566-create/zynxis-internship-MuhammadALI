from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "interns.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"


def evaluate_model() -> dict:
    df = load_dataset(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(json.dumps({"f1_macro": round(metrics["f1_macro"], 4)}, indent=2))
    print(classification_report(y_test, y_pred))
    return metrics


if __name__ == "__main__":
    evaluate_model()
