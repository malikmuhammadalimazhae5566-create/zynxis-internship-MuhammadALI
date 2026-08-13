from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "interns.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
MODEL_META_PATH = ROOT / "models" / "model_metadata.json"


def create_synthetic_dataset(path: Path) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(42)
    domains = ["AI/ML", "Web Development", "Data Science", "Product", "UX Research", "Marketing"]
    education = ["Bachelor's", "Master's", "Diploma"]
    labels = ["Excellent", "Good", "Needs Improvement"]

    rows = []
    for i in range(500):
        domain = rng.choice(domains)
        edu = rng.choice(education)
        coding = int(rng.integers(40, 100))
        communication = int(rng.integers(45, 99))
        teamwork = int(rng.integers(50, 100))
        discipline = int(rng.integers(55, 100))
        attendance = int(rng.integers(60, 100))
        hours = int(rng.integers(30, 90))
        project = int(rng.integers(40, 100))
        prior_exp = int(rng.integers(0, 2))
        mentor = int(rng.integers(50, 100))

        score = (
            coding * 0.25
            + communication * 0.18
            + teamwork * 0.17
            + discipline * 0.20
            + attendance * 0.12
            + project * 0.08
        )

        if score >= 820:
            label = "Excellent"
        elif score >= 700:
            label = "Good"
        else:
            label = "Needs Improvement"

        rows.append(
            {
                "domain": domain,
                "education_level": edu,
                "coding_score": coding,
                "communication_score": communication,
                "teamwork_score": teamwork,
                "discipline_score": discipline,
                "attendance_pct": attendance,
                "hours_logged": hours,
                "project_quality": project,
                "prior_experience": prior_exp,
                "mentor_rating": mentor,
                "performance_label": label,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def train_and_save_model() -> dict:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        create_synthetic_dataset(DATA_PATH)

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

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    with MODEL_META_PATH.open("w", encoding="utf-8") as f:
        json.dump({"model_type": "RandomForestClassifier", **metrics}, f, indent=2)

    print(f"Saved trained model to {MODEL_PATH}")
    print(f"Macro F1 score: {metrics['f1_macro']:.4f}")
    return metrics


if __name__ == "__main__":
    train_and_save_model()
