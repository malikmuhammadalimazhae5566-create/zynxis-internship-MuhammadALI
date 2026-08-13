import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def create_intern_performance_dataset(n_samples=1000, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.55, 0.45],
        class_sep=1.2,
        random_state=random_state,
    )

    feature_names = [
        "coding_score",
        "communication_score",
        "problem_solving_score",
        "attendance_rate",
        "project_completion_rate",
        "peer_review_score",
        "learning_speed",
        "deadline_met_rate",
    ]

    df = pd.DataFrame(X, columns=feature_names)
    df["coding_score"] = (df["coding_score"] - df["coding_score"].min()) / (
        df["coding_score"].max() - df["coding_score"].min()
    ) * 100
    df["communication_score"] = (df["communication_score"] - df["communication_score"].min()) / (
        df["communication_score"].max() - df["communication_score"].min()
    ) * 100
    df["problem_solving_score"] = (df["problem_solving_score"] - df["problem_solving_score"].min()) / (
        df["problem_solving_score"].max() - df["problem_solving_score"].min()
    ) * 100
    df["attendance_rate"] = np.clip(df["attendance_rate"] * 10 + 85, 0, 100)
    df["project_completion_rate"] = np.clip(df["project_completion_rate"] * 10 + 85, 0, 100)
    df["peer_review_score"] = np.clip(df["peer_review_score"] * 10 + 80, 0, 100)
    df["learning_speed"] = np.clip(df["learning_speed"] * 10 + 75, 0, 100)
    df["deadline_met_rate"] = np.clip(df["deadline_met_rate"] * 10 + 80, 0, 100)

    df["performance_label"] = y
    df["performance_label"] = df["performance_label"].map({0: "needs_improvement", 1: "meets_expectations"})
    return df


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="meets_expectations"),
        "recall": recall_score(y_test, y_pred, pos_label="meets_expectations"),
        "f1": f1_score(y_test, y_pred, pos_label="meets_expectations"),
        "report": classification_report(y_test, y_pred, target_names=["needs_improvement", "meets_expectations"]),
    }


def main():
    df = create_intern_performance_dataset()
    X = df.drop(columns=["performance_label"])
    y = df["performance_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []
    best_model_name = None
    best_f1 = 0.0

    for name, model in models.items():
        model_result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        results.append(
            {
                "model": name,
                "accuracy": model_result["accuracy"],
                "precision": model_result["precision"],
                "recall": model_result["recall"],
                "f1": model_result["f1"],
            }
        )

        print(f"\n=== {name} ===")
        print(model_result["report"])

        if model_result["f1"] > best_f1:
            best_f1 = model_result["f1"]
            best_model_name = name

    summary_df = pd.DataFrame(results).set_index("model")
    print("\n=== Model comparison ===")
    print(summary_df.round(4))

    print(f"\nBest model: {best_model_name} with F1 score = {best_f1:.4f}")
    print(
        "\nExplanation: The best model is selected by the highest F1 score because it balances "
        "precision and recall for the positive class (interns who meet expectations)."
    )


if __name__ == "__main__":
    main()
