import csv
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.model import SentimentModel
from src.preprocessing import preprocess_text, tokenize


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "feedback.csv"


def load_feedback(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row["label"].strip().lower()
            rows.append({
                "text": row["text"].strip(),
                "label": 1 if label == "positive" else 0,
            })
    return rows


def main():
    print("Loading feedback dataset...")
    rows = load_feedback(DATA_PATH)
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]

    processed_texts = [preprocess_text(text) for text in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    print(f"\nDataset loaded from: {DATA_PATH}")
    print(f"Total feedback rows: {len(rows)}")

    sample_text = texts[0]
    print("\nSample preprocessing:")
    print("Original text:", sample_text)
    print("Tokenized text:", tokenize(sample_text))
    print("Processed text:", preprocess_text(sample_text))

    model = SentimentModel()
    model.fit(X_train, y_train)

    X_train_vec = model.vectorizer.transform(X_train)
    print("\nVectorized training shape:", X_train_vec.shape)

    accuracy, report = model.evaluate(X_test, y_test)
    print("\nAccuracy:", round(accuracy, 3))
    print("Classification report:\n", report)

    sample_inputs = [
        "The internship was amazing and the mentors were very supportive.",
        "I felt lost because the guidance was poor and the tasks were confusing."
    ]

    processed_samples = [preprocess_text(text) for text in sample_inputs]
    predicted_labels = model.predict(processed_samples)
    sentiment_map = {1: "positive", 0: "negative"}
    print("\nSample predictions:")
    for text, label in zip(sample_inputs, predicted_labels):
        print(f"- {text} -> {sentiment_map[int(label)]}")


if __name__ == "__main__":
    main()
