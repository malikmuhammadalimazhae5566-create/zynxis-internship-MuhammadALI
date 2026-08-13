import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"

with (MODEL_DIR / "trained_model.pkl").open("rb") as f:
    model = pickle.load(f)
with (MODEL_DIR / "scaler.pkl").open("rb") as f:
    scaler = pickle.load(f)
with (MODEL_DIR / "encoder.pkl").open("rb") as f:
    encoder = pickle.load(f)


def predict_species(features):
    scaled_features = scaler.transform(features)
    probabilities = model.predict_proba(scaled_features)[0]
    class_index = int(np.argmax(probabilities))
    predicted_label = encoder.inverse_transform([class_index])[0]

    class_probabilities = {
        label: round(float(prob), 3) for label, prob in zip(encoder.classes_, probabilities)
    }
    return predicted_label, class_probabilities
