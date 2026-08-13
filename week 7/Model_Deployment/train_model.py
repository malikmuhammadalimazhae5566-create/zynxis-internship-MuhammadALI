from pathlib import Path
import pickle

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target_names[iris.target])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y_encoded)

with (MODEL_DIR / "trained_model.pkl").open("wb") as f:
    pickle.dump(model, f)
with (MODEL_DIR / "scaler.pkl").open("wb") as f:
    pickle.dump(scaler, f)
with (MODEL_DIR / "encoder.pkl").open("wb") as f:
    pickle.dump(label_encoder, f)

sample_df = X.head(10).copy()
sample_df["species"] = y.head(10).values
sample_df.to_csv(DATA_DIR / "sample_data.csv", index=False)

print("Model training completed.")
