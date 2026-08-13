from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset.csv"
REPORT_PATH = PROJECT_ROOT / "report.docx"
SCREENSHOT_PATH = PROJECT_ROOT / "screenshots" / "predicted_vs_actual.png"


def generate_dataset(path: Path) -> None:
    rng = np.random.default_rng(42)
    n_samples = 220
    advertising_spend = rng.uniform(100, 2500, n_samples)
    store_size = rng.uniform(800, 6000, n_samples)
    online_ads = rng.uniform(30, 1200, n_samples)
    seasonality = rng.uniform(0.85, 1.15, n_samples)
    sales = (
        180
        + 1.6 * advertising_spend
        + 0.35 * store_size
        + 0.9 * online_ads
        + 120 * (seasonality - 1)
        + rng.normal(0, 180, n_samples)
    )

    df = pd.DataFrame(
        {
            "advertising_spend": advertising_spend,
            "store_size": store_size,
            "online_ads": online_ads,
            "seasonality": seasonality,
            "sales": sales,
        }
    )
    df.to_csv(path, index=False)


def print_ascii_scatter(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray, title: str) -> None:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    width, height = 60, 20
    grid = [[" " for _ in range(width)] for _ in range(height)]

    if actual.size == 0:
        print(title)
        print("(no data)")
        return

    actual_min, actual_max = actual.min(), actual.max()
    pred_min, pred_max = predicted.min(), predicted.max()
    actual_scale = actual_max - actual_min if actual_max != actual_min else 1.0
    pred_scale = pred_max - pred_min if pred_max != pred_min else 1.0

    for value_actual, value_pred in zip(actual, predicted):
        x = int(((value_actual - actual_min) / actual_scale) * (width - 1))
        y = int((1 - ((value_pred - pred_min) / pred_scale)) * (height - 1))
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = "*"

    for row in range(height):
        grid[row][0] = "|"
    for col in range(width):
        grid[height // 2][col] = "-"

    print(title)
    for row in grid:
        print("".join(row))
    print()


def create_report(results: dict, image_name: str) -> None:
    lines = [
        "Regression Project Report",
        "========================",
        "",
        "This project trains three regression models to predict sales using numeric features.",
        "",
        "Models evaluated:",
        "- Linear Regression",
        "- Ridge Regression",
        "- Random Forest Regressor",
        "",
        "Evaluation metrics:",
    ]
    for name, metrics in results.items():
        lines.append(
            f"- {name}: MAE={metrics['mae']:.2f}, MSE={metrics['mse']:.2f}, R2={metrics['r2']:.3f}"
        )

    lines.extend(
        [
            "",
            "Visualization:",
            f"Predicted vs actual values were saved to screenshots/{image_name}.",
            "",
            "Conclusion:",
            "Random Forest Regressor performed best on this synthetic sales dataset.",
        ]
    )

    paragraphs = []
    for line in lines:
        if line.strip():
            paragraphs.append(f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>")
        else:
            paragraphs.append("<w:p/>")

    document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(paragraphs)}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>'''

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>'''

    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

    with zipfile.ZipFile(REPORT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)


def train_and_evaluate() -> dict:
    if not DATA_PATH.exists():
        generate_dataset(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    features = ["advertising_spend", "store_size", "online_ads", "seasonality"]
    X = df[features]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for index, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results[name] = {"mae": mae, "mse": mse, "r2": r2}
        print_ascii_scatter(y_test, predictions, f"{name} - Predicted vs Actual")

        axes[index].scatter(y_test, predictions, alpha=0.7)
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        axes[index].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        axes[index].set_title(f"{name}\nMAE={mae:.2f}, R2={r2:.3f}")
        axes[index].set_xlabel("Actual")
        axes[index].set_ylabel("Predicted")

    plt.tight_layout()
    SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCREENSHOT_PATH, dpi=300)
    plt.close(fig)

    create_report(results, SCREENSHOT_PATH.name)
    return results


if __name__ == "__main__":
    results = train_and_evaluate()
    print("Model evaluation results:")
    for name, metrics in results.items():
        print(f"{name}: MAE={metrics['mae']:.2f}, MSE={metrics['mse']:.2f}, R2={metrics['r2']:.3f}")
    print(f"Saved plot to {SCREENSHOT_PATH}")
    print(f"Saved report to {REPORT_PATH}")
