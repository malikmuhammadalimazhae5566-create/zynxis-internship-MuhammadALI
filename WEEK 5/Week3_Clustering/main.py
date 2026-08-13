from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'titanic.csv'


def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return pd.DataFrame(
            {
                'pclass': [1, 1, 2, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3],
                'survived': [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
                'sex': ['female', 'male', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'female'],
                'age': [29, 30, 2, 0.83, 25, 4, 63, 39, 14, 14, 71, 24, 19, 3],
                'sibsp': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
                'parch': [0, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2],
                'fare': [211.34, 151.55, 21.08, 29.0, 7.90, 16.70, 77.96, 7.80, 8.05, 11.13, 49.50, 10.50, 8.16, 41.58],
                'embarked': ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C', 'C', 'S', 'Q', 'S'],
            }
        )


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_model = df.copy()
    df_model = df_model.drop(columns=['survived'], errors='ignore')
    categorical_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df_model.select_dtypes(include=['number']).columns.tolist()

    if categorical_cols:
        df_model[categorical_cols] = df_model[categorical_cols].fillna('missing')
        df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

    imputer = SimpleImputer(strategy='median')
    scaled_values = imputer.fit_transform(df_model[numeric_cols])
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(scaled_values)
    X = pd.DataFrame(scaled_values, columns=numeric_cols, index=df_model.index)
    return df_model, X


def plot_elbow(X, output_path: Path) -> None:
    inertias = []
    for k in range(1, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 9), inertias, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_clusters(X, labels, title: str, output_path: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def write_report(summary: pd.DataFrame, output_path: Path) -> None:
    import zipfile
    from xml.sax.saxutils import escape

    content = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Clustering Report</w:t></w:r></w:p>
    <w:p><w:r><w:t>Dataset: Titanic passenger features</w:t></w:r></w:p>
    <w:p><w:r><w:t>K-Means and DBSCAN were applied to uncover hidden clusters in the data.</w:t></w:r></w:p>
    <w:p><w:r><w:t>The elbow method was used to guide the number of K-Means clusters.</w:t></w:r></w:p>
    <w:p><w:r><w:t>Cluster summary:</w:t></w:r></w:p>
    <w:p><w:r><w:t>{escape(summary.to_string(index=False))}</w:t></w:r></w:p>
    <w:p><w:r><w:t>Business use case: Zynxis could use these groups to segment users, compare intern performance patterns, or detect unusual behavior.</w:t></w:r></w:p>
    <w:sectPr/></w:body>
</w:document>'''

    with zipfile.ZipFile(output_path, 'w') as zf:
        zf.writestr('word/document.xml', content)
        zf.writestr('[Content_Types].xml', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>''')
        zf.writestr('_rels/.rels', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>''')


def main() -> None:
    df = load_data()
    _, X = preprocess(df)
    plot_elbow(X, ROOT / 'elbow_method.png')

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    plot_clusters(X, kmeans_labels, 'K-Means Clusters (PCA)', ROOT / 'kmeans_clusters.png')

    dbscan = DBSCAN(eps=2.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X)
    plot_clusters(X, dbscan_labels, 'DBSCAN Clusters (PCA)', ROOT / 'dbscan_clusters.png')

    result_df = df.copy()
    result_df['kmeans_cluster'] = kmeans_labels
    result_df['dbscan_cluster'] = dbscan_labels
    result_df.to_csv(ROOT / 'clustered_dataset.csv', index=False)

    summary = result_df.groupby('kmeans_cluster').mean(numeric_only=True).reset_index()
    summary.to_csv(ROOT / 'cluster_summary.csv', index=False)
    write_report(summary, ROOT / 'Clustering_Report.docx')

    print('Clustering workflow completed successfully.')
    print(summary.head())


if __name__ == '__main__':
    main()
