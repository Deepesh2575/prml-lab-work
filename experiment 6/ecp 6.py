"""Use PCA to reduce Kaggle's Iris dataset to two dimensions."""

from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data_dir = Path("data/kaggle_iris")
data_dir.mkdir(parents=True, exist_ok=True)
csv_path = data_dir / "Iris.csv"

if not csv_path.exists():
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", "uciml/iris", "-p", str(data_dir), "--unzip"],
        check=True,
    )

data = pd.read_csv(csv_path)
labels = data["Species"]
features = data.drop(columns=["Id", "Species"])

if features.isnull().any().any():
    raise ValueError("Dataset contains missing feature values.")

scaled = StandardScaler().fit_transform(features)
pca = PCA(n_components=2)
scores = pca.fit_transform(scaled)

result = pd.DataFrame(scores, columns=["PC1", "PC2"])
result["Species"] = labels
result.to_csv("pca_output.csv", index=False)

for species in labels.unique():
    rows = labels == species
    plt.scatter(scores[rows, 0], scores[rows, 1], label=species)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.title("PCA of Kaggle Iris Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("pca_plot.png")

print(f"Variance retained: {pca.explained_variance_ratio_.sum():.1%}")
print("Saved results to pca_output.csv and pca_plot.png")
