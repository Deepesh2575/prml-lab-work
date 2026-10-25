"""Segment mall customers using K-Means and hierarchical clustering."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

folder = Path(__file__).parent
data = pd.read_csv(folder / "Mall_Customers.csv")
features = data[["Annual Income (k$)", "Spending Score (1-100)"]].fillna(
	data[["Annual Income (k$)", "Spending Score (1-100)"]].median()
)
X = StandardScaler().fit_transform(features)

ks = range(2, 11)
scores, labels = {"K-Means": [], "Hierarchical": []}, {}
for k in ks:
	for name, model in {
		"K-Means": KMeans(n_clusters=k, random_state=42, n_init=10),
		"Hierarchical": AgglomerativeClustering(n_clusters=k),
	}.items():
		cluster = model.fit_predict(X)
		scores[name].append(silhouette_score(X, cluster))
		labels[name, k] = cluster

best_score, method, best_k = max(
	(score, name, count)
	for name, values in scores.items()
	for count, score in zip(ks, values)
)
print("Silhouette scores:")
for count, kmeans_score, hierarchical_score in zip(
	ks, scores["K-Means"], scores["Hierarchical"]
):
	print(
		f"{count} clusters: K-Means {kmeans_score:.3f}, "
		f"Hierarchical {hierarchical_score:.3f}"
	)
print(f"\nBest: {method}, {best_k} clusters (score {best_score:.3f}).")

for name in scores:
	column = "KMeans_Cluster" if name == "K-Means" else "Hierarchical_Cluster"
	data[column] = labels[name, best_k]
data.to_csv(folder / "customer_segments.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for name in scores:
	axes[0].plot(ks, scores[name], marker="o", label=name)
axes[0].set(xlabel="Clusters", ylabel="Silhouette score", title="Model comparison")
axes[0].legend()
for axis, name in zip(axes[1:], scores):
	axis.scatter(
		features.iloc[:, 0],
		features.iloc[:, 1],
		c=labels[name, best_k],
		cmap="tab10",
	)
	axis.set(
		xlabel="Annual income (k$)",
		ylabel="Spending score",
		title=f"{name} (k={best_k})",
	)
fig.tight_layout()
fig.savefig(folder / "customer_segments.png", dpi=150)
plt.show()
print("Saved customer_segments.csv and customer_segments.png")
