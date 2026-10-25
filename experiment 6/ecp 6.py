"""Dimensionality reduction for CSV and image datasets using PCA.

Usage:
	python "ecp 6.py" dataset.csv --components 2 --label-column target
	python "ecp 6.py" image_folder --components 20 --image-size 32

Install dependencies with: pip install numpy pandas scikit-learn pillow
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_csv(path, label_column=None):
	data = pd.read_csv(path)
	labels = None
	if label_column:
		if label_column not in data.columns:
			raise ValueError(f"Label column {label_column!r} was not found.")
		labels = data.pop(label_column)
	features = data.select_dtypes(include=[np.number])
	if features.empty:
		raise ValueError("The CSV must contain at least one numeric feature column.")
	if features.isna().any().any():
		raise ValueError("Numeric feature columns contain missing values; clean them before PCA.")
	return features.to_numpy(dtype=float), labels


def load_images(directory, image_size):
	try:
		from PIL import Image
	except ImportError as exc:
		raise ImportError("Image input requires Pillow: pip install pillow") from exc

	root = Path(directory)
	paths = sorted(path for path in root.rglob("*")
				   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
	if not paths:
		raise ValueError(f"No supported image files found in {root}.")

	samples, labels = [], []
	for path in paths:
		with Image.open(path) as image:
			image = image.convert("L").resize((image_size, image_size))
			samples.append(np.asarray(image, dtype=np.float32).ravel())
		# Class-named subdirectories are kept as labels when available.
		labels.append(path.parent.name if path.parent != root else "")
	return np.stack(samples), pd.Series(labels, name="label")


def main():
	parser = argparse.ArgumentParser(description="Apply PCA to a CSV or image dataset.")
	parser.add_argument("input", help="CSV file or directory containing images")
	parser.add_argument("-k", "--components", type=int, default=2,
						help="number of principal components (default: 2)")
	parser.add_argument("-o", "--output", default="pca_output.csv",
						help="output CSV path (default: pca_output.csv)")
	parser.add_argument("--label-column", help="CSV label column to preserve in the output")
	parser.add_argument("--image-size", type=int, default=32,
						help="resize images to this square side length (default: 32)")
	args = parser.parse_args()

	if args.components < 1:
		parser.error("--components must be at least 1")
	if args.image_size < 1:
		parser.error("--image-size must be at least 1")

	source = Path(args.input)
	if source.is_dir():
		values, labels = load_images(source, args.image_size)
	elif source.is_file() and source.suffix.lower() == ".csv":
		values, labels = load_csv(source, args.label_column)
	else:
		parser.error("input must be a CSV file or a directory of images")

	# Standardize features before PCA so different scales do not dominate.
	values = StandardScaler().fit_transform(values)
	component_count = min(args.components, values.shape[0], values.shape[1])
	if component_count < args.components:
		print(f"Using {component_count} components (limited by dataset dimensions).")

	pca = PCA(n_components=component_count)
	reduced = pca.fit_transform(values)
	result = pd.DataFrame(reduced, columns=[f"PC{i + 1}" for i in range(component_count)])
	if labels is not None:
		result.insert(0, labels.name or "label", labels.to_numpy())
	result.to_csv(args.output, index=False)

	variance = pca.explained_variance_ratio_.sum() * 100
	print(f"Saved {len(result)} samples with {component_count} components to {args.output}")
	print(f"Variance explained: {variance:.2f}%")


if __name__ == "__main__":
	main()
