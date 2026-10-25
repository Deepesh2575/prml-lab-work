"""Load, explore, and preprocess a tabular dataset for machine learning."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(filepath):
	"""Load a CSV dataset into a pandas DataFrame."""
	path = Path(filepath)
	if not path.is_file():
		raise FileNotFoundError(f"Dataset not found: {path}")
	return pd.read_csv(path)


def explore_dataset(data):
	"""Print useful information about the dataset and its missing values."""
	print("Dataset shape:", data.shape)
	print("\nFirst five rows:")
	print(data.head())
	print("\nColumn data types:")
	print(data.dtypes)
	print("\nSummary statistics:")
	print(data.describe(include="all").transpose())
	print("\nMissing values per column:")
	print(data.isna().sum())


def preprocess_and_split(data, target, test_size=0.2, random_state=42):
	"""Impute, encode, scale, then split data into train and test sets."""
	print("\n[3/6] Checking the target column and handling missing target values")
	if target not in data.columns:
		raise ValueError(f"Target column {target!r} is not present in the dataset.")

	# Rows without a target cannot be used for supervised training.
	missing_targets = data[target].isna().sum()
	print(f"Target column: {target!r}")
	print(f"Rows with a missing target to remove: {missing_targets}")
	data = data.dropna(subset=[target]).copy()
	X = data.drop(columns=[target])
	y = data[target]
	if X.empty:
		raise ValueError("The dataset must contain at least one feature column.")

	print("\n[4/6] Identifying numeric and categorical features")
	numeric_columns = X.select_dtypes(include="number").columns.tolist()
	categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
	print(f"Numeric features ({len(numeric_columns)}): {numeric_columns}")
	print(f"Categorical features ({len(categorical_columns)}): {categorical_columns}")
	print(
		"Missing numeric feature values:",
		int(X[numeric_columns].isna().sum().sum()),
	)
	print(
		"Missing categorical feature values:",
		int(X[categorical_columns].isna().sum().sum()),
	)

	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", OneHotEncoder(handle_unknown="ignore")),
		]
	)
	transformer = ColumnTransformer(
		transformers=[
			("numeric", numeric_pipeline, numeric_columns),
			("categorical", categorical_pipeline, categorical_columns),
		],
		remainder="drop",
	)

	# Fit preprocessing only on training data to prevent data leakage.
	print(f"\n[5/6] Splitting data into training and test sets ({1 - test_size:.0%}/{test_size:.0%})")
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state
	)
	print(f"Training rows: {len(X_train)}; test rows: {len(X_test)}")

	print("\n[6/6] Imputing, encoding, and scaling features")
	print("Numeric: fill missing values with the median, then standardize.")
	print("Categorical: fill missing values with the most frequent value, then one-hot encode.")
	print("Fitting preprocessing on training data only; applying it to train and test data.")
	X_train = transformer.fit_transform(X_train)
	X_test = transformer.transform(X_test)
	print(f"Processed feature count: {X_train.shape[1]}")
	return X_train, X_test, y_train, y_test, transformer


def main():
	parser = argparse.ArgumentParser(
		description="Explore and preprocess a CSV dataset for machine learning."
	)
	parser.add_argument("csv_file", help="Path to the input CSV file")
	parser.add_argument("--target", required=True, help="Name of the target column")
	parser.add_argument(
		"--test-size", type=float, default=0.2,
		help="Fraction of data reserved for testing (default: 0.2)",
	)
	parser.add_argument(
		"--random-state", type=int, default=42,
		help="Random seed for the train/test split (default: 42)",
	)
	args = parser.parse_args()

	if not 0 < args.test_size < 1:
		parser.error("--test-size must be greater than 0 and less than 1")

	print(f"[1/6] Loading CSV dataset: {args.csv_file}")
	data = load_dataset(args.csv_file)
	print(f"Loaded {len(data)} rows and {len(data.columns)} columns.")

	print("\n[2/6] Exploring the dataset")
	explore_dataset(data)
	X_train, X_test, y_train, y_test, _ = preprocess_and_split(
		data, args.target, args.test_size, args.random_state
	)
	print("\nPreprocessed data split:")
	print("X_train:", X_train.shape, "X_test:", X_test.shape)
	print("y_train:", y_train.shape, "y_test:", y_test.shape)


if __name__ == "__main__":
	main()
