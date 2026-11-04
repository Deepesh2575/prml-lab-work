"""Email spam detection with an SVM and a Kaggle spam/ham email dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

DATA_PATH = Path(__file__).with_name("spam_ham_dataset.csv")
LABELS = ["ham", "spam"]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Download spam_ham_dataset.csv from "
        "https://www.kaggle.com/datasets/venky73/spam-mails-dataset "
        "and place it beside exp8.py."
    )

data = pd.read_csv(DATA_PATH)
if not {"label", "text"}.issubset(data.columns):
    raise ValueError("The CSV must contain 'label' and 'text' columns.")

missing_rows = data[["label", "text"]].isna().any(axis=1).sum()
data = data.dropna(subset=["label", "text"]).copy()
data["label"] = data["label"].astype(str).str.strip().str.lower()
data["text"] = data["text"].astype(str).str.strip()
invalid_rows = (~data["label"].isin(LABELS) | data["text"].eq("")).sum()
data = data[data["label"].isin(LABELS) & data["text"].ne("")]
if data["label"].nunique() != 2:
    raise ValueError("The CSV must contain both ham and spam emails.")

print(f"Emails in CSV: {len(data) + missing_rows + invalid_rows}")
print(f"Rows excluded (missing fields or invalid/empty values): {missing_rows + invalid_rows}")
print(f"Emails used: {len(data)}")
print("Class counts:\n", data["label"].value_counts().to_string())

X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
)
model = make_pipeline(TfidfVectorizer(stop_words="english"), LinearSVC())
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nTest results:\n", classification_report(y_test, predictions, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
data["label"].value_counts().reindex(LABELS).plot.bar(
    ax=axes[0], color=["steelblue", "tomato"]
)
axes[0].set(title="Email class balance", xlabel="Class", ylabel="Emails")
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, labels=LABELS, display_labels=LABELS, cmap="Blues", ax=axes[1]
)
axes[1].set_title("SVM confusion matrix")
fig.tight_layout()
fig.savefig(Path(__file__).with_name("email_spam_results.png"), dpi=150)
plt.show()

email = input("Paste an email to classify (or press Enter to finish): ").strip()
if email:
    print("Prediction:", model.predict([email])[0].upper())
