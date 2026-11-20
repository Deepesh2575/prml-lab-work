"""Train an email spam SVM, print results, and save evaluation graphs."""

import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

FOLDER = Path(__file__).parent
DATA = FOLDER / "spam_ham_dataset.csv"
LABELS = ["ham", "spam"]

if not DATA.exists():
    print("Dataset missing; downloading from Kaggle...")
    url = "https://www.kaggle.com/api/v1/datasets/download/venky73/spam-mails-dataset"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                files = [f for f in archive.namelist() if f.lower().endswith(".csv")]
                name = next(
                    (f for f in files if PurePosixPath(f).name.lower() == DATA.name.lower()),
                    files[0] if len(files) == 1 else None,
                )
                if name is None:
                    raise ValueError("Kaggle archive must contain one email CSV.")
                with archive.open(name) as source:
                    data = pd.read_csv(source)
        if not {"label", "text"}.issubset(data.columns):
            raise ValueError("Dataset must have 'label' and 'text' columns.")
        data.to_csv(DATA, index=False)
        print(f"Downloaded {len(data):,} emails.")
    except (OSError, urllib.error.URLError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"Dataset download failed: {error}") from error
else:
    data = pd.read_csv(DATA)

data = data.dropna(subset=["label", "text"]).copy()
data["label"] = data["label"].astype(str).str.strip().str.lower()
data["text"] = data["text"].astype(str).str.strip()
data = data[data["label"].isin(LABELS) & data["text"].ne("")]
if data["label"].nunique() != 2:
    raise ValueError("Dataset must contain both ham and spam emails.")

print(f"\nEmails: {len(data):,}")
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
    ax=axes[0], color=["steelblue", "tomato"], title="Email class counts"
)
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, labels=LABELS, cmap="Blues", ax=axes[1]
)
fig.tight_layout()
graph = FOLDER / "email_spam_results.png"
fig.savefig(graph, dpi=150)
print(f"Graph saved to: {graph}")
plt.show()

try:
    email = input("\nPaste an email to classify (Enter to skip): ").strip()
except EOFError:
    email = ""
if email:
    print("Prediction:", model.predict([email])[0].upper())
