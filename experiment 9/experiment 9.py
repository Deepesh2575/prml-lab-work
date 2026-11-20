from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

folder = Path(__file__).parent
data_file = folder / "Tweets.csv"
if not data_file.exists():
    print("Downloading Kaggle sentiment data...")
    url = "https://www.kaggle.com/api/v1/datasets/download/yasserh/twitter-tweets-sentiment-dataset"
    with urlopen(url, timeout=60) as response, ZipFile(BytesIO(response.read())) as archive:
        data_file.write_bytes(archive.read("Tweets.csv"))

data = pd.read_csv(data_file, usecols=["text", "sentiment"]).dropna()
data = data[data["sentiment"].isin(["positive", "negative"])]
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["sentiment"], test_size=0.2, random_state=42, stratify=data["sentiment"]
)
model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"\nTweets used: {len(data):,} (neutral tweets excluded)")
print("Class counts:\n", data["sentiment"].value_counts().to_string())
print("\nTest results:\n", classification_report(y_test, predictions, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
data["sentiment"].value_counts().plot.bar(ax=axes[0], title="Positive / negative tweets")
ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=axes[1], cmap="Blues")
fig.tight_layout()
fig.savefig(folder / "sentiment_results.png", dpi=150)
print(f"Graph saved: {folder / 'sentiment_results.png'}")
plt.show()

statement = input("\nEnter a statement: ").strip()
if statement:
    print("Sentiment:", model.predict([statement])[0].upper())