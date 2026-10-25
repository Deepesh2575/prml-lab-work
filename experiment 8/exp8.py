"""Short SVM spam classifier using the UCI SMS Spam Collection."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

data = pd.read_csv(Path(__file__).with_name("SMSSpamCollection"), sep="\t", header=None, names=["label", "text"])
X_train, X_test, y_train, y_test = train_test_split(
	data["text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
)
model = make_pipeline(TfidfVectorizer(), LinearSVC())
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test), digits=3))