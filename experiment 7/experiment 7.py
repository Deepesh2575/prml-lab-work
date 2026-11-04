"""Train an ANN to recognize digits in Kaggle's MNIST dataset."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


data_dir = Path(__file__).resolve().parent / "mnist_data"
data_dir.mkdir(exist_ok=True)
train_file, test_file = data_dir / "mnist_train.csv", data_dir / "mnist_test.csv"

if not train_file.exists() or not test_file.exists():
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", "oddrationale/mnist-in-csv", "-p", str(data_dir), "--unzip"],
        check=True,
    )

train, test = pd.read_csv(train_file), pd.read_csv(test_file)
x_train = train.drop(columns="label").to_numpy(dtype=np.float32) / 255
y_train = train["label"].to_numpy()
x_test = test.drop(columns="label").to_numpy(dtype=np.float32) / 255
y_test = test["label"].to_numpy()

if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
    raise ValueError("MNIST pixel data contains missing or invalid values.")

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, epochs=20, batch_size=128, validation_data=(x_val, y_val),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
)

plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST ANN Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(data_dir / "accuracy.png")
plt.show()

_, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.2%}")
print(classification_report(y_test, model.predict(x_test, verbose=0).argmax(axis=1)))
model.save(data_dir / "mnist_ann.keras")