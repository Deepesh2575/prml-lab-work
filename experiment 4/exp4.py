import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv(r"C:\programming files of my\PRML lab Work\experiment 4\Social_Network_Ads.csv")
X = df[["Gender", "Age", "EstimatedSalary"]]
y = df["Purchased"]

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["Age", "EstimatedSalary"]),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), ["Gender"]),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
for name, clf in models.items():
    m = make_pipeline(pre, clf)
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    print(name, "Acc=", round(accuracy_score(y_test, p), 3), "Prec=", round(precision_score(y_test, p, zero_division=0), 3), "Rec=", round(recall_score(y_test, p, zero_division=0), 3), "F1=", round(f1_score(y_test, p, zero_division=0), 3))

fig, ax = plt.subplots(figsize=(8, 5))
vals = [accuracy_score(y_test, make_pipeline(pre, clf).fit(X_train, y_train).predict(X_test)) for clf in models.values()]
ax.bar(models.keys(), vals)
ax.set_title("Accuracy")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(r"C:\programming files of my\PRML lab Work\experiment 4\model_compare.png", dpi=150)
