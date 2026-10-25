"""Predict Boston housing prices using multivariate linear regression."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def main():
    print("Loading Boston Housing data...")
    ds = fetch_openml(name="boston", version=1, as_frame=True)
    X = ds.data.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(ds.target, errors="coerce")

    X = X.loc[y.notna()]
    y = y[y.notna()]
    if X.empty:
        raise ValueError("No valid target values found in the Boston dataset.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    actual = y_test.to_numpy()
    axes[0].scatter(actual, pred, alpha=0.7)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    axes[0].plot([lo, hi], [lo, hi], color="red", linestyle="--", label="Perfect")
    axes[0].set(title="Actual vs Predicted", xlabel="Actual", ylabel="Predicted")
    axes[0].legend(); axes[0].grid(alpha=0.25)

    residuals = actual - pred
    axes[1].scatter(pred, residuals, alpha=0.7)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set(title="Residuals", xlabel="Predicted", ylabel="Residual")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    plot_path = Path(__file__).with_name("boston_housing_regression.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")

    print(pd.DataFrame({
        "Actual": y_test.to_numpy()[:5],
        "Predicted": pred[:5],
    }).round(2).to_string(index=False))

    reg = model.named_steps["linearregression"]
    print("Coefficients:")
    for col, coef in zip(X.columns, reg.coef_):
        print(f"{col}: {coef:.3f}")


if __name__ == "__main__":
    main()
