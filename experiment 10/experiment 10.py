from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = Path(__file__).parent
data_file = folder / "TSLA.csv"
if not data_file.exists():
    print("Downloading Kaggle stock prices...")
    url = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/sp-500-stocks"
    with urlopen(url, timeout=120) as response, ZipFile(BytesIO(response.read())) as archive:
        stocks = pd.read_csv(archive.open("sp500_stocks.csv"), usecols=["Date", "Symbol", "Close"])
    stocks = stocks[stocks["Symbol"].eq("TSLA")].drop(columns="Symbol")
    if stocks.empty:
        raise ValueError("The Kaggle data contains no TSLA prices.")
    stocks.to_csv(data_file, index=False)

prices = pd.read_csv(data_file, parse_dates=["Date"]).dropna().sort_values("Date")
prices = prices[prices["Close"] > 0]
returns = np.log(prices["Close"]).diff().dropna()
days, simulations = 30, 2000
rng = np.random.default_rng(42)
paths = prices["Close"].iloc[-1] * np.exp(
    np.cumsum(rng.normal(returns.mean(), returns.std(), (days, simulations)), axis=0)
)
current = prices["Close"].iloc[-1]
paths = np.vstack([np.full(simulations, current), paths])
average = paths.mean(axis=1)
low, middle, high = np.percentile(paths[-1], [5, 50, 95])

print(f"\nCompany: Tesla (TSLA) | History: {len(prices)} days")
print(f"Current stock price: ${current:,.2f}")
print(f"Predicted average price after {days} days: ${average[-1]:,.2f}")
print(f"Simulated median after {days} days: ${middle:,.2f}")
print(f"90% simulation range: ${low:,.2f} - ${high:,.2f}")
print("Note: Monte Carlo scenarios are estimates, not guaranteed predictions.")

plt.figure(figsize=(9, 5))
plt.plot(range(days + 1), paths[:, :100], alpha=0.25, linewidth=0.8)
plt.plot(range(days + 1), average, color="red", linewidth=2, label="Average Predicted Price")
plt.title("Monte Carlo Stock Price Simulation")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
graph = folder / "stock_monte_carlo.png"
plt.savefig(graph, dpi=150)
print(f"Graph saved: {graph}")
plt.show()