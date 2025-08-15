import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import GradientBoostingRegressor
import warnings

# ── optional: silence pandas/sklearn future warnings ──────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)

# ── file paths ────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent
DATA   = ROOT / "data"
TRAIN  = DATA / "train.csv"
TEST   = DATA / "test.csv"
STORE  = DATA / "store.csv"

# ── load & merge ──────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN, parse_dates=["Date"], dtype={"StateHoliday": "string"})
test  = pd.read_csv(TEST,  parse_dates=["Date"], dtype={"StateHoliday": "string"})
store = pd.read_csv(STORE)

train = train.merge(store, on="Store", how="left")
test  = test.merge(store,  on="Store", how="left")

# ── helper functions ──────────────────────────────────────────────────────────
def stl_decompose(series: pd.Series, seasonal: int = 13):
    res = STL(series, seasonal=seasonal).fit()
    return res.trend, res.seasonal, res.resid

def features(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)       # explicit cast for linting
    df["DayOfWeek"] = idx.dayofweek # type: ignore
    df["Month"]     = idx.month # type: ignore
    df["Promo"]     = df["Promo"].fillna(0)
    return df

def extrapolate_trend(tr: pd.Series, steps: int) -> np.ndarray:
    last  = tr.iloc[-1]
    slope = tr.iloc[-1] - tr.iloc[-2]      # last observed slope
    return last + slope * np.arange(1, steps + 1)

def repeat_seasonality(season: np.ndarray, length: int) -> np.ndarray:
    return np.tile(season, int(np.ceil(length / len(season))))[:length]

def train_residual_model(X, y):
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X, y)
    return model

# ── forecasting loop ──────────────────────────────────────────────────────────
STORE_LIMIT = 10                                       # quick demo; raise if desired
store_ids   = sorted(train["Store"].unique())[:STORE_LIMIT]

rows     = []                                          # collect forecasts for CSV
png_list = []                                          # collect image filenames

for sid in store_ids:
    tr = train[train["Store"] == sid].sort_values("Date")
    te =  test[test["Store"]  == sid].sort_values("Date")
    if tr.empty or te.empty:
        continue

    # STL decomposition
    tr = tr.set_index("Date")
    trend, seas, resid = stl_decompose(tr["Sales"], seasonal=13)
    tr["trend"], tr["seasonal"], tr["residual"] = trend, seas, resid

    # Feature engineering
    tr = features(tr)
    te = features(te.set_index("Date"))

    FEATS = ["DayOfWeek", "Month", "Promo"]
    model = train_residual_model(tr[FEATS], tr["residual"])

    # Predict residuals and rebuild forecast
    te["residual_pred"] = model.predict(te[FEATS])
    te["trend"]    = extrapolate_trend(tr["trend"], len(te))
    te["seasonal"] = repeat_seasonality(tr["seasonal"][-12:], len(te)) # type: ignore
    te["Forecast"] = te["trend"] + te["seasonal"] + te["residual_pred"]

    # Collect rows for CSV
    rows.extend(
        {"Store": sid, "Date": d, "Forecast": f}
        for d, f in zip(te.index, te["Forecast"])
    )

    # Plot and save PNG
    plt.figure(figsize=(10, 5))
    plt.plot(tr.index[-120:], tr["Sales"][-120:], label="Train Sales")
    plt.plot(te.index, te["Forecast"], "--", label="Forecast")
    plt.legend()
    plt.title(f"Hybrid Forecast – Store {sid}")
    plt.tight_layout()

    png_file = ROOT / f"forecast_store_{sid}.png"
    plt.savefig(png_file)
    plt.close()

    png_list.append(png_file.name)
    print(f"✔ Plot saved for Store {sid}")

# ── write combined forecast CSV ───────────────────────────────────────────────
pd.DataFrame(rows).to_csv(ROOT / "all_store_forecasts.csv", index=False)
print("✔ Combined forecasts CSV saved.")

# ── quick HTML dashboard ──────────────────────────────────────────────────────
html_parts = [
    "<html><head><title>Sales Forecast Dashboard</title></head><body>",
    "<h1>Hybrid Sales Forecast Dashboard</h1>"
]
for sid, img in zip(store_ids, png_list):
    html_parts.append(f"<h2>Store {sid}</h2>")
    html_parts.append(f"<img src='{img}' width='650'><br><br>")
html_parts.append("</body></html>")

(ROOT / "dashboard.html").write_text("\n".join(html_parts))
print("✔ Dashboard created.")
