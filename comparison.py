"""
Model comparison for Store 1
Benchmarks:
  • Hybrid  (STL  + GradientBoost residual)
  • XGBoost (fallback to GradientBoost if XGBoost unavailable)
  • ARIMA   (5,1,1)

Evaluates on last 30 days, prints RMSE / MAE / MAPE, and saves model_comparison.png
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL

# ── XGBoost optional ----------------------------------------------------------
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    print("⚠️  XGBoost not available – using GradientBoost baseline.")
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths --------------------------------------------------------------------
ROOT  = Path(__file__).resolve().parent
DATA  = ROOT / "data"
TRAIN = DATA / "train.csv"
STORE = DATA / "store.csv"

# ── load data ----------------------------------------------------------------
train = pd.read_csv(TRAIN, parse_dates=["Date"], dtype={"StateHoliday": "string"})
store = pd.read_csv(STORE)
train = train.merge(store, on="Store", how="left")

# ── configuration -------------------------------------------------------------
STORE_ID  = 1
HOLDOUT   = 30                 # days in validation window
DEBUG_PLOT = False             # set True to see trend extrapolation

# ── subset store --------------------------------------------------------------
df = train[train["Store"] == STORE_ID].sort_values("Date").set_index("Date")
train_part = df.iloc[:-HOLDOUT]
test_part  = df.iloc[-HOLDOUT:]

# ── feature engineering -------------------------------------------------------
def make_features(dfr: pd.DataFrame) -> pd.DataFrame:
    dfr = dfr.copy()
    dfr.index = pd.DatetimeIndex(dfr.index)      # ensures .dayofweek/.month
    dfr["DayOfWeek"] = dfr.index.dayofweek # type: ignore
    dfr["Month"]     = dfr.index.month # type: ignore
    dfr["Promo"]     = dfr["Promo"].fillna(0)
    return dfr

# ── safe metrics (no divide‑by‑zero MAPE) -------------------------------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return rmse, mae, mape

actual = test_part["Sales"].to_numpy()
results: dict[str, tuple[float, float, float]] = {}

# ── 1. Hybrid -----------------------------------------------------------------
stl = STL(train_part["Sales"], seasonal=13).fit()
train_h = train_part.copy()
train_h["trend"]    = stl.trend
train_h["seasonal"] = stl.seasonal
train_h["residual"] = stl.resid
train_h = make_features(train_h)

FEATS = ["DayOfWeek", "Month", "Promo"]
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
gb.fit(train_h[FEATS], train_h["residual"])

# trend extrapolation (linear slope, can average more points if noisy)
slope       = stl.trend.iloc[-1] - stl.trend.iloc[-2]
trend_fore  = stl.trend.iloc[-1] + slope * np.arange(1, HOLDOUT + 1)
season_last = stl.seasonal[-12:]
seas_fore   = np.tile(season_last, int(np.ceil(HOLDOUT / 12)))[:HOLDOUT]

# (optional) visualize projected trend
if DEBUG_PLOT:
    plt.figure(figsize=(10, 4))
    plt.plot(train_part.tail(60).index, stl.trend.tail(60), label="Historical trend")
    plt.plot(test_part.index, trend_fore, "--", label="Projected trend")
    plt.title("Trend extrapolation check")
    plt.legend(); plt.tight_layout(); plt.show()

test_feat   = make_features(test_part.copy())
resid_pred  = gb.predict(test_feat[FEATS])
hybrid_pred = trend_fore + seas_fore + resid_pred
results["Hybrid"] = metrics(actual, hybrid_pred)

# ── 2. XGBoost / GradientBoost baseline --------------------------------------
train_ml = make_features(train_part.copy())
if XGB_AVAILABLE:
    ml_model = XGBRegressor(
        n_estimators=400, learning_rate=0.05,
        objective="reg:squarederror", random_state=42
    )
    label = "XGBoost"
else:
    ml_model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, random_state=42
    )
    label = "GradBoost"

ml_model.fit(train_ml[FEATS], train_ml["Sales"])
ml_pred = ml_model.predict(test_feat[FEATS])
results[label] = metrics(actual, ml_pred)

# ── 3. ARIMA ------------------------------------------------------------------
ar_model = ARIMA(train_part["Sales"], order=(5, 1, 1)).fit()
arima_pred = ar_model.forecast(HOLDOUT).to_numpy()
results["ARIMA(5,1,1)"] = metrics(actual, arima_pred)

# ── print results -------------------------------------------------------------
print(f"\nModel comparison on last {HOLDOUT} days – Store {STORE_ID}\n")
print(f"{'Model':<12} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8}")
for name, (rmse, mae, mape) in results.items():
    print(f"{name:<12} {rmse:8.1f} {mae:8.1f} {mape:8.2f}")

# ── plot forecasts ------------------------------------------------------------
plt.figure(figsize=(11, 6))
plt.plot(test_part.index, actual,       label="Actual", color="black")
plt.plot(test_part.index, hybrid_pred,  label="Hybrid", linestyle="--")
plt.plot(test_part.index, ml_pred,      label=label,   linestyle=":")
plt.plot(test_part.index, arima_pred,   label="ARIMA", linestyle="-.")

plt.title(f"Forecast Comparison – Store {STORE_ID}")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "model_comparison.png")
plt.show()

print("\n✔ model_comparison.png saved.")
