"""
Multi‑store sales forecasting evaluation script
------------------------------------------------
* Evaluates Hybrid (STL+GB), tuned XGBoost, and auto‑ARIMA on the first 10 stores
* Saves per‑store forecast plots (`figures/forecast_store_<id>.png`)
* Aggregates metrics to `results/metrics_summary.csv`
* Outputs XGBoost feature‑importance plot + SHAP summary plot
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA

# ── optional libraries -------------------------------------------------------
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import pmdarima as pm  # type: ignore # for auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

try:
    import shap # type: ignore
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths -------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results";  RESULTS.mkdir(exist_ok=True)
FIGURES = ROOT / "figures";  FIGURES.mkdir(exist_ok=True)

TRAIN_CSV = DATA / "train.csv"
STORE_CSV = DATA / "store.csv"

HOLDOUT = 30
FEATS   = ["DayOfWeek", "Month", "Promo"]

# ── helpers -----------------------------------------------------------------

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index)
    df["DayOfWeek"] = df.index.dayofweek # type: ignore
    df["Month"]     = df.index.month # type: ignore
    df["Promo"]     = df["Promo"].fillna(0)
    return df


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return rmse, mae, mape

# ── load dataset ------------------------------------------------------------
train = pd.read_csv(TRAIN_CSV, parse_dates=["Date"], dtype={"StateHoliday": "string"})
store = pd.read_csv(STORE_CSV)
train = train.merge(store, on="Store", how="left")

# ── 1. global hyper‑parameter tuning for XGBoost ----------------------------
BEST_XGB_PARAMS: dict[str, object] = {}
if XGB_AVAILABLE:
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 6, 8],
        "n_estimators": [200, 400, 600]
    }
    sample = train.sample(5000, random_state=42)  # speed‑up
    sample = make_features(sample.set_index("Date"))

    gs = GridSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=42),
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        verbose=0,
    )
    gs.fit(sample[FEATS], sample["Sales"])
    BEST_XGB_PARAMS = gs.best_params_
    print("✔ Global best XGB params:", BEST_XGB_PARAMS)

# ── function to run models on one store ------------------------------------

def run_models(train_part: pd.DataFrame, test_part: pd.DataFrame):
    # Hybrid (STL + GB) ------------------------------------------------------
    stl = STL(train_part["Sales"], seasonal=13).fit()
    tr_h = make_features(train_part.assign(trend=stl.trend, seasonal=stl.seasonal, residual=stl.resid))

    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    gb.fit(tr_h[FEATS], tr_h["residual"])

    slope = stl.trend.iloc[-1] - stl.trend.iloc[-2]
    trend_fore = stl.trend.iloc[-1] + slope * np.arange(1, HOLDOUT + 1)
    seas_fore  = np.tile(stl.seasonal[-12:], int(np.ceil(HOLDOUT / 12)))[:HOLDOUT]

    tst_feat = make_features(test_part.copy())
    resid_pred  = gb.predict(tst_feat[FEATS])
    hybrid_pred = trend_fore + seas_fore + resid_pred

    # XGBoost / GradBoost ----------------------------------------------------
    tr_ml = make_features(train_part.copy())
    if XGB_AVAILABLE:
        xgb = XGBRegressor(**BEST_XGB_PARAMS, objective="reg:squarederror", random_state=42)
    else:
        xgb = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, random_state=42)
    xgb.fit(tr_ml[FEATS], tr_ml["Sales"])
    xgb_pred = xgb.predict(tst_feat[FEATS])

    # Auto‑ARIMA (if available) ---------------------------------------------
    if PMDARIMA_AVAILABLE:
        ar_model = pm.auto_arima(train_part["Sales"], seasonal=False, suppress_warnings=True)
        arima_pred = ar_model.predict(HOLDOUT)
    else:
        ar_model = ARIMA(train_part["Sales"], order=(5, 1, 1)).fit()
        arima_pred = ar_model.forecast(HOLDOUT).to_numpy()

    actual = test_part["Sales"].to_numpy()
    return actual, hybrid_pred, xgb_pred, arima_pred, xgb

# ── main multi‑store loop ----------------------------------------------------
metrics_rows = []
last_xgb_model = None

for sid in sorted(train["Store"].unique())[:10]:  # first 10 stores
    df_store = train[train["Store"] == sid].sort_values("Date").set_index("Date")
    train_part = df_store.iloc[:-HOLDOUT]
    test_part  = df_store.iloc[-HOLDOUT:]

    actual, hybrid_pred, xgb_pred, arima_pred, last_xgb_model = run_models(train_part, test_part)

    for mdl, pred in zip(["Hybrid", "XGBoost" if XGB_AVAILABLE else "GradBoost", "ARIMA"],
                          [hybrid_pred, xgb_pred, arima_pred]):
        rmse, mae, mape = metrics(actual, pred)
        metrics_rows.append({"Store": sid, "Model": mdl, "RMSE": rmse, "MAE": mae, "MAPE": mape})

    # save store plot --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(test_part.index, actual, label="Actual", color="black")
    plt.plot(test_part.index, hybrid_pred, label="Hybrid", linestyle="--")
    plt.plot(test_part.index, xgb_pred, label="XGBoost", linestyle=":")
    plt.plot(test_part.index, arima_pred, label="ARIMA", linestyle="-.")
    plt.title(f"Forecast – Store {sid}")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / f"forecast_store_{sid}.png")
    plt.close()

# ── save metrics ------------------------------------------------------------
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(RESULTS / "metrics_summary.csv", index=False)
print("✔ metrics_summary.csv saved")

# ── XGB feature importance + SHAP ------------------------------------------
if XGB_AVAILABLE and last_xgb_model is not None:
    # 1) Feature importance plot
    fi_series = (pd.Series(last_xgb_model.feature_importances_, index=FEATS)
                   .sort_values())
    fi_series.plot(kind="barh", title="XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES / "xgb_feature_importance.png")
    plt.close()
    print("✔ xgb_feature_importance.png saved")

    # 2) SHAP summary plot (optional)
    if SHAP_AVAILABLE:
        explainer = shap.Explainer(last_xgb_model)
        # Use a small sample for speed
        sample_df = make_features(train.sample(2000, random_state=42).set_index("Date"))
        shap_values = explainer(sample_df[FEATS])
        shap.summary_plot(shap_values, sample_df[FEATS], show=False)
        plt.tight_layout()
        plt.savefig(FIGURES / "shap_summary.png")
        plt.close()
        print("✔ shap_summary.png saved")

print("✅ Evaluation complete.")

# ---- Aggregate bar chart of average errors -----------------
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path(__file__).resolve().parent / "results" / "metrics_summary.csv"
df = pd.read_csv(csv_path)
avg_metrics = df.groupby("Model")[["RMSE", "MAE", "MAPE"]].mean()

avg_metrics.plot(kind="bar", figsize=(8, 6))
plt.title("Average Forecasting Errors Across All Stores")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()

out_file = Path(__file__).resolve().parent / "figures" / "model_avg_error_comparison.png"
plt.savefig(out_file)
plt.show()
print(f"✔ model_avg_error_comparison.png saved in figures/")
