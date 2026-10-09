"""
Advertising Sales Prediction using Linear Regression
====================================================
Dataset : Advertising_Budget_and_Sales.csv  (200 markets: TV, Radio, Newspaper budgets -> Sales)

Pipeline (follows the assignment steps):
  1. Load data, print shape / columns / target statistics
  2. Preprocess: missing values, duplicates, text columns, outliers
  3. Split 80/20 (random_state=42); scale with StandardScaler INSIDE a Pipeline (no data leakage)
  4. Compare 3 algorithms:
       - Multiple Linear Regression          (baseline, easy to interpret)
       - Polynomial Linear Regression (deg 2) (still linear regression: linear in its coefficients,
                                               adds squared and interaction terms)
       - Random Forest Regressor              (non-linear benchmark)
  5. Choose the final model with 5-fold cross-validation on the TRAINING set (not the test score)
  6. Evaluate on the test set: MAE, MSE, RMSE, R2, adjusted R2
  7. Save 7 plots to plots/, the model to model.json (used by the desktop app) and sales_model.joblib

Run:  python train_model.py
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "Advertising_Budget_and_Sales.csv"
PLOT_DIR = BASE / "plots"
PLOT_DIR.mkdir(exist_ok=True)

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"
SEED = 42

# Categorical colours (first three slots of a colour-blind-validated palette). One colour per model,
# used the same way in every chart.
C_BASE, C_POLY, C_RF = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
MODEL_COLORS = {"Multiple Linear Regression": C_BASE,
                "Polynomial Linear Regression (deg 2)": C_POLY,
                "Random Forest": C_RF}

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOT_DIR / name, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- 1. Load
raw = pd.read_csv(DATA_FILE)
print(f"Shape: {raw.shape}")
print("Columns:", list(raw.columns))

# The file has an unnamed index column and long headers like 'TV Ad Budget ($)'.
index_cols = [c for c in raw.columns if str(c).startswith("Unnamed")]
df = raw.drop(columns=index_cols)
original_labels = dict(zip(FEATURES + [TARGET], df.columns))
df.columns = FEATURES + [TARGET]
print("Original labels:", original_labels)
print(f"\n{TARGET} (target) summary:\n{df[TARGET].describe().round(2)}")
print("(Regression problem: the target is a number, so there are no class counts.)")

# ----------------------------------------------------------------------------- 2. Preprocess
prep = {
    "rows_raw": int(len(df)),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "text_columns": int((df.dtypes == "object").sum()),
    "dropped_index_columns": index_cols,
}
df = df.drop_duplicates().dropna().reset_index(drop=True)
prep["rows_clean"] = int(len(df))

outliers = {}
for col in FEATURES + [TARGET]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers[col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
prep["iqr_outliers"] = outliers
print("\nPreprocessing:", json.dumps(prep, indent=1))

describe = df.describe().round(3)
corr = df.corr()
print("\nCorrelation matrix:\n", corr.round(3))

# ----------------------------------------------------------------------------- 3. Split
X, y = df[FEATURES], df[TARGET]
# stratify= is for classification; a numeric target is split randomly.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

# ----------------------------------------------------------------------------- 4. Models
models = {
    "Multiple Linear Regression": Pipeline([
        ("scale", StandardScaler()), ("model", LinearRegression())]),
    "Polynomial Linear Regression (deg 2)": Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()), ("model", LinearRegression())]),
    "Random Forest": Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=300, random_state=SEED))]),
}
short = {"Multiple Linear Regression": "Multiple LR",
         "Polynomial Linear Regression (deg 2)": "Polynomial LR (deg 2)",
         "Random Forest": "Random Forest"}

# ----------------------------------------------------------------------------- 5. Cross-validate (training set only)
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}
for name, pipe in models.items():
    r = cross_validate(pipe, X_train, y_train, cv=cv,
                       scoring=("r2", "neg_root_mean_squared_error"))
    cv_results[name] = {
        "cv_r2_mean": float(r["test_r2"].mean()), "cv_r2_std": float(r["test_r2"].std()),
        "cv_rmse_mean": float(-r["test_neg_root_mean_squared_error"].mean()),
        "cv_rmse_std": float(r["test_neg_root_mean_squared_error"].std()),
    }
    print(f"CV  {name:38s} R2 {cv_results[name]['cv_r2_mean']:.4f} "
          f"+/- {cv_results[name]['cv_r2_std']:.3f}   RMSE {cv_results[name]['cv_rmse_mean']:.3f}")

# Final model = best CV RMSE among the LINEAR REGRESSION models (this project's topic).
# Random Forest is a benchmark only; we report honestly if it would score higher.
linear_names = [n for n in models if "Linear" in n]
final_name = min(linear_names, key=lambda n: cv_results[n]["cv_rmse_mean"])
best_any = min(models, key=lambda n: cv_results[n]["cv_rmse_mean"])
print(f"\nFinal model (best CV RMSE among linear regression models): {final_name}")
print(f"Best CV RMSE overall: {best_any}")


# ----------------------------------------------------------------------------- 6. Train + test
def regression_metrics(y_true, y_pred, n_params):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    adj = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)), "R2": float(r2), "Adj_R2": float(adj)}


fitted, preds_test, preds_train, metrics_test, metrics_train = {}, {}, {}, {}, {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    preds_train[name], preds_test[name] = pipe.predict(X_train), pipe.predict(X_test)
    n_params = pipe[:-1].transform(X_train.iloc[:1]).shape[1]  # number of predictors used
    metrics_train[name] = regression_metrics(y_train, preds_train[name], n_params)
    metrics_test[name] = regression_metrics(y_test, preds_test[name], n_params)
    m = metrics_test[name]
    print(f"TEST {name:38s} MAE {m['MAE']:.3f} RMSE {m['RMSE']:.3f} R2 {m['R2']:.4f} AdjR2 {m['Adj_R2']:.4f}")

base_name = "Multiple Linear Regression"
final_pipe = fitted[final_name]

# Baseline coefficients in the original units (for interpretation) + standardised ones
base_raw = LinearRegression().fit(X_train, y_train)
coef_raw = dict(zip(FEATURES, base_raw.coef_))
coef_std = dict(zip(FEATURES, fitted[base_name]["model"].coef_))
print("\nBaseline equation: Sales = %.4f + %.5f*TV + %.5f*Radio + %.5f*Newspaper" %
      (base_raw.intercept_, *coef_raw.values()))

# ----------------------------------------------------------------------------- 7. Plots
sns.set_style("whitegrid")
LABELS = {"TV": "TV budget", "Radio": "Radio budget", "Newspaper": "Newspaper budget", "Sales": "Sales"}

# 1. Distributions (target + features)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, col in zip(axes, FEATURES + [TARGET]):
    ax.hist(df[col], bins=18, color=C_BASE, edgecolor="white", linewidth=1)
    ax.set_title(LABELS[col]); ax.set_ylabel("Number of markets" if col == "TV" else "")
save(fig, "01_distributions.png")

# 2. Correlation heatmap
fig, ax = plt.subplots(figsize=(5.2, 4.4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, square=True,
            cbar=False, ax=ax, linewidths=2, linecolor="white", annot_kws={"size": 11})
ax.set_title("Correlation matrix"); ax.grid(False)
save(fig, "02_correlation_heatmap.png")

# 3. Each channel vs Sales
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
for ax, col in zip(axes, FEATURES):
    ax.scatter(df[col], df[TARGET], s=22, color=C_BASE, alpha=0.7, edgecolor="white", linewidth=0.5)
    b, a = np.polyfit(df[col], df[TARGET], 1)
    xs = np.linspace(df[col].min(), df[col].max(), 50)
    ax.plot(xs, a + b * xs, color=C_POLY, lw=2)
    ax.set_title(f"{LABELS[col]} vs Sales  (r = {corr.loc[col, TARGET]:.2f})")
    ax.set_xlabel(f"{LABELS[col]} ($)")
axes[0].set_ylabel("Sales ($)")
save(fig, "03_features_vs_sales.png")

# 4. Actual vs predicted: baseline vs final
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True, sharey=True)
lim = [0, max(y_test.max(), *[p.max() for p in preds_test.values()]) + 1]
for ax, name in zip(axes, [base_name, final_name]):
    ax.scatter(y_test, preds_test[name], s=34, color=MODEL_COLORS[name], alpha=0.85,
               edgecolor="white", linewidth=0.6)
    ax.plot(lim, lim, "--", color=MUTED, lw=1.2)
    ax.text(0.04, 0.93, f"R² = {metrics_test[name]['R2']:.3f}   RMSE = {metrics_test[name]['RMSE']:.2f}",
            transform=ax.transAxes, fontsize=10, va="top")
    ax.set_title(short[name]); ax.set_xlabel("Actual sales ($)"); ax.set_xlim(lim); ax.set_ylim(lim)
axes[0].set_ylabel("Predicted sales ($)")
save(fig, "04_actual_vs_predicted.png")

# 5. Residuals: baseline vs final
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
for row, name in enumerate([base_name, final_name]):
    res = y_test - preds_test[name]
    ax = axes[row, 0]
    ax.scatter(preds_test[name], res, s=30, color=MODEL_COLORS[name], alpha=0.85,
               edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=MUTED, ls="--", lw=1.2)
    ax.set_title(f"{short[name]}: residuals vs predicted")
    ax.set_xlabel("Predicted sales ($)"); ax.set_ylabel("Residual ($)")
    ax = axes[row, 1]
    ax.hist(res, bins=12, color=MODEL_COLORS[name], edgecolor="white", linewidth=1)
    ax.axvline(0, color=MUTED, ls="--", lw=1.2)
    ax.set_title(f"{short[name]}: residual distribution")
    ax.set_xlabel("Residual ($)"); ax.set_ylabel("Count")
save(fig, "05_residuals.png")

# 6. Model comparison (CV RMSE on training folds, RMSE on test set)
names = list(models)
x = np.arange(len(names)); w = 0.36
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
cv_rmse = [cv_results[n]["cv_rmse_mean"] for n in names]
te_rmse = [metrics_test[n]["RMSE"] for n in names]
for ax, vals, title in [(axes[0], cv_rmse, "5-fold CV RMSE (training set) - lower is better"),
                        (axes[1], te_rmse, "Test-set RMSE - lower is better")]:
    bars = ax.bar(x, vals, color=[MODEL_COLORS[n] for n in names], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([short[n].replace(" (", "\n(") for n in names])
    ax.set_title(title, fontsize=10); ax.set_ylabel("RMSE"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(vals) * 1.18)
save(fig, "06_model_comparison.png")

# 7. Why the baseline under-fits: the effect of TV depends on Radio (interaction)
fig, ax = plt.subplots(figsize=(6.2, 4.4))
q = pd.qcut(df["Radio"], 3, labels=["Low radio", "Medium radio", "High radio"])
for lab, col in zip(["Low radio", "Medium radio", "High radio"], [C_BASE, C_POLY, C_RF]):
    sub = df[q == lab]
    ax.scatter(sub["TV"], sub[TARGET], s=24, color=col, alpha=0.85, edgecolor="white",
               linewidth=0.5, label=lab)
    b, a = np.polyfit(sub["TV"], sub[TARGET], 1)
    xs = np.linspace(sub["TV"].min(), sub["TV"].max(), 30)
    ax.plot(xs, a + b * xs, color=col, lw=2)
ax.set_xlabel("TV budget ($)"); ax.set_ylabel("Sales ($)")
ax.set_title("TV and Radio work together"); ax.legend(frameon=False, loc="upper left")
save(fig, "07_tv_radio_interaction.png")

# ----------------------------------------------------------------------------- 8. Save model
def export_linear(pipe):
    """Export a (poly ->) scaler -> LinearRegression pipeline as plain numbers (no sklearn needed to predict)."""
    poly = pipe.named_steps.get("poly")
    powers = poly.powers_.tolist() if poly is not None else np.eye(len(FEATURES), dtype=int).tolist()
    sc, lr = pipe.named_steps["scale"], pipe.named_steps["model"]
    names_out = (poly.get_feature_names_out(FEATURES).tolist() if poly is not None else FEATURES)
    return {"powers": powers, "terms": names_out, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": lr.coef_.tolist(), "intercept": float(lr.intercept_)}


model_json = {
    "features": FEATURES,
    "labels": original_labels,
    "final": final_name,
    "baseline": base_name,
    "models": {},
    "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in FEATURES + [TARGET]},
    "means": {c: float(df[c].mean()) for c in FEATURES + [TARGET]},
}
for name in (base_name, "Polynomial Linear Regression (deg 2)"):
    model_json["models"][name] = {
        **export_linear(fitted[name]),
        "test": metrics_test[name], "cv": cv_results[name]}
(BASE / "model.json").write_text(json.dumps(model_json, indent=1))
joblib.dump(final_pipe, BASE / "sales_model.joblib")

comparison = pd.DataFrame({
    "Model": names,
    "CV R2": [cv_results[n]["cv_r2_mean"] for n in names],
    "CV R2 std": [cv_results[n]["cv_r2_std"] for n in names],
    "CV RMSE": cv_rmse,
    "Test MAE": [metrics_test[n]["MAE"] for n in names],
    "Test MSE": [metrics_test[n]["MSE"] for n in names],
    "Test RMSE": te_rmse,
    "Test R2": [metrics_test[n]["R2"] for n in names],
    "Test Adj R2": [metrics_test[n]["Adj_R2"] for n in names],
    "Train R2": [metrics_train[n]["R2"] for n in names],
})
comparison.round(4).to_csv(BASE / "model_comparison.csv", index=False)
print("\n", comparison.round(4).to_string(index=False))

results = {
    "data_file": DATA_FILE.name, "original_labels": original_labels, "preprocessing": prep,
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "describe": json.loads(describe.to_json()),
    "correlation_with_sales": {c: float(corr.loc[c, TARGET]) for c in FEATURES},
    "radio_newspaper_corr": float(corr.loc["Radio", "Newspaper"]),
    "tv_radio_corr": float(corr.loc["TV", "Radio"]),
    "baseline_intercept": float(base_raw.intercept_),
    "baseline_coef": {k: float(v) for k, v in coef_raw.items()},
    "baseline_coef_standardised": {k: float(v) for k, v in coef_std.items()},
    "final_model": final_name, "best_cv_overall": best_any,
    "cv": cv_results, "test": metrics_test, "train": metrics_train,
}
(BASE / "results.json").write_text(json.dumps(results, indent=1))
print("\nSaved: plots/*.png, model.json, sales_model.joblib, model_comparison.csv, results.json")
