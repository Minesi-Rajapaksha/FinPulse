import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from arch import arch_model


# =====================================================
# EVALUATION METRIC FUNCTIONS
# =====================================================

def qlike(actual, predicted):
    """
    Quasi-Likelihood (QLIKE) loss function for volatility forecasting:
    QLIKE = mean( log(predicted_var) + actual_var / predicted_var )
    """
    epsilon = 1e-8
    actual_var = np.maximum(np.array(actual) ** 2, epsilon)
    predicted_var = np.maximum(np.array(predicted) ** 2, epsilon)
    return float(np.mean(np.log(predicted_var) + actual_var / predicted_var))


def evaluate_metrics(actual, predicted):
    """
    Computes MAE, RMSE, and QLIKE for given actual and predicted series.
    """
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    ql = qlike(actual, predicted)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "QLIKE": ql
    }


def diebold_mariano(actual, pred1, pred2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy under squared-error loss.
    H0: E[e1^2 - e2^2] = 0.
    Returns (dm_statistic, p_value).
    """
    e1 = (np.array(actual) - np.array(pred1)) ** 2
    e2 = (np.array(actual) - np.array(pred2)) ** 2
    d = e1 - e2
    n = len(d)
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    
    if var_d == 0 or n <= 1:
        return 0.0, 1.0
        
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_val = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    return dm_stat, p_val


# =====================================================
# DATA LOADING & SPLITTING
# =====================================================

data = pd.read_csv(
    "FinPulse_Processed_Data.csv",
    parse_dates=["Date"]
)

print(f"Total dataset shape: {data.shape}")

n = len(data)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = data.iloc[:train_end].copy()
validation = data.iloc[train_end:val_end].copy()
test = data.iloc[val_end:].copy()
train_val = pd.concat([train, validation]).copy()

print("\n--- Data Splits (Chronological) ---")
print(f"Train:      {len(train)} rows ({train['Date'].min().strftime('%Y-%m-%d')} to {train['Date'].max().strftime('%Y-%m-%d')})")
print(f"Validation: {len(validation)} rows ({validation['Date'].min().strftime('%Y-%m-%d')} to {validation['Date'].max().strftime('%Y-%m-%d')})")
print(f"Test:       {len(test)} rows ({test['Date'].min().strftime('%Y-%m-%d')} to {test['Date'].max().strftime('%Y-%m-%d')})")

feature_cols = [
    "Return",
    "Abs_Return",
    "Vol_5",
    "Vol_10",
    "Vol_21",
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Return_Lag5",
    "AbsReturn_Lag1",
    "AbsReturn_Lag2",
    "AbsReturn_Lag5",
    "VIX",
    "VIX_Lag1",
    "VIX_Change_1D",
    "VIX_Change_5D",
    "Parkinson",
    "Volume_Change_1D"
]

X_train = train[feature_cols]
y_train = train["Target"]

X_val = validation[feature_cols]
y_val = validation["Target"]

X_test = test[feature_cols]
y_test = test["Target"]

X_train_val = train_val[feature_cols]
y_train_val = train_val["Target"]

# Feature scalers (fit strictly without lookahead bias)
scaler_tuning = StandardScaler()
X_train_scaled = scaler_tuning.fit_transform(X_train)
X_val_scaled = scaler_tuning.transform(X_val)

final_scaler = StandardScaler()
X_train_val_scaled = final_scaler.fit_transform(X_train_val)
X_test_scaled = final_scaler.transform(X_test)


# =====================================================
# 1. BASELINE MODELS
# =====================================================

# 1a. Naive Persistence Baseline: today's Abs_Return
test["Naive_Prediction"] = test["Abs_Return"]

# 1b. Historical Volatility Baseline: Vol_21
test["Baseline_Vol21"] = test["Vol_21"]

# 1c. Implied Volatility Baseline: VIX / sqrt(252) scaled by sqrt(2/pi)
test["Baseline_VIX"] = (test["VIX"] / np.sqrt(252)) * np.sqrt(2.0 / np.pi)


# =====================================================
# 2. TRADITIONAL TIME-SERIES MODELS (GARCH FAMILY)
# =====================================================
# Standard GARCH forecasts conditional standard deviation sigma.
# Under r ~ N(0, sigma^2), E[|r|] = sigma * sqrt(2/pi) approx 0.79788 * sigma.
# We report both raw sigma and scaled forecasts, caching to CSV.

cache_file = "garch_cache.csv"
if os.path.exists(cache_file):
    print("\nLoading cached GARCH forecasts from garch_cache.csv...")
    cache_df = pd.read_csv(cache_file)
    test["GARCH_raw"] = cache_df["GARCH"].values
    test["TGARCH_raw"] = cache_df["TGARCH"].values
    test["EGARCH_raw"] = cache_df["EGARCH"].values
else:
    print("\nFitting rolling GARCH, TGARCH, and EGARCH models...")
    garch_preds = []
    tgarch_preds = []
    egarch_preds = []
    
    test_start = val_end
    for i in range(test_start, len(data)):
        history = data["Return"].iloc[:i + 1]
        
        # GARCH(1,1)
        m_garch = arch_model(history, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        garch_preds.append(np.sqrt(m_garch.forecast(horizon=1).variance.iloc[-1, 0]))
        
        # TGARCH(1,1) / TARCH
        m_tgarch = arch_model(history, mean="Constant", vol="GARCH", p=1, o=1, q=1, power=1.0, dist="normal").fit(disp="off")
        tgarch_preds.append(np.sqrt(m_tgarch.forecast(horizon=1).variance.iloc[-1, 0]))
        
        # EGARCH(1,1)
        m_egarch = arch_model(history, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="normal").fit(disp="off")
        egarch_preds.append(np.sqrt(m_egarch.forecast(horizon=1).variance.iloc[-1, 0]))
        
    test["GARCH_raw"] = garch_preds
    test["TGARCH_raw"] = tgarch_preds
    test["EGARCH_raw"] = egarch_preds
    
    cache_df = pd.DataFrame({
        "Date": test["Date"].values,
        "GARCH": garch_preds,
        "TGARCH": tgarch_preds,
        "EGARCH": egarch_preds
    })
    cache_df.to_csv(cache_file, index=False)
    print(f"GARCH forecasts cached to {cache_file}.")

# Scaled GARCH versions aligned to expected absolute return
norm_scaling = np.sqrt(2.0 / np.pi)
test["GARCH_scaled"] = test["GARCH_raw"] * norm_scaling
test["TGARCH_scaled"] = test["TGARCH_raw"] * norm_scaling
test["EGARCH_scaled"] = test["EGARCH_raw"] * norm_scaling


# =====================================================
# 3. MACHINE LEARNING MODELS
# =====================================================

# -----------------------------
# 3a. Ridge Regression
# -----------------------------
print("\n--- Tuning Ridge Regression on Validation Set ---")
alphas = [0.01, 0.1, 1, 10, 100]
best_alpha = None
best_ridge_val_mae = float("inf")

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    val_pred = np.clip(ridge.predict(X_val_scaled), 0, None)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"Ridge alpha={alpha:<5} | Validation MAE: {val_mae:.4f}")
    if val_mae < best_ridge_val_mae:
        best_ridge_val_mae = val_mae
        best_alpha = alpha

print(f"Optimal Ridge alpha: {best_alpha} (Val MAE: {best_ridge_val_mae:.4f})")

# Retrain Ridge on train+val with optimal alpha
ridge_model = Ridge(alpha=best_alpha, random_state=42)
ridge_model.fit(X_train_val_scaled, y_train_val)
test["Ridge_Prediction"] = np.clip(ridge_model.predict(X_test_scaled), 0, None)


# -----------------------------
# 3b. Support Vector Regression (SVR - RBF Kernel)
# -----------------------------
print("\n--- Tuning Support Vector Regression (SVR RBF) on Validation Set ---")
Cs = [0.1, 1, 10, 100]
gammas = ["scale", 0.01, 0.1]
epsilons = [0.01, 0.05, 0.1]

best_svr_params = None
best_svr_val_mae = float("inf")

for C in Cs:
    for gamma in gammas:
        for eps in epsilons:
            svr = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps)
            svr.fit(X_train_scaled, y_train)
            val_pred = np.clip(svr.predict(X_val_scaled), 0, None)
            val_mae = mean_absolute_error(y_val, val_pred)
            if val_mae < best_svr_val_mae:
                best_svr_val_mae = val_mae
                best_svr_params = {"C": C, "gamma": gamma, "epsilon": eps}

print(f"Optimal SVR Parameters: {best_svr_params} (Val MAE: {best_svr_val_mae:.4f})")

# Retrain SVR on train+val with optimal parameters
svr_model = SVR(kernel="rbf", **best_svr_params)
svr_model.fit(X_train_val_scaled, y_train_val)
test["SVR_Prediction"] = np.clip(svr_model.predict(X_test_scaled), 0, None)


# -----------------------------
# 3c. XGBoost Regressor
# -----------------------------
print("\n--- Tuning XGBoost Regressor on Validation Set ---")
learning_rates = [0.01, 0.05, 0.1]
max_depths = [2, 3, 4]
best_xgb_params = None
best_xgb_val_mae = float("inf")

for lr in learning_rates:
    for depth in max_depths:
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=lr,
            max_depth=depth,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        )
        xgb.fit(X_train, y_train)
        val_pred = np.clip(xgb.predict(X_val), 0, None)
        val_mae = mean_absolute_error(y_val, val_pred)
        print(f"XGBoost lr={lr:<4} depth={depth} | Validation MAE: {val_mae:.4f}")
        if val_mae < best_xgb_val_mae:
            best_xgb_val_mae = val_mae
            best_xgb_params = {"learning_rate": lr, "max_depth": depth}

print(f"Optimal XGBoost Parameters: {best_xgb_params} (Val MAE: {best_xgb_val_mae:.4f})")

# Retrain XGBoost on train+val with optimal parameters
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=best_xgb_params["learning_rate"],
    max_depth=best_xgb_params["max_depth"],
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1
)
xgb_model.fit(X_train_val, y_train_val)
test["XGBoost_Prediction"] = np.clip(xgb_model.predict(X_test), 0, None)


# =====================================================
# 4. COMPILATION & PERFORMANCE EVALUATION
# =====================================================

all_model_preds = {
    "Naive": test["Naive_Prediction"],
    "Baseline (Vol_21)": test["Baseline_Vol21"],
    "Baseline (VIX scaled)": test["Baseline_VIX"],
    "GARCH(1,1) [raw]": test["GARCH_raw"],
    "TGARCH [raw]": test["TGARCH_raw"],
    "EGARCH [raw]": test["EGARCH_raw"],
    "GARCH(1,1) [scaled]": test["GARCH_scaled"],
    "TGARCH [scaled]": test["TGARCH_scaled"],
    "EGARCH [scaled]": test["EGARCH_scaled"],
    "Ridge": test["Ridge_Prediction"],
    "SVR": test["SVR_Prediction"],
    "XGBoost": test["XGBoost_Prediction"],
}

# Evaluate metrics across all models
metric_records = []
for name, preds in all_model_preds.items():
    res = evaluate_metrics(test["Target"], preds)
    res["Model"] = name
    metric_records.append(res)

all_results_df = pd.DataFrame(metric_records)[["Model", "MAE", "RMSE", "QLIKE"]]

print("\n" + "=" * 60)
print("ALL EVALUATED MODELS (INCLUDING RAW AND SCALED GARCH)")
print("=" * 60)
print(all_results_df.round(4).to_string(index=False))

# Ranking rule:
# Report raw and scaled GARCH, but use scaled versions in final ranking.
ranking_subset = [m for m in all_model_preds.keys() if "[raw]" not in m]
ranking_df = all_results_df[all_results_df["Model"].isin(ranking_subset)].copy()

# Sort by QLIKE first, then MAE as tiebreaker
ranking_df = ranking_df.sort_values(by=["QLIKE", "MAE"]).reset_index(drop=True)
ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))

best_overall_model = ranking_df.iloc[0]["Model"]
second_overall_model = ranking_df.iloc[1]["Model"]
best_overall_preds = all_model_preds[best_overall_model]

# Diebold-Mariano test against best-scoring overall model
dm_stats = []
dm_pvals = []
for m_name in ranking_df["Model"]:
    if m_name == best_overall_model:
        dm_stats.append(0.0)
        dm_pvals.append(1.0)
    else:
        stat, pval = diebold_mariano(test["Target"], all_model_preds[m_name], best_overall_preds)
        dm_stats.append(stat)
        dm_pvals.append(pval)

ranking_df["DM_Stat"] = dm_stats
ranking_df["DM_p_value"] = dm_pvals

print("\n" + "=" * 70)
print("FINAL RANKING TABLE (Ranked by QLIKE, then MAE)")
print("=" * 70)
print(ranking_df.round(4).to_string(index=False))

# Save comparison table
ranking_df.to_csv("model_comparison.csv", index=False)
print("\nSaved ranking table to model_comparison.csv")

# Statistical test between top 2 overall models
dm_top2_stat, dm_top2_p = diebold_mariano(
    test["Target"],
    all_model_preds[second_overall_model],
    all_model_preds[best_overall_model]
)
is_sig_top2 = dm_top2_p < 0.05
print(f"\nDiebold-Mariano Test (Overall #1 {best_overall_model} vs #2 {second_overall_model}):")
print(f"DM Statistic = {dm_top2_stat:.4f}, p-value = {dm_top2_p:.4f}")
print(f"Significant difference at alpha=0.05: {is_sig_top2}")


# =====================================================
# 5. MODEL SELECTION & DEPLOYMENT DECISION
# =====================================================
# Deployment Rule:
# Deploy only models that work from the feature pipeline (Ridge, SVR, XGBoost).
# If top two deployable models are not significantly different (DM p > 0.05),
# prefer the simpler or more interpretable one (Ridge over SVR over XGBoost).

deployable_models = ["Ridge", "SVR", "XGBoost"]
ml_df = ranking_df[ranking_df["Model"].isin(deployable_models)].sort_values(by=["QLIKE", "MAE"]).reset_index(drop=True)

ml_top1 = ml_df.iloc[0]["Model"]
ml_top2 = ml_df.iloc[1]["Model"]

dm_ml_stat, dm_ml_p = diebold_mariano(
    test["Target"],
    all_model_preds[ml_top2],
    all_model_preds[ml_top1]
)
is_sig_ml = dm_ml_p < 0.05

print("\n" + "=" * 60)
print("DEPLOYABLE MACHINE LEARNING MODELS")
print("=" * 60)
print(ml_df[["Model", "MAE", "RMSE", "QLIKE"]].round(4).to_string(index=False))
print(f"\nDM Test between top ML models ({ml_top1} vs {ml_top2}): DM Stat = {dm_ml_stat:.4f}, p-value = {dm_ml_p:.4f}")

# Simplicity priority: Ridge (linear, L2) > SVR (support vectors) > XGBoost (gradient boosted trees)
simplicity_order = {"Ridge": 1, "SVR": 2, "XGBoost": 3}

if not is_sig_ml:
    # Not significantly different: prefer the simpler model
    candidate_pair = [ml_top1, ml_top2]
    candidate_pair.sort(key=lambda m: simplicity_order[m])
    selected_model_name = candidate_pair[0]
    selection_reasoning = (
        f"Among deployable ML models, {ml_top1} (QLIKE: {ml_df.loc[ml_df['Model']==ml_top1, 'QLIKE'].values[0]:.4f}) "
        f"and {ml_top2} (QLIKE: {ml_df.loc[ml_df['Model']==ml_top2, 'QLIKE'].values[0]:.4f}) showed no statistically "
        f"significant difference in predictive accuracy (Diebold-Mariano p-value = {dm_ml_p:.4f} > 0.05). "
        f"Under the decision rule, the more interpretable and stable model, {selected_model_name}, was selected for deployment."
    )
else:
    selected_model_name = ml_top1
    selection_reasoning = (
        f"{selected_model_name} demonstrated statistically significant superior performance over {ml_top2} "
        f"(Diebold-Mariano p-value = {dm_ml_p:.4f} < 0.05) and was selected for deployment."
    )

print("\n" + "=" * 60)
print(f"DEPLOYMENT SELECTION: {selected_model_name}")
print("=" * 60)
print(f"Reasoning:\n{selection_reasoning}")

# Map selected model instance
model_mapping = {
    "Ridge": (ridge_model, True, {"alpha": float(best_alpha), "random_state": 42}),
    "SVR": (svr_model, True, best_svr_params),
    "XGBoost": (xgb_model, False, best_xgb_params)
}

best_estimator, needs_scaling, chosen_params = model_mapping[selected_model_name]
selected_metrics = evaluate_metrics(test["Target"], all_model_preds[selected_model_name])

# Save artifacts for app.py
joblib.dump(best_estimator, "best_model.pkl")
joblib.dump(final_scaler, "scaler.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

# Also maintain backward-compatibility with ridge_model.pkl
joblib.dump(ridge_model, "ridge_model.pkl")

model_info = {
    "model_name": selected_model_name,
    "model_type": selected_model_name,
    "parameters": chosen_params,
    "needs_scaling": needs_scaling,
    "test_metrics": {k: round(v, 4) for k, v in selected_metrics.items()},
    "selection_reasoning": selection_reasoning,
    "best_overall_model": best_overall_model,
    "best_overall_qlike": round(float(ranking_df.iloc[0]["QLIKE"]), 4)
}

with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=4)

print("Saved best_model.pkl, scaler.pkl, feature_cols.pkl, and model_info.json successfully.")


# =====================================================
# 6. VISUALIZATION EXPORTS
# =====================================================

# 6a. Actual vs Predicted for Top 3 Models
top3_names = ranking_df["Model"].iloc[:3].tolist()
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(test["Date"], test["Target"], label="Actual (|Return|)", color="#111827", linewidth=1.5, alpha=0.85)

colors = ["#2563EB", "#059669", "#D97706"]
styles = ["-", "--", "-."]
for idx, name in enumerate(top3_names):
    ax.plot(
        test["Date"],
        all_model_preds[name],
        label=f"{name} (Rank #{idx+1})",
        color=colors[idx % len(colors)],
        linestyle=styles[idx % len(styles)],
        linewidth=1.3,
        alpha=0.9
    )

ax.set_title("Actual vs. Predicted Next-Session Volatility (Top 3 Ranked Models)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Trading Date", fontsize=11)
ax.set_ylabel("Volatility Magnitude (%)", fontsize=11)
ax.legend(loc="upper left", frameon=True, framealpha=0.9)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("actual_vs_predicted_top3.png", dpi=300)
plt.close()
print("Saved plot to actual_vs_predicted_top3.png")

# 6b. Bar Chart of Model Metrics
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

plot_df = ranking_df.copy()
# Highlight the selected model
bar_colors = ["#3B82F6" if m == selected_model_name else "#9CA3AF" for m in plot_df["Model"]]

# MAE
axes[0].barh(plot_df["Model"], plot_df["MAE"], color=bar_colors)
axes[0].set_title("Mean Absolute Error (MAE)", fontweight="bold")
axes[0].set_xlabel("MAE (%)")
axes[0].invert_yaxis()
axes[0].grid(axis="x", linestyle=":", alpha=0.6)

# RMSE
axes[1].barh(plot_df["Model"], plot_df["RMSE"], color=bar_colors)
axes[1].set_title("Root Mean Squared Error (RMSE)", fontweight="bold")
axes[1].set_xlabel("RMSE (%)")
axes[1].invert_yaxis()
axes[1].grid(axis="x", linestyle=":", alpha=0.6)

# QLIKE (Cap display for Naive / SVR if needed or use log scale)
axes[2].barh(plot_df["Model"], plot_df["QLIKE"], color=bar_colors)
axes[2].set_title("Quasi-Likelihood (QLIKE - Log Scale)", fontweight="bold")
axes[2].set_xlabel("QLIKE Loss (Log Scale)")
axes[2].set_xscale("log")
axes[2].invert_yaxis()
axes[2].grid(axis="x", linestyle=":", alpha=0.6)

plt.suptitle(f"Model Forecasting Performance Benchmark (Highlighted: Selected Model '{selected_model_name}')", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("model_metrics_barchart.png", dpi=300)
plt.close()
print("Saved metrics bar chart to model_metrics_barchart.png")
print("\nmodels.py execution completed successfully!")