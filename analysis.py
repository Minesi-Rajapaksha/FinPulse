import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

os.makedirs("figures", exist_ok=True)
np.random.seed(42)

# Load data and splits
data = pd.read_csv("FinPulse_Processed_Data.csv", parse_dates=["Date"])
n = len(data)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = data.iloc[:train_end].copy()
val = data.iloc[train_end:val_end].copy()
test = data.iloc[val_end:].copy()
train_val = data.iloc[:val_end].copy()

test_dates = test["Date"].values
actual = test["Target"].values
actual_var = np.maximum(actual ** 2, 1e-8)
norm_scaling = np.sqrt(2.0 / np.pi)

# Load predictions_test.csv from models.py
preds_df = pd.read_csv("predictions_test.csv", parse_dates=["Date"])

def calc_qlike(point_pred, is_sigma=False):
    if is_sigma:
        s = np.maximum(point_pred, 0.05)
    else:
        s = np.maximum(point_pred / norm_scaling, 0.05)
    v = s ** 2
    return float(np.mean(np.log(v) + actual_var / v))

def calc_metrics(act, pred, is_sigma=False):
    if is_sigma:
        pt = pred * norm_scaling
        sig = pred
    else:
        pt = pred
        sig = pred / norm_scaling
    mae = float(mean_absolute_error(act, pt))
    rmse = float(np.sqrt(mean_squared_error(act, pt)))
    sig_f = np.maximum(sig, 0.05)
    var_f = sig_f ** 2
    act_v = np.maximum(np.array(act) ** 2, 1e-8)
    ql = float(np.mean(np.log(var_f) + act_v / var_f))
    return mae, rmse, ql


# =====================================================
# E1. FEATURE-SET ABLATION ANALYSIS
# =====================================================
print("\n--- Executing E1: Feature-Set Ablation Analysis ---")
return_feats = [
    "Return", "Abs_Return", "Vol_5", "Vol_10", "Vol_21",
    "Return_Lag1", "Return_Lag2", "Return_Lag3", "Return_Lag5",
    "AbsReturn_Lag1", "AbsReturn_Lag2", "AbsReturn_Lag5",
    "Volume_Change_1D"
]
parkinson_feats = ["Parkinson"]
vix_feats = ["VIX", "VIX_Lag1", "VIX_Change_1D", "VIX_Change_5D"]

ablation_sets = {
    "A (Return History Only)": return_feats,
    "B (Return + Parkinson)": return_feats + parkinson_feats,
    "C (Return + VIX Features)": return_feats + vix_feats,
    "D (All 18 Features)": return_feats + parkinson_feats + vix_feats
}

feature_cols = return_feats + parkinson_feats + vix_feats

ablation_records = []

# Baseline Traditional Comparisons
egarch_mae, egarch_rmse, egarch_ql = calc_metrics(actual, preds_df["EGARCH"].values)
garch_mae, garch_rmse, garch_ql = calc_metrics(actual, preds_df["GARCH(1,1)"].values)
har_mae, har_rmse, har_ql = calc_metrics(actual, preds_df["HAR"].values)

ablation_records.append({"Feature_Set": "Univariate GARCH(1,1)", "Model": "GARCH(1,1)", "MAE": round(garch_mae, 4), "RMSE": round(garch_rmse, 4), "QLIKE": round(garch_ql, 4)})
ablation_records.append({"Feature_Set": "Univariate EGARCH(1,1)", "Model": "EGARCH(1,1)", "MAE": round(egarch_mae, 4), "RMSE": round(egarch_rmse, 4), "QLIKE": round(egarch_ql, 4)})
ablation_records.append({"Feature_Set": "HAR-RV (OLS)", "Model": "HAR", "MAE": round(har_mae, 4), "RMSE": round(har_rmse, 4), "QLIKE": round(har_ql, 4)})

for set_name, cols in ablation_sets.items():
    # Ridge
    sc = StandardScaler()
    X_tv_s = sc.fit_transform(train_val[cols])
    X_te_s = sc.transform(test[cols])
    r = Ridge(alpha=10.0, random_state=42).fit(X_tv_s, train_val["Target"])
    p_r = np.clip(r.predict(X_te_s), 0, None)
    mae_r, rmse_r, ql_r = calc_metrics(actual, p_r)
    ablation_records.append({
        "Feature_Set": set_name,
        "Model": "Ridge",
        "MAE": round(mae_r, 4),
        "RMSE": round(rmse_r, 4),
        "QLIKE": round(ql_r, 4)
    })
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=4, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=1)
    xgb.fit(train_val[cols], train_val["Target"])
    p_x = np.clip(xgb.predict(test[cols]), 0, None)
    mae_x, rmse_x, ql_x = calc_metrics(actual, p_x)
    ablation_records.append({
        "Feature_Set": set_name,
        "Model": "XGBoost",
        "MAE": round(mae_x, 4),
        "RMSE": round(rmse_x, 4),
        "QLIKE": round(ql_x, 4)
    })

ablation_df = pd.DataFrame(ablation_records)
ablation_df.to_csv("ablation_results.csv", index=False)
print("Saved ablation results to ablation_results.csv.")

# Figure 15: Grouped Bar Chart of Ablation Results
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
set_labels = list(ablation_sets.keys())
x_pos = np.arange(len(set_labels))
width = 0.35

ridge_maes = [ablation_df[(ablation_df["Feature_Set"]==s) & (ablation_df["Model"]=="Ridge")]["MAE"].values[0] for s in set_labels]
xgb_maes = [ablation_df[(ablation_df["Feature_Set"]==s) & (ablation_df["Model"]=="XGBoost")]["MAE"].values[0] for s in set_labels]

ridge_rmses = [ablation_df[(ablation_df["Feature_Set"]==s) & (ablation_df["Model"]=="Ridge")]["RMSE"].values[0] for s in set_labels]
xgb_rmses = [ablation_df[(ablation_df["Feature_Set"]==s) & (ablation_df["Model"]=="XGBoost")]["RMSE"].values[0] for s in set_labels]

ridge_qlikes = [ablation_df[(ablation_df["Feature_Set"]==s) & (ablation_df["Model"]=="Ridge")]["QLIKE"].values[0] for s in set_labels]
xgb_qlikes = [ablation_df[(ablation_df["Feature_Set"]==s) & (ablation_df["Model"]=="XGBoost")]["QLIKE"].values[0] for s in set_labels]

# MAE
axes[0].bar(x_pos - width/2, ridge_maes, width, label="Ridge", color="#2563EB", alpha=0.85)
axes[0].bar(x_pos + width/2, xgb_maes, width, label="XGBoost", color="#10B981", alpha=0.85)
axes[0].axhline(egarch_mae, color="#DC2626", linestyle="--", label=f"EGARCH ({egarch_mae:.3f})")
axes[0].set_title("MAE Across Feature Subsets (%)", fontweight="bold")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(["A (Returns)", "B (+Parkinson)", "C (+VIX)", "D (All 18)"], rotation=15)
axes[0].grid(axis="y", linestyle=":", alpha=0.6)
axes[0].legend(frameon=True, fontsize=9)

# RMSE
axes[1].bar(x_pos - width/2, ridge_rmses, width, label="Ridge", color="#2563EB", alpha=0.85)
axes[1].bar(x_pos + width/2, xgb_rmses, width, label="XGBoost", color="#10B981", alpha=0.85)
axes[1].axhline(egarch_rmse, color="#DC2626", linestyle="--", label=f"EGARCH ({egarch_rmse:.3f})")
axes[1].set_title("RMSE Across Feature Subsets (%)", fontweight="bold")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(["A (Returns)", "B (+Parkinson)", "C (+VIX)", "D (All 18)"], rotation=15)
axes[1].grid(axis="y", linestyle=":", alpha=0.6)
axes[1].legend(frameon=True, fontsize=9)

# QLIKE
axes[2].bar(x_pos - width/2, ridge_qlikes, width, label="Ridge", color="#2563EB", alpha=0.85)
axes[2].bar(x_pos + width/2, xgb_qlikes, width, label="XGBoost", color="#10B981", alpha=0.85)
axes[2].axhline(egarch_ql, color="#DC2626", linestyle="--", label=f"EGARCH ({egarch_ql:.3f})")
axes[2].set_title("Corrected QLIKE Across Feature Subsets", fontweight="bold")
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(["A (Returns)", "B (+Parkinson)", "C (+VIX)", "D (All 18)"], rotation=15)
axes[2].grid(axis="y", linestyle=":", alpha=0.6)
axes[2].legend(frameon=True, fontsize=9)

plt.suptitle("Figure 15: Feature-Set Ablation Analysis: Isolating the Information Advantage of VIX and Parkinson Volatility", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/fig15_ablation_analysis.png", dpi=200)
plt.close()


# =====================================================
# E2. SHOCK & REGIME ANALYSIS
# =====================================================
print("\n--- Executing E2: Historical Stress & Market Regime Analysis ---")

all_eval_models = [
    "Naive", "Baseline (Vol_21)", "Baseline (VIX)", "GARCH(1,1)",
    "TGARCH", "EGARCH", "HAR", "Ridge", "SVR", "XGBoost"
]

# (i) April 2025 Stress Window
april_mask = (test["Date"] >= "2025-04-01") & (test["Date"] <= "2025-04-30")
n_april = int(april_mask.sum())
n_non_april = int((~april_mask).sum())

shock_rows = []
for m in all_eval_models:
    p = preds_df[m].values
    act = actual
    
    sq_err_total = np.sum((act - p) ** 2)
    sq_err_april = np.sum((act[april_mask] - p[april_mask]) ** 2)
    sq_err_non_april = np.sum((act[~april_mask] - p[~april_mask]) ** 2)
    
    april_share = (sq_err_april / sq_err_total) * 100.0
    
    mae_all = mean_absolute_error(act, p)
    rmse_all = np.sqrt(mean_squared_error(act, p))
    
    mae_apr = mean_absolute_error(act[april_mask], p[april_mask])
    rmse_apr = np.sqrt(mean_squared_error(act[april_mask], p[april_mask]))
    
    mae_non = mean_absolute_error(act[~april_mask], p[~april_mask])
    rmse_non = np.sqrt(mean_squared_error(act[~april_mask], p[~april_mask]))
    
    shock_rows.append({
        "Model": m,
        "Total_RMSE": round(rmse_all, 4),
        "April_RMSE": round(rmse_apr, 4),
        "Non_April_RMSE": round(rmse_non, 4),
        "April_SqErr_Share_Pct": round(april_share, 2),
        "Total_MAE": round(mae_all, 4),
        "April_MAE": round(mae_apr, 4),
        "Non_April_MAE": round(mae_non, 4)
    })

shock_df = pd.DataFrame(shock_rows)
shock_df.to_csv("shock_regime_results.csv", index=False)
print("Saved April shock analysis to shock_regime_results.csv.")

# (ii) VIX Regimes using prior day's VIX (VIX_Lag1)
vix_prev = test["VIX_Lag1"].values
low_mask = vix_prev < 15.0
med_mask = (vix_prev >= 15.0) & (vix_prev <= 25.0)
high_mask = vix_prev > 25.0

regime_rows = []
for m in all_eval_models:
    p = preds_df[m].values
    for reg_name, reg_mask, n_days in [
        ("Low Volatility (VIX < 15)", low_mask, int(low_mask.sum())),
        ("Medium Volatility (15 <= VIX <= 25)", med_mask, int(med_mask.sum())),
        ("High Volatility / Panic (VIX > 25)", high_mask, int(high_mask.sum()))
    ]:
        if n_days > 0:
            mae_reg = float(mean_absolute_error(actual[reg_mask], p[reg_mask]))
            rmse_reg = float(np.sqrt(mean_squared_error(actual[reg_mask], p[reg_mask])))
            sig_reg = np.maximum(p[reg_mask] / norm_scaling, 0.05)
            ql_reg = float(np.mean(np.log(sig_reg**2) + (actual[reg_mask]**2)/(sig_reg**2)))
            regime_rows.append({
                "Model": m,
                "Regime": reg_name,
                "Days": n_days,
                "MAE": round(mae_reg, 4),
                "RMSE": round(rmse_reg, 4),
                "QLIKE": round(ql_reg, 4)
            })

regime_df = pd.DataFrame(regime_rows)
regime_df.to_csv("vix_regime_results.csv", index=False)
print("Saved VIX regime performance to vix_regime_results.csv.")

# (iii) Rolling 60-Day RMSE Plot (Figure 16)
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Top panel: Rolling 60-day RMSE
window = 60
roll_dates = test["Date"].iloc[window-1:].values
models_for_roll = ["Ridge", "XGBoost", "EGARCH", "GARCH(1,1)", "Baseline (VIX)"]
colors_roll = {"Ridge": "#2563EB", "XGBoost": "#10B981", "EGARCH": "#DC2626", "GARCH(1,1)": "#9CA3AF", "Baseline (VIX)": "#7C3AED"}

for m in models_for_roll:
    sq_errs = (actual - preds_df[m].values) ** 2
    roll_rmse = pd.Series(sq_errs).rolling(window).mean().dropna().apply(np.sqrt).values
    axes[0].plot(roll_dates, roll_rmse, label=m, color=colors_roll[m], linewidth=1.5)

axes[0].axvspan(pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30"), color="#FEE2E2", alpha=0.5, label="April 2025 Shock")
axes[0].set_title("Rolling 60-Day Out-of-Sample RMSE Across Models", fontweight="bold", fontsize=11)
axes[0].set_ylabel("Rolling RMSE (%)")
axes[0].legend(loc="upper left", frameon=True)
axes[0].grid(True, linestyle=":", alpha=0.5)

# Bottom panel: SPY Return and VIX during the test period
axes[1].plot(test["Date"], test["Return"], color="#1F2937", label="SPY Daily Return (%)", linewidth=1.0, alpha=0.8)
axes[1].plot(test["Date"], test["VIX"]/10.0, color="#7C3AED", label="VIX / 10", linewidth=1.2, linestyle="--")
axes[1].axhline(0, color="red", linestyle=":")
axes[1].axvspan(pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30"), color="#FEE2E2", alpha=0.5)
axes[1].set_title("Market Context: Daily SPY Returns and CBOE VIX Level", fontweight="bold", fontsize=11)
axes[1].set_ylabel("Percentage / Scaled VIX")
axes[1].legend(loc="lower left", frameon=True)
axes[1].grid(True, linestyle=":", alpha=0.5)

plt.suptitle("Figure 16: Out-of-Sample Volatility Dynamics: Rolling 60-Day Forecast Stability and Shock Persistence", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/fig16_regime_rolling_rmse.png", dpi=200)
plt.close()

# (iv) Top 10 Largest Forecast Errors per Model
top_err_records = []
for m in ["Ridge", "XGBoost", "EGARCH", "GARCH(1,1)"]:
    err_df = pd.DataFrame({
        "Date": test["Date"].dt.strftime("%Y-%m-%d"),
        "Model": m,
        "Actual": actual,
        "Forecast": preds_df[m].values,
        "Abs_Error": np.abs(actual - preds_df[m].values),
        "Squared_Error": (actual - preds_df[m].values) ** 2,
        "VIX": test["VIX"].values
    }).sort_values(by="Squared_Error", ascending=False).head(10)
    top_err_records.append(err_df)

top10_df = pd.concat(top_err_records).reset_index(drop=True)
top10_df.to_csv("top10_errors.csv", index=False)
print("Saved largest test errors to top10_errors.csv.")


# =====================================================
# E3. MULTI-HORIZON FORECASTING (1-DAY, 5-DAY, 10-DAY)
# =====================================================
print("\n--- Executing E3: Multi-Horizon Volatility Forecasting (1, 5, 10 Days) ---")

horizons = [1, 5, 10]
horizon_models = ["Ridge", "XGBoost", "HAR", "EGARCH"]
horizon_results = []

# Prepare rolling targets for 1, 5, 10 days strictly without leakage
# Target_H is the average daily absolute return over sessions t+1 through t+H
for H in horizons:
    data_h = data.copy()
    # Compute future average absolute return over next H days
    roll_forward = data_h["Abs_Return"].iloc[::-1].rolling(H).mean().iloc[::-1].shift(-1)
    data_h[f"Target_{H}"] = roll_forward
    
    # Clean tail rows where forward window exceeds dataset
    valid_mask = data_h[f"Target_{H}"].notnull()
    dh_clean = data_h[valid_mask].copy()
    
    n_h = len(dh_clean)
    val_end_h = int(n * 0.80)
    train_val_h = dh_clean.iloc[:val_end_h]
    test_h = dh_clean.iloc[val_end_h:]
    
    act_h = test_h[f"Target_{H}"].values
    
    # 1. Ridge
    sc_h = StandardScaler()
    X_tv_h = sc_h.fit_transform(train_val_h[feature_cols])
    X_te_h = sc_h.transform(test_h[feature_cols])
    r_h = Ridge(alpha=10.0, random_state=42).fit(X_tv_h, train_val_h[f"Target_{H}"])
    pred_r_h = np.clip(r_h.predict(X_te_h), 0, None)
    mae_r, rmse_r, ql_r = calc_metrics(act_h, pred_r_h)
    horizon_results.append({"Horizon": f"{H}-Day", "Horizon_Days": H, "Model": "Ridge", "MAE": round(mae_r, 4), "RMSE": round(rmse_r, 4), "QLIKE": round(ql_r, 4)})
    
    # 2. XGBoost
    xgb_h = XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=4, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=1)
    xgb_h.fit(train_val_h[feature_cols], train_val_h[f"Target_{H}"])
    pred_x_h = np.clip(xgb_h.predict(test_h[feature_cols]), 0, None)
    mae_x, rmse_x, ql_x = calc_metrics(act_h, pred_x_h)
    horizon_results.append({"Horizon": f"{H}-Day", "Horizon_Days": H, "Model": "XGBoost", "MAE": round(mae_x, 4), "RMSE": round(rmse_x, 4), "QLIKE": round(ql_x, 4)})
    
    # 3. HAR
    # HAR features: d, w, m
    abs_vals = dh_clean["Abs_Return"].values
    har_d_h = abs_vals
    har_w_h = pd.Series(abs_vals).rolling(5, min_periods=1).mean().values
    har_m_h = pd.Series(abs_vals).rolling(21, min_periods=1).mean().values
    X_har_h = np.column_stack([har_d_h, har_w_h, har_m_h])
    
    har_reg_h = LinearRegression().fit(X_har_h[:val_end_h], train_val_h[f"Target_{H}"])
    pred_har_h = np.clip(har_reg_h.predict(X_har_h[val_end_h:val_end_h+len(test_h)]), 0, None)
    mae_h, rmse_h, ql_h = calc_metrics(act_h, pred_har_h)
    horizon_results.append({"Horizon": f"{H}-Day", "Horizon_Days": H, "Model": "HAR", "MAE": round(mae_h, 4), "RMSE": round(rmse_h, 4), "QLIKE": round(ql_h, 4)})
    
    # 4. EGARCH forecast: standard 1-step annualized or rolling
    # For H=1, use test 1-step EGARCH. For H>1, EGARCH multi-step variance mean
    if H == 1:
        pred_eg = preds_df["EGARCH"].values[:len(act_h)]
    else:
        # Expanding rolling EGARCH multi-step average
        pred_eg = pd.Series(preds_df["EGARCH"].values).rolling(H, min_periods=1).mean().values[:len(act_h)]
    mae_e, rmse_e, ql_e = calc_metrics(act_h, pred_eg)
    horizon_results.append({"Horizon": f"{H}-Day", "Horizon_Days": H, "Model": "EGARCH", "MAE": round(mae_e, 4), "RMSE": round(rmse_e, 4), "QLIKE": round(ql_e, 4)})

horizon_df = pd.DataFrame(horizon_results)
horizon_df.to_csv("horizon_results.csv", index=False)
print("Saved multi-horizon results to horizon_results.csv.")

# Figure 18: Multi-Horizon Line Chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
h_vals = [1, 5, 10]
m_colors = {"Ridge": "#2563EB", "XGBoost": "#10B981", "HAR": "#F59E0B", "EGARCH": "#DC2626"}

for m in ["Ridge", "XGBoost", "HAR", "EGARCH"]:
    sub = horizon_df[horizon_df["Model"] == m]
    axes[0].plot(sub["Horizon_Days"], sub["MAE"], marker="o", label=m, color=m_colors[m], linewidth=1.5)
    axes[1].plot(sub["Horizon_Days"], sub["RMSE"], marker="s", label=m, color=m_colors[m], linewidth=1.5)
    axes[2].plot(sub["Horizon_Days"], sub["QLIKE"], marker="^", label=m, color=m_colors[m], linewidth=1.5)

axes[0].set_title("Mean Absolute Error (MAE %) by Horizon", fontweight="bold")
axes[0].set_xlabel("Forecast Horizon (Trading Days)")
axes[0].set_ylabel("MAE (%)")
axes[0].set_xticks(h_vals)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(frameon=True)

axes[1].set_title("Root Mean Squared Error (RMSE %) by Horizon", fontweight="bold")
axes[1].set_xlabel("Forecast Horizon (Trading Days)")
axes[1].set_ylabel("RMSE (%)")
axes[1].set_xticks(h_vals)
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(frameon=True)

axes[2].set_title("QLIKE Loss by Horizon", fontweight="bold")
axes[2].set_xlabel("Forecast Horizon (Trading Days)")
axes[2].set_ylabel("QLIKE")
axes[2].set_xticks(h_vals)
axes[2].grid(True, linestyle=":", alpha=0.6)
axes[2].legend(frameon=True)

plt.suptitle("Figure 18: Multi-Horizon Forecasting Accuracy: 1-Day, 5-Day, and 10-Day Term Structure", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/fig18_multi_horizon.png", dpi=200)
plt.close()


# =====================================================
# E4. STATISTICAL TESTS: DIEBOLD-MARIANO (HLN-CORRECTED)
# =====================================================
print("\n--- Executing E4: Diebold-Mariano Tests & Heatmap ---")

def dm_test_hln(act, p1, p2, loss_type="se"):
    act = np.array(act)
    p1 = np.array(p1)
    p2 = np.array(p2)
    norm_s = np.sqrt(2.0 / np.pi)
    
    if loss_type == "se":
        d = (act - p1) ** 2 - (act - p2) ** 2
    elif loss_type == "qlike":
        s1 = np.maximum(p1 / norm_s, 0.05)
        s2 = np.maximum(p2 / norm_s, 0.05)
        act_var = np.maximum(act ** 2, 1e-8)
        d = (np.log(s1**2) + act_var/(s1**2)) - (np.log(s2**2) + act_var/(s2**2))
    else:
        raise ValueError("Unknown loss type")
        
    n = len(d)
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    
    if var_d <= 1e-12 or n <= 1:
        return 0.0, 1.0
        
    base_stat = mean_d / np.sqrt(var_d / n)
    hln_factor = np.sqrt((n - 1) / n)
    stat = base_stat * hln_factor
    p_val = float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))
    return float(stat), p_val

# Key required comparison pairs
target_pairs = [
    ("EGARCH", "GARCH(1,1)"),
    ("TGARCH", "GARCH(1,1)"),
    ("Ridge", "GARCH(1,1)"),
    ("Ridge", "HAR"),
    ("TGARCH", "Ridge"),
    ("EGARCH", "Ridge"),
    ("Ridge", "XGBoost"),
    ("EGARCH", "TGARCH"),
    ("EGARCH", "Baseline (VIX)"),
    ("Ridge", "Baseline (VIX)"),
    ("XGBoost", "Baseline (VIX)"),
    ("EGARCH", "Naive"),
    ("Ridge", "Naive"),
    ("XGBoost", "Naive")
]

dm_records = []
for m1, m2 in target_pairs:
    p1 = preds_df[m1].values
    p2 = preds_df[m2].values
    
    # Squared Error Loss
    stat_se, pval_se = dm_test_hln(actual, p1, p2, loss_type="se")
    fav_se = m1 if stat_se < 0 and pval_se < 0.05 else (m2 if stat_se > 0 and pval_se < 0.05 else "No Difference")
    dm_records.append({
        "Model_1": m1, "Model_2": m2, "Loss_Function": "Squared Error (MSE)",
        "DM_Stat": round(stat_se, 4), "p_value": round(pval_se, 4), "Significant_05": pval_se < 0.05,
        "Favored_Model": fav_se
    })
    
    # QLIKE Loss
    stat_ql, pval_ql = dm_test_hln(actual, p1, p2, loss_type="qlike")
    fav_ql = m1 if stat_ql < 0 and pval_ql < 0.05 else (m2 if stat_ql > 0 and pval_ql < 0.05 else "No Difference")
    dm_records.append({
        "Model_1": m1, "Model_2": m2, "Loss_Function": "QLIKE Loss",
        "DM_Stat": round(stat_ql, 4), "p_value": round(pval_ql, 4), "Significant_05": pval_ql < 0.05,
        "Favored_Model": fav_ql
    })

dm_df = pd.DataFrame(dm_records)
dm_df.to_csv("dm_tests.csv", index=False)
print("Saved Diebold-Mariano tests to dm_tests.csv.")

# Full pairwise p-value matrix for heatmap (Figure 17)
heatmap_models = ["Naive", "GARCH(1,1)", "TGARCH", "EGARCH", "HAR", "Ridge", "XGBoost", "Baseline (VIX)"]
n_hm = len(heatmap_models)
p_matrix_se = np.ones((n_hm, n_hm))
p_matrix_ql = np.ones((n_hm, n_hm))

for i, m1 in enumerate(heatmap_models):
    for j, m2 in enumerate(heatmap_models):
        if i != j:
            _, p_se = dm_test_hln(actual, preds_df[m1].values, preds_df[m2].values, loss_type="se")
            _, p_ql = dm_test_hln(actual, preds_df[m1].values, preds_df[m2].values, loss_type="qlike")
            p_matrix_se[i, j] = p_se
            p_matrix_ql[i, j] = p_ql

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Squared Error p-value matrix
im0 = axes[0].imshow(p_matrix_se, cmap="Blues_r", vmin=0, vmax=0.10)
axes[0].set_title("Pairwise DM Test p-values (Squared-Error Loss)", fontweight="bold")
axes[0].set_xticks(np.arange(n_hm))
axes[0].set_yticks(np.arange(n_hm))
axes[0].set_xticklabels(heatmap_models, rotation=45, ha="right")
axes[0].set_yticklabels(heatmap_models)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

for i in range(n_hm):
    for j in range(n_hm):
        val = p_matrix_se[i, j]
        text_col = "white" if val < 0.05 else "black"
        axes[0].text(j, i, f"{val:.3f}" if i != j else "-", ha="center", va="center", color=text_col, fontsize=8)

# QLIKE p-value matrix
im1 = axes[1].imshow(p_matrix_ql, cmap="Blues_r", vmin=0, vmax=0.10)
axes[1].set_title("Pairwise DM Test p-values (QLIKE Loss)", fontweight="bold")
axes[1].set_xticks(np.arange(n_hm))
axes[1].set_yticks(np.arange(n_hm))
axes[1].set_xticklabels(heatmap_models, rotation=45, ha="right")
axes[1].set_yticklabels(heatmap_models)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

for i in range(n_hm):
    for j in range(n_hm):
        val = p_matrix_ql[i, j]
        text_col = "white" if val < 0.05 else "black"
        axes[1].text(j, i, f"{val:.3f}" if i != j else "-", ha="center", va="center", color=text_col, fontsize=8)

plt.suptitle("Figure 17: Pairwise Diebold-Mariano Statistical Significance Matrix (Harvey-Leybourne-Newbold Correction)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/fig17_dm_heatmap.png", dpi=200)
plt.close()


# =====================================================
# E5. VALUE-AT-RISK (VaR) BACKTESTING
# =====================================================
print("\n--- Executing E5: 95% and 99% VaR Backtesting ---")

def var_backtest(actual_returns, sigma_forecasts, alpha=0.05):
    """
    Kupiec POF test and Christoffersen independence test for 1-day VaR.
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    var_threshold = z_alpha * sigma_forecasts
    hits = (np.abs(actual_returns) > var_threshold).astype(int)
    
    n_total = len(hits)
    x_hits = int(np.sum(hits))
    failure_rate = x_hits / n_total
    
    # Kupiec POF Likelihood Ratio test
    p_hat = failure_rate
    if x_hits == 0:
        lr_pof = -2.0 * n_total * np.log(1.0 - alpha)
    elif x_hits == n_total:
        lr_pof = -2.0 * n_total * np.log(alpha)
    else:
        log_num = (n_total - x_hits) * np.log(1.0 - alpha) + x_hits * np.log(alpha)
        log_den = (n_total - x_hits) * np.log(1.0 - p_hat) + x_hits * np.log(p_hat)
        lr_pof = -2.0 * (log_num - log_den)
    p_pof = float(1.0 - stats.chi2.cdf(lr_pof, df=1))
    
    # Christoffersen Independence Test
    hit_pairs = list(zip(hits[:-1], hits[1:]))
    n00 = sum(1 for (a, b) in hit_pairs if a == 0 and b == 0)
    n01 = sum(1 for (a, b) in hit_pairs if a == 0 and b == 1)
    n10 = sum(1 for (a, b) in hit_pairs if a == 1 and b == 0)
    n11 = sum(1 for (a, b) in hit_pairs if a == 1 and b == 1)
    
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0
    
    if (pi0 == 0 and pi1 == 0) or (pi0 == 1 and pi1 == 1) or pi == 0:
        lr_ind = 0.0
    else:
        term1 = (n00 + n10) * np.log(1 - pi + 1e-12) + (n01 + n11) * np.log(pi + 1e-12)
        term2 = (n00 * np.log(1 - pi0 + 1e-12) + n01 * np.log(pi0 + 1e-12) +
                 n10 * np.log(1 - pi1 + 1e-12) + n11 * np.log(pi1 + 1e-12))
        lr_ind = -2.0 * (term1 - term2)
    p_ind = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
    
    lr_cc = lr_pof + lr_ind
    p_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
    
    return {
        "Confidence_Level": f"{(1-alpha)*100:.0f}%",
        "Expected_Failures": round(alpha * n_total, 1),
        "Actual_Failures": x_hits,
        "Failure_Rate_Pct": round(failure_rate * 100, 2),
        "Kupiec_LR_POF": round(float(lr_pof), 4),
        "Kupiec_p_val": round(p_pof, 4),
        "Kupiec_Pass": bool(p_pof >= 0.05),
        "Christoffersen_LR_Ind": round(float(lr_ind), 4),
        "Christoffersen_p_val": round(p_ind, 4),
        "Independence_Pass": bool(p_ind >= 0.05),
        "Conditional_Coverage_p_val": round(p_cc, 4)
    }

var_records = []
for model_name, sig_f in [
    ("Ridge", np.maximum(preds_df["Ridge"].values / norm_scaling, 0.05)),
    ("EGARCH", np.maximum(preds_df["EGARCH"].values / norm_scaling, 0.05)),
    ("Baseline (VIX)", np.maximum(preds_df["Baseline (VIX)"].values / norm_scaling, 0.05))
]:
    for alpha_lvl in [0.05, 0.01]:
        res_v = var_backtest(actual, sig_f, alpha=alpha_lvl)
        res_v["Model"] = model_name
        var_records.append(res_v)

var_df = pd.DataFrame(var_records)
var_df.to_csv("var_backtest.csv", index=False)
print("Saved Value-at-Risk backtest results to var_backtest.csv.")
print("\nanalysis.py execution completed successfully!")
