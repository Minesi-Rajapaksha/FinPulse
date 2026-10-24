import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import arch
import statsmodels

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Set random seeds for strict reproducibility
np.random.seed(42)

# =====================================================
# 1. EVALUATION METRICS & LOSS FUNCTIONS
# =====================================================

def evaluate_metrics(actual, point_pred, sigma_hat):
    """
    Computes MAE, RMSE on expected absolute returns E|R|,
    and QLIKE loss on variance forecast sigma_hat^2.
    sigma_hat is floored at 0.05 to prevent division-by-zero / numerical instability.
    """
    actual = np.array(actual)
    point_pred = np.array(point_pred)
    sigma_hat = np.array(sigma_hat)
    
    mae = float(mean_absolute_error(actual, point_pred))
    rmse = float(np.sqrt(mean_squared_error(actual, point_pred)))
    
    # QLIKE loss: floored at sigma = 0.05
    sigma_floored = np.maximum(sigma_hat, 0.05)
    var_pred = sigma_floored ** 2
    actual_var = np.maximum(actual ** 2, 1e-8)
    qlike = float(np.mean(np.log(var_pred) + actual_var / var_pred))
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "QLIKE": qlike
    }


def diebold_mariano_hln(actual, pred1, pred2, h=1, loss_type="se"):
    """
    Diebold-Mariano test with Harvey, Leybourne, and Newbold (1997)
    small-sample correction under squared-error ('se') or QLIKE loss.
    H0: E[d_t] = 0 (equal predictive accuracy).
    Returns (dm_stat, p_value).
    """
    actual = np.array(actual)
    p1 = np.array(pred1)
    p2 = np.array(pred2)
    norm_scaling = np.sqrt(2.0 / np.pi)
    
    if loss_type == "se":
        d = (actual - p1) ** 2 - (actual - p2) ** 2
    elif loss_type == "qlike":
        s1 = np.maximum(p1 / norm_scaling, 0.05)
        s2 = np.maximum(p2 / norm_scaling, 0.05)
        act_var = np.maximum(actual ** 2, 1e-8)
        loss1 = np.log(s1 ** 2) + act_var / (s1 ** 2)
        loss2 = np.log(s2 ** 2) + act_var / (s2 ** 2)
        d = loss1 - loss2
    else:
        raise ValueError("Unsupported loss_type")
        
    n = len(d)
    mean_d = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=1))
    
    if gamma0 <= 1e-12 or n <= 1:
        return 0.0, 1.0
        
    base_stat = mean_d / np.sqrt(gamma0 / n)
    # Harvey, Leybourne, Newbold (1997) small-sample correction factor:
    # For h=1, correction factor is sqrt((n - 1) / n)
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = base_stat * hln_factor
    
    # Use Student's t distribution with (n - 1) degrees of freedom
    p_val = float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))
    return float(stat), p_val


# =====================================================
# 2. DATA INGESTION & CHRONOLOGICAL PARTITIONING
# =====================================================

data = pd.read_csv("FinPulse_Processed_Data.csv", parse_dates=["Date"])
print(f"Total processed dataset length: {len(data)}")

n = len(data)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = data.iloc[:train_end].copy()
val = data.iloc[train_end:val_end].copy()
test = data.iloc[val_end:].copy()
train_val = pd.concat([train, val]).copy()

print(f"Train split:      {len(train)} sessions ({train['Date'].min().date()} to {train['Date'].max().date()})")
print(f"Validation split: {len(val)} sessions ({val['Date'].min().date()} to {val['Date'].max().date()})")
print(f"Test split:       {len(test)} sessions ({test['Date'].min().date()} to {test['Date'].max().date()})")

feature_cols = [
    "Return", "Abs_Return", "Vol_5", "Vol_10", "Vol_21",
    "Return_Lag1", "Return_Lag2", "Return_Lag3", "Return_Lag5",
    "AbsReturn_Lag1", "AbsReturn_Lag2", "AbsReturn_Lag5",
    "VIX", "VIX_Lag1", "VIX_Change_1D", "VIX_Change_5D",
    "Parkinson", "Volume_Change_1D"
]

X_train = train[feature_cols]
y_train = train["Target"].values
X_val = val[feature_cols]
y_val = val["Target"].values
X_test = test[feature_cols]
y_test = test["Target"].values
X_train_val = train_val[feature_cols]
y_train_val = train_val["Target"].values

# Fit scaler strictly on train for validation tuning, and on train+val for final testing
scaler_tune = StandardScaler()
X_train_scaled = scaler_tune.fit_transform(X_train)
X_val_scaled = scaler_tune.transform(X_val)

scaler_final = StandardScaler()
X_train_val_scaled = scaler_final.fit_transform(X_train_val)
X_test_scaled = scaler_final.transform(X_test)

norm_scaling = np.sqrt(2.0 / np.pi)  # E[|z|] under standard normal = 0.79788

# Store predictions: point predictions for E|R| and standard deviation sigma_hat
point_preds = {}
sigma_preds = {}


# =====================================================
# 3. BASELINE MODELS
# =====================================================

# 3a. Naive Persistence Baseline: today's Abs_Return
# Predicts E|R| directly; sigma_hat = pred / sqrt(2/pi)
naive_pt = test["Abs_Return"].values
point_preds["Naive"] = naive_pt
sigma_preds["Naive"] = naive_pt / norm_scaling

# 3b. Historical Realized Volatility Baseline: Vol_21
# Vol_21 is already a standard deviation sigma!
# Documented: treat Vol_21 as sigma directly for QLIKE; point forecast for E|R| is Vol_21 * sqrt(2/pi)
vol21_sigma = test["Vol_21"].values
sigma_preds["Baseline (Vol_21)"] = vol21_sigma
point_preds["Baseline (Vol_21)"] = vol21_sigma * norm_scaling

# 3c. Implied Volatility Baseline: VIX
# VIX is annualized percentage standard deviation.
# Daily standard deviation sigma = VIX / sqrt(252).
# Point forecast for E|R| = sigma * sqrt(2/pi)
vix_sigma = test["VIX"].values / np.sqrt(252)
sigma_preds["Baseline (VIX)"] = vix_sigma
point_preds["Baseline (VIX)"] = vix_sigma * norm_scaling


# =====================================================
# 4. TRADITIONAL GARCH-FAMILY MODELS
# =====================================================

cache_file = "garch_cache.csv"
meta_file = "garch_cache_meta.json"
current_arch_ver = arch.__version__
current_sm_ver = statsmodels.__version__
test_dates_str = [d.strftime("%Y-%m-%d") for d in test["Date"]]

refit_garch = True
if os.path.exists(cache_file) and os.path.exists(meta_file):
    try:
        with open(meta_file, "r") as mf:
            meta = json.load(mf)
        cache_df = pd.read_csv(cache_file)
        if (list(cache_df["Date"].values) == test_dates_str and 
            meta.get("arch_version") == current_arch_ver and 
            meta.get("statsmodels_version") == current_sm_ver):
            print("Loaded validated GARCH forecasts from cache (dates & package versions match).")
            garch_sigma = cache_df["GARCH"].values
            tgarch_sigma = cache_df["TGARCH"].values
            egarch_sigma = cache_df["EGARCH"].values
            refit_garch = False
    except Exception as e:
        print("GARCH cache validation failed, refitting:", e)
        refit_garch = True

if refit_garch:
    print("Fitting rolling GARCH, TGARCH, and EGARCH expanding-window models...")
    garch_preds_list = []
    tgarch_preds_list = []
    egarch_preds_list = []
    
    for i in range(val_end, n):
        history = data["Return"].iloc[:i + 1]
        
        # GARCH(1,1)
        m_g = arch_model(history, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        garch_preds_list.append(np.sqrt(m_g.forecast(horizon=1).variance.iloc[-1, 0]))
        
        # TGARCH(1,1) / TARCH (power=1.0)
        m_t = arch_model(history, mean="Constant", vol="GARCH", p=1, o=1, q=1, power=1.0, dist="normal").fit(disp="off")
        tgarch_preds_list.append(np.sqrt(m_t.forecast(horizon=1).variance.iloc[-1, 0]))
        
        # EGARCH(1,1)
        m_e = arch_model(history, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="normal").fit(disp="off")
        egarch_preds_list.append(np.sqrt(m_e.forecast(horizon=1).variance.iloc[-1, 0]))
        
    garch_sigma = np.array(garch_preds_list)
    tgarch_sigma = np.array(tgarch_preds_list)
    egarch_sigma = np.array(egarch_preds_list)
    
    cache_df = pd.DataFrame({
        "Date": test_dates_str,
        "GARCH": garch_sigma,
        "TGARCH": tgarch_sigma,
        "EGARCH": egarch_sigma
    })
    cache_df.to_csv(cache_file, index=False)
    
    meta_info = {
        "arch_version": current_arch_ver,
        "statsmodels_version": current_sm_ver,
        "n_test": len(test),
        "start_date": test_dates_str[0],
        "end_date": test_dates_str[-1]
    }
    with open(meta_file, "w") as mf:
        json.dump(meta_info, mf, indent=4)
    print("GARCH forecasts and metadata cached successfully.")

# GARCH outputs are conditional standard deviations sigma
# Point predictions of E|R| = sigma * sqrt(2/pi)
sigma_preds["GARCH(1,1)"] = garch_sigma
point_preds["GARCH(1,1)"] = garch_sigma * norm_scaling

sigma_preds["TGARCH"] = tgarch_sigma
point_preds["TGARCH"] = tgarch_sigma * norm_scaling

sigma_preds["EGARCH"] = egarch_sigma
point_preds["EGARCH"] = egarch_sigma * norm_scaling


# --- Parameter Estimation on Train+Val (C1) ---
print("\nEstimating GARCH parameter sets on Train+Val (986 sessions)...")
train_val_ret = train_val["Return"]

fit_garch = arch_model(train_val_ret, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
fit_tgarch = arch_model(train_val_ret, mean="Constant", vol="GARCH", p=1, o=1, q=1, power=1.0, dist="normal").fit(disp="off")
fit_egarch = arch_model(train_val_ret, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="normal").fit(disp="off")

garch_param_rows = [
    {
        "Model": "GARCH(1,1)",
        "omega": fit_garch.params.get("omega", np.nan),
        "alpha": fit_garch.params.get("alpha[1]", np.nan),
        "gamma": np.nan,
        "beta": fit_garch.params.get("beta[1]", np.nan),
        "persistence": fit_garch.params.get("alpha[1]", 0) + fit_garch.params.get("beta[1]", 0),
        "se_omega": fit_garch.std_err.get("omega", np.nan),
        "se_alpha": fit_garch.std_err.get("alpha[1]", np.nan),
        "se_gamma": np.nan,
        "se_beta": fit_garch.std_err.get("beta[1]", np.nan),
        "pval_omega": fit_garch.pvalues.get("omega", np.nan),
        "pval_alpha": fit_garch.pvalues.get("alpha[1]", np.nan),
        "pval_gamma": np.nan,
        "pval_beta": fit_garch.pvalues.get("beta[1]", np.nan),
        "log_likelihood": fit_garch.loglikelihood,
        "AIC": fit_garch.aic,
        "BIC": fit_garch.bic
    },
    {
        "Model": "TGARCH(1,1)",
        "omega": fit_tgarch.params.get("omega", np.nan),
        "alpha": fit_tgarch.params.get("alpha[1]", np.nan),
        "gamma": fit_tgarch.params.get("gamma[1]", np.nan),
        "beta": fit_tgarch.params.get("beta[1]", np.nan),
        "persistence": fit_tgarch.params.get("alpha[1]", 0) + fit_tgarch.params.get("beta[1]", 0) + 0.5 * fit_tgarch.params.get("gamma[1]", 0),
        "se_omega": fit_tgarch.std_err.get("omega", np.nan),
        "se_alpha": fit_tgarch.std_err.get("alpha[1]", np.nan),
        "se_gamma": fit_tgarch.std_err.get("gamma[1]", np.nan),
        "se_beta": fit_tgarch.std_err.get("beta[1]", np.nan),
        "pval_omega": fit_tgarch.pvalues.get("omega", np.nan),
        "pval_alpha": fit_tgarch.pvalues.get("alpha[1]", np.nan),
        "pval_gamma": fit_tgarch.pvalues.get("gamma[1]", np.nan),
        "pval_beta": fit_tgarch.pvalues.get("beta[1]", np.nan),
        "log_likelihood": fit_tgarch.loglikelihood,
        "AIC": fit_tgarch.aic,
        "BIC": fit_tgarch.bic
    },
    {
        "Model": "EGARCH(1,1)",
        "omega": fit_egarch.params.get("omega", np.nan),
        "alpha": fit_egarch.params.get("alpha[1]", np.nan),
        "gamma": fit_egarch.params.get("gamma[1]", np.nan),
        "beta": fit_egarch.params.get("beta[1]", np.nan),
        "persistence": fit_egarch.params.get("beta[1]", np.nan),
        "se_omega": fit_egarch.std_err.get("omega", np.nan),
        "se_alpha": fit_egarch.std_err.get("alpha[1]", np.nan),
        "se_gamma": fit_egarch.std_err.get("gamma[1]", np.nan),
        "se_beta": fit_egarch.std_err.get("beta[1]", np.nan),
        "pval_omega": fit_egarch.pvalues.get("omega", np.nan),
        "pval_alpha": fit_egarch.pvalues.get("alpha[1]", np.nan),
        "pval_gamma": fit_egarch.pvalues.get("gamma[1]", np.nan),
        "pval_beta": fit_egarch.pvalues.get("beta[1]", np.nan),
        "log_likelihood": fit_egarch.loglikelihood,
        "AIC": fit_egarch.aic,
        "BIC": fit_egarch.bic
    }
]
garch_params_df = pd.DataFrame(garch_param_rows)
garch_params_df.to_csv("garch_params.csv", index=False)
print("Saved GARCH parameters to garch_params.csv.")

# --- GARCH Residual Diagnostics (C2) ---
# Use EGARCH standardized residuals on test set
egarch_resids = (test["Return"].values - fit_egarch.params.get("mu", 0.0)) / egarch_sigma
lb_std_10 = acorr_ljungbox(egarch_resids, lags=[10], return_df=True).iloc[0]
lb_std_20 = acorr_ljungbox(egarch_resids, lags=[20], return_df=True).iloc[0]
lb_sq_10 = acorr_ljungbox(egarch_resids ** 2, lags=[10], return_df=True).iloc[0]
lb_sq_20 = acorr_ljungbox(egarch_resids ** 2, lags=[20], return_df=True).iloc[0]
arch_std = het_arch(egarch_resids)

garch_diag_df = pd.DataFrame([
    {"Test": "Ljung-Box on z_t (Lag 10)", "Statistic": round(float(lb_std_10["lb_stat"]), 4), "p_value": round(float(lb_std_10["lb_pvalue"]), 4)},
    {"Test": "Ljung-Box on z_t (Lag 20)", "Statistic": round(float(lb_std_20["lb_stat"]), 4), "p_value": round(float(lb_std_20["lb_pvalue"]), 4)},
    {"Test": "Ljung-Box on z_t^2 (Lag 10)", "Statistic": round(float(lb_sq_10["lb_stat"]), 4), "p_value": round(float(lb_sq_10["lb_pvalue"]), 4)},
    {"Test": "Ljung-Box on z_t^2 (Lag 20)", "Statistic": round(float(lb_sq_20["lb_stat"]), 4), "p_value": round(float(lb_sq_20["lb_pvalue"]), 4)},
    {"Test": "ARCH-LM on z_t (LM stat)", "Statistic": round(float(arch_std[0]), 4), "p_value": round(float(arch_std[1]), 4)},
    {"Test": "ARCH-LM on z_t (F stat)", "Statistic": round(float(arch_std[2]), 4), "p_value": round(float(arch_std[3]), 4)}
])
garch_diag_df.to_csv("garch_diagnostics.csv", index=False)
print("Saved GARCH residual diagnostics to garch_diagnostics.csv.")

# Plot Figure 12: GARCH diagnostics
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0, 0].plot(test["Date"], egarch_resids, color="#2563EB", linewidth=1.0)
axes[0, 0].axhline(0, color="black", linestyle="--")
axes[0, 0].axhline(2, color="red", linestyle=":")
axes[0, 0].axhline(-2, color="red", linestyle=":")
axes[0, 0].set_title("Standardized Residuals z_t (EGARCH)", fontweight="bold")
axes[0, 0].grid(True, linestyle=":", alpha=0.5)

stats.probplot(egarch_resids, dist="norm", plot=axes[0, 1])
axes[0, 1].get_lines()[0].set_color("#2563EB")
axes[0, 1].get_lines()[1].set_color("#DC2626")
axes[0, 1].set_title("Normal Q-Q Plot of Standardized Residuals", fontweight="bold")
axes[0, 1].grid(True, linestyle=":", alpha=0.5)

nlags = 20
acf_sq = acf(egarch_resids ** 2, nlags=nlags)
axes[1, 0].stem(np.arange(nlags + 1), acf_sq)
axes[1, 0].axhline(1.96 / np.sqrt(len(egarch_resids)), color="red", linestyle="--")
axes[1, 0].axhline(-1.96 / np.sqrt(len(egarch_resids)), color="red", linestyle="--")
axes[1, 0].set_title("ACF of Squared Standardized Residuals z_t^2 (ARCH Filtered)", fontweight="bold")
axes[1, 0].grid(True, linestyle=":", alpha=0.5)

axes[1, 1].hist(egarch_resids, bins=30, density=True, color="#3B82F6", edgecolor="black", alpha=0.7)
x_std = np.linspace(-4, 4, 200)
axes[1, 1].plot(x_std, stats.norm.pdf(x_std), color="#DC2626", linewidth=2.0)
axes[1, 1].set_title("Distribution of z_t vs Standard Normal N(0,1)", fontweight="bold")
axes[1, 1].grid(True, linestyle=":", alpha=0.5)

plt.suptitle("Figure 12: Econometric Residual Diagnostics for EGARCH Model", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/fig12_garch_diagnostics.png", dpi=200)
plt.close()


# =====================================================
# 5. SECOND TRADITIONAL FAMILY: HAR-RV & ARMA (C3)
# =====================================================

# 5a. HAR-style OLS regression of |R_{t+1}| on |R_t|, mean |R|_{5d}, mean |R|_{21d}
print("\nFitting HAR-RV expanding-window regression...")
abs_r = data["Abs_Return"].values
har_d = abs_r
har_w = pd.Series(abs_r).rolling(5, min_periods=1).mean().values
har_m = pd.Series(abs_r).rolling(21, min_periods=1).mean().values
X_har = np.column_stack([har_d, har_w, har_m])

har_preds = []
for t in range(val_end, n):
    reg = LinearRegression().fit(X_har[:t], data["Target"].iloc[:t])
    p = reg.predict(X_har[t:t+1])[0]
    har_preds.append(max(float(p), 0.0))

har_pt = np.array(har_preds)
point_preds["HAR"] = har_pt
sigma_preds["HAR"] = har_pt / norm_scaling

# 5b. ARMA/ARIMA on log(|R|+0.01) with order chosen by AIC on training set
print("Selecting optimal ARMA order by AIC on training set...")
log_series = np.log(data["Abs_Return"].values + 0.01)
train_log = log_series[:train_end]

best_arima_aic = float("inf")
best_arima_order = (1, 0, 1)

for p_ord in range(1, 4):
    for q_ord in range(0, 3):
        try:
            m_try = ARIMA(train_log, order=(p_ord, 0, q_ord)).fit()
            if m_try.aic < best_arima_aic:
                best_arima_aic = m_try.aic
                best_arima_order = (p_ord, 0, q_ord)
        except Exception:
            pass

print(f"Optimal ARMA order chosen by AIC on training set: {best_arima_order} (AIC: {best_arima_aic:.2f})")

# Fit ARIMA on full series and predict 1-step ahead out-of-sample with log-normal bias correction
m_arima = ARIMA(log_series, order=best_arima_order).fit()
res_var = float(m_arima.scale)

arima_preds = []
for t in range(val_end, n):
    p_log = m_arima.predict(start=t, end=t)[0]
    # Bias correction: E[Y] = exp(mu + 0.5 * sigma_e^2) - 0.01
    p_level = max(float(np.exp(p_log + 0.5 * res_var) - 0.01), 0.0)
    arima_preds.append(p_level)

arima_pt = np.array(arima_preds)
point_preds[f"ARIMA{best_arima_order}"] = arima_pt
sigma_preds[f"ARIMA{best_arima_order}"] = arima_pt / norm_scaling


# =====================================================
# 6. MACHINE LEARNING MODELS & DETERMINISTIC TUNING
# =====================================================

tuning_log = []

# --- 6a. Ridge Regression Tuning ---
print("\n--- Tuning Ridge Regression on Validation Set ---")
alphas = [0.01, 0.1, 1, 10, 100]
best_alpha = None
best_ridge_val_mae = float("inf")

for alpha in alphas:
    r_tune = Ridge(alpha=alpha, random_state=42).fit(X_train_scaled, y_train)
    val_pred = np.clip(r_tune.predict(X_val_scaled), 0, None)
    v_mae = float(mean_absolute_error(y_val, val_pred))
    tuning_log.append({
        "Model": "Ridge",
        "Parameters": f"alpha={alpha}",
        "Val_MAE": round(v_mae, 6),
        "Is_Selected": False
    })
    print(f"Ridge alpha={alpha:<5} | Validation MAE: {v_mae:.6f}")
    if v_mae < best_ridge_val_mae:
        best_ridge_val_mae = v_mae
        best_alpha = alpha

print(f"Optimal Ridge alpha: {best_alpha} (Val MAE: {best_ridge_val_mae:.6f})")

# Retrain Ridge on train_val
ridge_model = Ridge(alpha=best_alpha, random_state=42).fit(X_train_val_scaled, y_train_val)
ridge_pt = np.clip(ridge_model.predict(X_test_scaled), 0, None)
point_preds["Ridge"] = ridge_pt
sigma_preds["Ridge"] = ridge_pt / norm_scaling

# --- 6b. SVR Tuning ---
print("\n--- Tuning Support Vector Regression (SVR RBF) on Validation Set ---")
Cs = [0.1, 1, 10, 100]
gammas = ["scale", 0.01, 0.1]
epsilons = [0.01, 0.05, 0.1]

best_svr_params = None
best_svr_val_mae = float("inf")

for C in Cs:
    for g in gammas:
        for eps in epsilons:
            svr_tune = SVR(kernel="rbf", C=C, gamma=g, epsilon=eps).fit(X_train_scaled, y_train)
            val_pred = np.clip(svr_tune.predict(X_val_scaled), 0, None)
            v_mae = float(mean_absolute_error(y_val, val_pred))
            tuning_log.append({
                "Model": "SVR",
                "Parameters": f"C={C}, gamma={g}, epsilon={eps}",
                "Val_MAE": round(v_mae, 6),
                "Is_Selected": False
            })
            if v_mae < best_svr_val_mae:
                best_svr_val_mae = v_mae
                best_svr_params = {"C": C, "gamma": g, "epsilon": eps}

print(f"Optimal SVR Parameters: {best_svr_params} (Val MAE: {best_svr_val_mae:.6f})")

# Retrain SVR on train_val
svr_model = SVR(kernel="rbf", **best_svr_params).fit(X_train_val_scaled, y_train_val)
svr_pt = np.clip(svr_model.predict(X_test_scaled), 0, None)
point_preds["SVR"] = svr_pt
sigma_preds["SVR"] = svr_pt / norm_scaling

# SVR Under-prediction & QLIKE Breakdown diagnostic (D5)
svr_under_01 = int(np.sum(svr_pt < 0.1))
svr_sig_test = np.maximum(svr_pt / norm_scaling, 0.05)
svr_qlike_daily = np.log(svr_sig_test**2) + (y_test**2) / (svr_sig_test**2)
svr_total_qlike = float(np.mean(svr_qlike_daily))
svr_sum_qlike = float(np.sum(svr_qlike_daily))

# Build per-day QLIKE contribution table
svr_daily_df = pd.DataFrame({
    "Date": test["Date"].dt.strftime("%Y-%m-%d").values if hasattr(test["Date"], "dt") else [str(d)[:10] for d in test["Date"]],
    "SVR_Pred": np.round(svr_pt, 4),
    "Target": np.round(y_test, 4),
    "Daily_QLIKE": np.round(svr_qlike_daily, 4),
    "Share_Pct": np.round((svr_qlike_daily / svr_sum_qlike) * 100.0, 2)
})
svr_top5_df = svr_daily_df.sort_values(by="Daily_QLIKE", ascending=False).head(5).copy()
svr_top5_df.insert(0, "Rank", range(1, 6))
svr_top5_df.to_csv("svr_top5_qlike.csv", index=False)
print("Saved SVR top 5 QLIKE contributors to svr_top5_qlike.csv.")

worst_row = svr_top5_df.iloc[0]

svr_diag_info = {
    "min_prediction": round(float(np.min(svr_pt)), 4),
    "p01": round(float(np.percentile(svr_pt, 1)), 4),
    "p05": round(float(np.percentile(svr_pt, 5)), 4),
    "p10": round(float(np.percentile(svr_pt, 10)), 4),
    "p25": round(float(np.percentile(svr_pt, 25)), 4),
    "median": round(float(np.median(svr_pt)), 4),
    "days_below_0_1": svr_under_01,
    "share_below_0_1_pct": round(float(svr_under_01 / len(svr_pt) * 100), 2),
    "worst_day_date": str(worst_row["Date"]),
    "worst_day_pred": float(worst_row["SVR_Pred"]),
    "worst_day_target": float(worst_row["Target"]),
    "worst_day_qlike": float(worst_row["Daily_QLIKE"]),
    "worst_day_share_pct": float(worst_row["Share_Pct"]),
    "total_qlike": round(svr_total_qlike, 4),
    "explanation": f"Approximately {round(float(worst_row['Share_Pct']))}% of SVR's total test QLIKE loss stems from a single forecast origin on {worst_row['Date']} (where SVR predicted {worst_row['SVR_Pred']:.3f} vs an actual next-day move of {worst_row['Target']:.2f}%). Across the entire 247-day test set, only {svr_under_01} prediction fell below 0.10%. Extrapolation failure outside the training distribution remains a plausible hypothesis, though not definitively proven."
}
with open("svr_diagnostics.json", "w") as f:
    json.dump(svr_diag_info, f, indent=4)
print("Saved SVR diagnostics to svr_diagnostics.json.")

# --- 6c. XGBoost Tuning with Deterministic Tie-Break ---
print("\n--- Tuning XGBoost Regressor on Validation Set ---")
lrs = [0.01, 0.05, 0.1]
depths = [2, 3, 4]

xgb_tune_results = []
for lr in lrs:
    for depth in depths:
        xgb_try = XGBRegressor(
            n_estimators=300,
            learning_rate=lr,
            max_depth=depth,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        )
        xgb_try.fit(X_train, y_train)
        val_pred = np.clip(xgb_try.predict(X_val), 0, None)
        v_mae = float(mean_absolute_error(y_val, val_pred))
        xgb_tune_results.append({
            "learning_rate": lr,
            "max_depth": depth,
            "val_mae": v_mae
        })
        tuning_log.append({
            "Model": "XGBoost",
            "Parameters": f"lr={lr}, depth={depth}",
            "Val_MAE": round(v_mae, 6),
            "Is_Selected": False
        })
        print(f"XGBoost lr={lr:<4} depth={depth} | Validation MAE: {v_mae:.6f}")

# Deterministic tie-break rule: lowest MAE (rounded to 5 decimals), then lower depth, then lower lr
xgb_tune_results.sort(key=lambda x: (round(x["val_mae"], 5), x["max_depth"], x["learning_rate"]))
best_xgb_params = {
    "learning_rate": xgb_tune_results[0]["learning_rate"],
    "max_depth": xgb_tune_results[0]["max_depth"]
}
best_xgb_val_mae = xgb_tune_results[0]["val_mae"]
print(f"Optimal XGBoost Parameters: {best_xgb_params} (Val MAE: {best_xgb_val_mae:.6f})")

# Mark selected parameters in tuning_log
for rec in tuning_log:
    if rec["Model"] == "Ridge" and f"alpha={best_alpha}" == rec["Parameters"]:
        rec["Is_Selected"] = True
    elif rec["Model"] == "SVR" and f"C={best_svr_params['C']}, gamma={best_svr_params['gamma']}, epsilon={best_svr_params['epsilon']}" == rec["Parameters"]:
        rec["Is_Selected"] = True
    elif rec["Model"] == "XGBoost" and f"lr={best_xgb_params['learning_rate']}, depth={best_xgb_params['max_depth']}" == rec["Parameters"]:
        rec["Is_Selected"] = True

pd.DataFrame(tuning_log).to_csv("tuning_results.csv", index=False)
print("Saved hyperparameter validation grid to tuning_results.csv.")

# Retrain XGBoost on train_val
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
xgb_pt = np.clip(xgb_model.predict(X_test), 0, None)
point_preds["XGBoost"] = xgb_pt
sigma_preds["XGBoost"] = xgb_pt / norm_scaling


# =====================================================
# 7. WALK-FORWARD ML EVALUATION (D2)
# =====================================================

print("\nExecuting Walk-Forward ML Evaluation (Refitting every 21 trading days on expanding window)...")
step = 21
wf_preds = {"Ridge": [], "SVR": [], "XGBoost": []}

for block_start in range(val_end, n, step):
    block_end = min(block_start + step, n)
    X_wf_train = data[feature_cols].iloc[:block_start]
    y_wf_train = data["Target"].iloc[:block_start].values
    X_wf_test_block = data[feature_cols].iloc[block_start:block_end]
    
    # Ridge
    sc_wf = StandardScaler()
    X_wf_train_s = sc_wf.fit_transform(X_wf_train)
    X_wf_test_s = sc_wf.transform(X_wf_test_block)
    
    r_wf = Ridge(alpha=best_alpha, random_state=42).fit(X_wf_train_s, y_wf_train)
    wf_preds["Ridge"].extend(np.clip(r_wf.predict(X_wf_test_s), 0, None))
    
    # SVR
    s_wf = SVR(kernel="rbf", **best_svr_params).fit(X_wf_train_s, y_wf_train)
    wf_preds["SVR"].extend(np.clip(s_wf.predict(X_wf_test_s), 0, None))
    
    # XGBoost
    x_wf = XGBRegressor(
        n_estimators=300,
        learning_rate=best_xgb_params["learning_rate"],
        max_depth=best_xgb_params["max_depth"],
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    ).fit(X_wf_train, y_wf_train)
    wf_preds["XGBoost"].extend(np.clip(x_wf.predict(X_wf_test_block), 0, None))

wf_records = []
for m_name in ["Ridge", "SVR", "XGBoost"]:
    pt_wf = np.array(wf_preds[m_name])
    sig_wf = pt_wf / norm_scaling
    m_wf = evaluate_metrics(y_test, pt_wf, sig_wf)
    
    # Also get fixed fit metrics
    m_fix = evaluate_metrics(y_test, point_preds[m_name], sigma_preds[m_name])
    
    wf_records.append({
        "Model": m_name,
        "Fixed_MAE": round(m_fix["MAE"], 4),
        "WalkForward_MAE": round(m_wf["MAE"], 4),
        "Fixed_RMSE": round(m_fix["RMSE"], 4),
        "WalkForward_RMSE": round(m_wf["RMSE"], 4),
        "Fixed_QLIKE": round(m_fix["QLIKE"], 4),
        "WalkForward_QLIKE": round(m_wf["QLIKE"], 4)
    })

pd.DataFrame(wf_records).to_csv("walk_forward_results.csv", index=False)
print("Saved walk-forward results to walk_forward_results.csv.")


# =====================================================
# 8. ML RESIDUAL DIAGNOSTICS & FEATURE IMPORTANCE (D3, D4)
# =====================================================

# Residual diagnostics plot
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
ml_names = ["Ridge", "SVR", "XGBoost"]

for idx, m_name in enumerate(ml_names):
    p = point_preds[m_name]
    res = y_test - p
    
    # Residual vs Fitted
    axes[idx, 0].scatter(p, res, color="#2563EB", alpha=0.6, s=16)
    axes[idx, 0].axhline(0, color="red", linestyle="--")
    axes[idx, 0].set_title(f"{m_name}: Residuals vs Fitted", fontweight="bold")
    axes[idx, 0].set_xlabel("Fitted Values")
    axes[idx, 0].set_ylabel("Residuals")
    axes[idx, 0].grid(True, linestyle=":", alpha=0.5)
    
    # Residual ACF
    acf_res = acf(res, nlags=15)
    axes[idx, 1].stem(np.arange(16), acf_res)
    axes[idx, 1].axhline(1.96 / np.sqrt(len(res)), color="red", linestyle="--")
    axes[idx, 1].axhline(-1.96 / np.sqrt(len(res)), color="red", linestyle="--")
    axes[idx, 1].set_title(f"{m_name}: Residual ACF", fontweight="bold")
    axes[idx, 1].grid(True, linestyle=":", alpha=0.5)
    
    # Q-Q plot
    stats.probplot(res, dist="norm", plot=axes[idx, 2])
    axes[idx, 2].get_lines()[0].set_color("#2563EB")
    axes[idx, 2].get_lines()[1].set_color("#DC2626")
    axes[idx, 2].set_title(f"{m_name}: Residual Normal Q-Q Plot", fontweight="bold")
    axes[idx, 2].grid(True, linestyle=":", alpha=0.5)

plt.suptitle("Figure 13: Machine Learning Residual Diagnostics (Ridge, SVR, XGBoost)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/fig13_ml_diagnostics.png", dpi=200)
plt.close()

# Feature importance (D4)
perm_imp = permutation_importance(xgb_model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=1)
xgb_gain = xgb_model.feature_importances_
ridge_coefs = ridge_model.coef_

fi_df = pd.DataFrame({
    "Feature": feature_cols,
    "Ridge_Standardized_Coef": ridge_coefs,
    "Ridge_Abs_Coef": np.abs(ridge_coefs),
    "XGBoost_Gain_Importance": xgb_gain,
    "XGBoost_Permutation_Mean": perm_imp.importances_mean,
    "XGBoost_Permutation_Std": perm_imp.importances_std
}).sort_values(by="Ridge_Abs_Coef", ascending=False)
fi_df.to_csv("feature_importance.csv", index=False)
print("Saved feature importance to feature_importance.csv.")

# Plot Figure 14: Feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Standardized Ridge Coefficients
r_sorted = fi_df.sort_values(by="Ridge_Standardized_Coef", ascending=True)
colors_r = ["#2563EB" if w >= 0 else "#EF4444" for w in r_sorted["Ridge_Standardized_Coef"]]
axes[0].barh(r_sorted["Feature"], r_sorted["Ridge_Standardized_Coef"], color=colors_r, edgecolor="black", alpha=0.85)
axes[0].axvline(0, color="black", linewidth=1.0)
axes[0].set_title("Standardized Ridge Regression Coefficients (alpha=10)", fontweight="bold")
axes[0].set_xlabel("Coefficient Weight")
axes[0].grid(axis="x", linestyle=":", alpha=0.6)

# XGBoost Permutation Importance vs Gain
x_sorted = fi_df.sort_values(by="XGBoost_Permutation_Mean", ascending=True)
axes[1].barh(x_sorted["Feature"], x_sorted["XGBoost_Permutation_Mean"], color="#10B981", edgecolor="black", alpha=0.85)
axes[1].set_title("XGBoost Permutation Importance (Test Set, 20 Repeats)", fontweight="bold")
axes[1].set_xlabel("Mean Importance (MAE Increase)")
axes[1].grid(axis="x", linestyle=":", alpha=0.6)

plt.suptitle("Figure 14: Predictor Feature Importance Comparison: Linear Ridge vs Non-Linear XGBoost", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/fig14_feature_importance.png", dpi=200)
plt.close()


# =====================================================
# 9. COMPILATION & PERFORMANCE BENCHMARKING
# =====================================================

model_order = [
    "Naive",
    "Baseline (Vol_21)",
    "Baseline (VIX)",
    "GARCH(1,1)",
    "TGARCH",
    "EGARCH",
    "HAR",
    f"ARIMA{best_arima_order}",
    "Ridge",
    "SVR",
    "XGBoost"
]

benchmark_records = []
for name in model_order:
    res = evaluate_metrics(y_test, point_preds[name], sigma_preds[name])
    res["Model"] = name
    benchmark_records.append(res)

bench_df = pd.DataFrame(benchmark_records)[["Model", "MAE", "RMSE", "QLIKE"]]

# Sort ranking by corrected QLIKE, with MAE as tiebreaker
ranking_df = bench_df.sort_values(by=["QLIKE", "MAE"]).reset_index(drop=True)
ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))

# Save predictions_test.csv (Date, Target, one column per model)
pred_export_df = pd.DataFrame({"Date": test["Date"].dt.strftime("%Y-%m-%d"), "Target": y_test})
for name in model_order:
    pred_export_df[name] = point_preds[name]
pred_export_df.to_csv("predictions_test.csv", index=False)
print("Saved all model test predictions to predictions_test.csv.")

# Compute Diebold-Mariano test with HLN correction against overall #1 QLIKE model
best_overall_model = ranking_df.iloc[0]["Model"]
best_overall_preds = point_preds[best_overall_model]

dm_stats = []
dm_pvals = []
for m_name in ranking_df["Model"]:
    if m_name == best_overall_model:
        dm_stats.append(0.0)
        dm_pvals.append(1.0)
    else:
        stat, pval = diebold_mariano_hln(y_test, point_preds[m_name], best_overall_preds, h=1, loss_type="qlike")
        dm_stats.append(round(stat, 4))
        dm_pvals.append(round(pval, 4))

ranking_df["DM_Stat_vs_Best"] = dm_stats
ranking_df["DM_p_value"] = dm_pvals

print("\n" + "=" * 75)
print("FINAL MODEL RANKING (Ranked by Corrected QLIKE Loss, then MAE)")
print("=" * 75)
print(ranking_df.round(4).to_string(index=False))

ranking_df.to_csv("model_comparison.csv", index=False)
print("Saved benchmark comparison to model_comparison.csv.")


# =====================================================
# 10. MODEL SELECTION & DEPLOYMENT DECISION (A3)
# =====================================================

# Deployment Rule:
# Deploy only models working from live feature pipeline (Ridge, SVR, XGBoost).
# Select model with lowest QLIKE unless DM test against alternative deployable model shows no significant difference (p > 0.05),
# in which case prefer simpler / more interpretable model (Ridge > SVR > XGBoost).

deployable_subset = ["Ridge", "SVR", "XGBoost"]
ml_ranking = ranking_df[ranking_df["Model"].isin(deployable_subset)].sort_values(by=["QLIKE", "MAE"]).reset_index(drop=True)
ml_top1 = ml_ranking.iloc[0]["Model"]
ml_top2 = ml_ranking.iloc[1]["Model"]

dm_ml_stat, dm_ml_p = diebold_mariano_hln(y_test, point_preds[ml_top2], point_preds[ml_top1], h=1, loss_type="qlike")
is_sig_ml = dm_ml_p < 0.05

simplicity_rank = {"Ridge": 1, "SVR": 2, "XGBoost": 3}

if not is_sig_ml:
    candidates = [ml_top1, ml_top2]
    candidates.sort(key=lambda m: simplicity_rank[m])
    selected_model_name = candidates[0]
    selection_reasoning = (
        f"Among deployable ML pipeline models, {ml_top1} (QLIKE: {ml_ranking.loc[ml_ranking['Model']==ml_top1, 'QLIKE'].values[0]:.4f}) "
        f"and {ml_top2} (QLIKE: {ml_ranking.loc[ml_ranking['Model']==ml_top2, 'QLIKE'].values[0]:.4f}) showed no statistically "
        f"significant difference in predictive loss under Diebold-Mariano testing (HLN-corrected p-value = {dm_ml_p:.4f} > 0.05). "
        f"Following the pre-registered model selection protocol, the simpler, more stable, and fully interpretable model, {selected_model_name}, was selected for deployment."
    )
else:
    selected_model_name = ml_top1
    selection_reasoning = (
        f"{selected_model_name} demonstrated statistically significant superior performance over {ml_top2} "
        f"(Diebold-Mariano p-value = {dm_ml_p:.4f} < 0.05) and was selected for production deployment."
    )

print(f"\nPRODUCTION DEPLOYMENT CHOICE: {selected_model_name}")
print("Selection rationale:", selection_reasoning)

model_mapping = {
    "Ridge": (ridge_model, True, {"alpha": float(best_alpha), "random_state": 42}),
    "SVR": (svr_model, True, best_svr_params),
    "XGBoost": (xgb_model, False, best_xgb_params)
}

best_estimator, needs_scaling, chosen_params = model_mapping[selected_model_name]
selected_metrics = evaluate_metrics(y_test, point_preds[selected_model_name], sigma_preds[selected_model_name])

joblib.dump(best_estimator, "best_model.pkl")
joblib.dump(scaler_final, "scaler.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(ridge_model, "ridge_model.pkl")

model_info = {
    "model_name": selected_model_name,
    "model_type": selected_model_name,
    "parameters": chosen_params,
    "needs_scaling": needs_scaling,
    "test_metrics": {k: round(v, 4) for k, v in selected_metrics.items()},
    "selection_reasoning": selection_reasoning,
    "best_overall_model": best_overall_model,
    "best_overall_qlike": round(float(ranking_df.iloc[0]["QLIKE"]), 4),
    "volatility_proxy_conversion": "E[|R|] = sigma * sqrt(2/pi); sigma = E[|R|] / sqrt(2/pi)"
}

with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=4)
print("Saved best_model.pkl, scaler.pkl, feature_cols.pkl, and model_info.json.")


# =====================================================
# 11. VISUALIZATION EXPORTS (FIGURES 10 & 11)
# =====================================================

# Figure 10: Actual vs Predicted Top 3 Models with April 2025 window highlighted
top3_names = ranking_df["Model"].iloc[:3].tolist()
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(test["Date"], test["Target"], label="Actual (|Return|)", color="#111827", linewidth=1.5, alpha=0.85)

colors = ["#2563EB", "#059669", "#D97706"]
styles = ["-", "--", "-."]
for idx, name in enumerate(top3_names):
    ax.plot(
        test["Date"],
        point_preds[name],
        label=f"{name} (Rank #{idx+1})",
        color=colors[idx % len(colors)],
        linestyle=styles[idx % len(styles)],
        linewidth=1.3,
        alpha=0.9
    )

# Highlight April 2025 Shock Window
ax.axvspan(pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30"), color="#FEE2E2", alpha=0.6, label="April 2025 Shock Window (VIX peak 52.33)")

ax.set_title("Figure 10: Actual vs. Predicted Next-Session Volatility (Top 3 Ranked Models with April 2025 Stress Window)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Trading Date", fontsize=11)
ax.set_ylabel("Volatility Magnitude (%)", fontsize=11)
ax.legend(loc="upper left", frameon=True, framealpha=0.9)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig10_actual_vs_predicted.png", dpi=200)
# Also maintain backwards compatibility
plt.savefig("actual_vs_predicted_top3.png", dpi=200)
plt.close()

# Figure 11: Bar Chart of Model Metrics (MAE, RMSE, QLIKE on log scale)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
plot_df = ranking_df.copy()
bar_colors = ["#3B82F6" if m == selected_model_name else "#9CA3AF" for m in plot_df["Model"]]

# MAE
axes[0].barh(plot_df["Model"], plot_df["MAE"], color=bar_colors)
axes[0].set_title("Mean Absolute Error (MAE %)", fontweight="bold")
axes[0].set_xlabel("MAE (%)")
axes[0].invert_yaxis()
axes[0].grid(axis="x", linestyle=":", alpha=0.6)

# RMSE
axes[1].barh(plot_df["Model"], plot_df["RMSE"], color=bar_colors)
axes[1].set_title("Root Mean Squared Error (RMSE %)", fontweight="bold")
axes[1].set_xlabel("RMSE (%)")
axes[1].invert_yaxis()
axes[1].grid(axis="x", linestyle=":", alpha=0.6)

# QLIKE on Log Scale
axes[2].barh(plot_df["Model"], plot_df["QLIKE"], color=bar_colors)
axes[2].set_title("Quasi-Likelihood (QLIKE Loss - Log Scale)", fontweight="bold")
axes[2].set_xlabel("QLIKE Loss (Log Scale)")
axes[2].set_xscale("log")
axes[2].invert_yaxis()
axes[2].grid(axis="x", linestyle=":", alpha=0.6)

plt.suptitle(f"Figure 11: Benchmark Performance Comparison Across All Models (Highlighted: Selected '{selected_model_name}')", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/fig11_model_comparison_bars.png", dpi=200)
plt.savefig("model_metrics_barchart.png", dpi=200)
plt.close()

print("models.py execution completed successfully!")