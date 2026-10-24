import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

os.makedirs("figures", exist_ok=True)

# 1. Load Processed Dataset
data = pd.read_csv("FinPulse_Processed_Data.csv", parse_dates=["Date"])
print(f"Loaded processed dataset with {len(data)} rows.")

# 18 Features defined for modeling
feature_cols = [
    "Return", "Abs_Return", "Vol_5", "Vol_10", "Vol_21",
    "Return_Lag1", "Return_Lag2", "Return_Lag3", "Return_Lag5",
    "AbsReturn_Lag1", "AbsReturn_Lag2", "AbsReturn_Lag5",
    "VIX", "VIX_Lag1", "VIX_Change_1D", "VIX_Change_5D",
    "Parkinson", "Volume_Change_1D"
]

# -------------------------------------------------------------
# FIGURE 1: SPY Closing Price
# -------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(data["Date"], data["Close"], color="#1E3A8A", linewidth=1.5)
plt.title("Figure 1: SPY Closing Price Trajectory (2021–2025)", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Date", fontsize=11)
plt.ylabel("Price ($)", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig01_spy_price.png", dpi=200)
plt.close()

# -------------------------------------------------------------
# FIGURE 2: SPY Daily Log Returns
# -------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(data["Date"], data["Return"], color="#2563EB", linewidth=1.0, alpha=0.85)
plt.axhline(0, color="#DC2626", linestyle="--", linewidth=1.0, alpha=0.7)
plt.title("Figure 2: SPY Daily Log Returns (%)", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Date", fontsize=11)
plt.ylabel("Return (%)", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig02_spy_returns.png", dpi=200)
plt.close()

# -------------------------------------------------------------
# FIGURE 3: SPY Absolute Returns
# -------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(data["Date"], data["Abs_Return"], color="#D97706", linewidth=1.0, alpha=0.85)
plt.title("Figure 3: SPY Daily Absolute Returns (|R|) Volatility Proxy", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Date", fontsize=11)
plt.ylabel("Absolute Return (%)", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig03_spy_abs_returns.png", dpi=200)
plt.close()

# -------------------------------------------------------------
# FIGURE 4: VIX with 20 and 30 Threshold Lines
# -------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(data["Date"], data["VIX"], color="#7C3AED", linewidth=1.2, label="VIX Index")
plt.axhline(20, color="#EAB308", linestyle="--", linewidth=1.2, label="Moderate Risk Threshold (VIX = 20)")
plt.axhline(30, color="#EF4444", linestyle="-.", linewidth=1.2, label="High Panic Threshold (VIX = 30)")
plt.title("Figure 4: CBOE Volatility Index (VIX) with Market Stress Thresholds", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Date", fontsize=11)
plt.ylabel("VIX Level", fontsize=11)
plt.legend(loc="upper right", frameon=True, framealpha=0.9)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig04_vix_index.png", dpi=200)
plt.close()

# -------------------------------------------------------------
# FIGURE 5: Returns Distribution with Normal Density Overlay
# -------------------------------------------------------------
ret = data["Return"].dropna()
mu, std = ret.mean(), ret.std()
x = np.linspace(ret.min(), ret.max(), 300)
p = stats.norm.pdf(x, mu, std)

plt.figure(figsize=(9, 5))
plt.hist(ret, bins=60, density=True, alpha=0.65, color="#3B82F6", edgecolor="#1D4ED8", label="Empirical Returns")
plt.plot(x, p, color="#DC2626", linewidth=2.0, label=f"Normal Density N({mu:.2f}, {std:.2f}^2)")
plt.title("Figure 5: Distribution of SPY Daily Returns vs. Theoretical Normal Density", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Daily Log Return (%)", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.legend(frameon=True)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig05_returns_distribution.png", dpi=200)
plt.close()

# -------------------------------------------------------------
# FIGURE 6: Q-Q Plot of Returns
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
stats.probplot(ret, dist="norm", plot=ax)
ax.get_lines()[0].set_markerfacecolor("#2563EB")
ax.get_lines()[0].set_markeredgecolor("#1D4ED8")
ax.get_lines()[0].set_markersize(4.0)
ax.get_lines()[1].set_color("#DC2626")
ax.get_lines()[1].set_linewidth(1.8)
ax.set_title("Figure 6: Normal Q-Q Plot of SPY Daily Returns (Leptokurtosis Evidence)", fontsize=13, fontweight="bold", pad=12)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig06_qq_plot.png", dpi=200)
plt.close()

# -------------------------------------------------------------
# FIGURE 7: ACF and PACF of Returns and Absolute Returns
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
nlags = 30
lags = np.arange(nlags + 1)
conf_bound = 1.96 / np.sqrt(len(ret))

acf_ret = acf(ret, nlags=nlags)
pacf_ret = pacf(ret, nlags=nlags)
acf_abs = acf(data["Abs_Return"], nlags=nlags)
pacf_abs = pacf(data["Abs_Return"], nlags=nlags)

# Return ACF
axes[0, 0].stem(lags, acf_ret)
axes[0, 0].axhline(conf_bound, color="red", linestyle="--", alpha=0.7)
axes[0, 0].axhline(-conf_bound, color="red", linestyle="--", alpha=0.7)
axes[0, 0].set_title("SPY Raw Returns ACF (No Linear Autocorrelation)", fontweight="bold", fontsize=11)
axes[0, 0].grid(True, linestyle=":", alpha=0.5)

# Return PACF
axes[0, 1].stem(lags, pacf_ret)
axes[0, 1].axhline(conf_bound, color="red", linestyle="--", alpha=0.7)
axes[0, 1].axhline(-conf_bound, color="red", linestyle="--", alpha=0.7)
axes[0, 1].set_title("SPY Raw Returns PACF", fontweight="bold", fontsize=11)
axes[0, 1].grid(True, linestyle=":", alpha=0.5)

# Abs Return ACF
axes[1, 0].stem(lags, acf_abs)
axes[1, 0].axhline(conf_bound, color="red", linestyle="--", alpha=0.7)
axes[1, 0].axhline(-conf_bound, color="red", linestyle="--", alpha=0.7)
axes[1, 0].set_title("SPY Absolute Returns ACF (Persistent Volatility Clustering)", fontweight="bold", fontsize=11)
axes[1, 0].grid(True, linestyle=":", alpha=0.5)

# Abs Return PACF
axes[1, 1].stem(lags, pacf_abs)
axes[1, 1].axhline(conf_bound, color="red", linestyle="--", alpha=0.7)
axes[1, 1].axhline(-conf_bound, color="red", linestyle="--", alpha=0.7)
axes[1, 1].set_title("SPY Absolute Returns PACF", fontweight="bold", fontsize=11)
axes[1, 1].grid(True, linestyle=":", alpha=0.5)

plt.suptitle("Figure 7: Autocorrelation (ACF) and Partial Autocorrelation (PACF) Diagnostics", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/fig07_acf_pacf.png", dpi=200)
plt.close()

# -------------------------------------------------------------
# FIGURE 8: Rolling 21-Day Volatility
# -------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(data["Date"], data["Vol_21"], color="#059669", linewidth=1.5, label="21-Day Realized Volatility (%)")
plt.title("Figure 8: SPY Historical 21-Day Rolling Volatility Benchmark", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Date", fontsize=11)
plt.ylabel("Volatility (%)", fontsize=11)
plt.legend(frameon=True)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig08_rolling_vol21.png", dpi=200)
plt.close()

# -------------------------------------------------------------
# FIGURE 9: Feature Correlation Bar Chart (18 Model Predictors Only)
# -------------------------------------------------------------
corr_series = data[feature_cols].corrwith(data["Target"]).sort_values(ascending=True)

plt.figure(figsize=(10, 7))
colors = ["#2563EB" if val >= 0 else "#EF4444" for val in corr_series.values]
plt.barh(corr_series.index, corr_series.values, color=colors, edgecolor="black", alpha=0.85)
plt.axvline(0, color="black", linewidth=1.0)
plt.title("Figure 9: Pearson Correlation of 18 Predictor Features with Next-Day Volatility (|R_{t+1}|)", fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Correlation Coefficient with Target (|R_{t+1}|)", fontsize=11)
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig09_feature_correlation.png", dpi=200)
plt.close()

print("All 9 figures saved to figures/ successfully at 200 dpi.")

# -------------------------------------------------------------
# STATISTICAL DIAGNOSTICS & EXPORTS (B2)
# -------------------------------------------------------------
stat_cols = ["Close", "Return", "Abs_Return", "VIX", "Parkinson"]
desc_stats = {}
desc_table_rows = []

for col in stat_cols:
    s = data[col].dropna()
    mean_val = float(s.mean())
    std_val = float(s.std())
    min_val = float(s.min())
    p25_val = float(s.quantile(0.25))
    median_val = float(s.median())
    p75_val = float(s.quantile(0.75))
    max_val = float(s.max())
    skew_val = float(s.skew())
    kurt_val = float(s.kurt()) # excess kurtosis (Fisher definition)
    
    desc_stats[col] = {
        "count": int(len(s)),
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "p25": p25_val,
        "median": median_val,
        "p75": p75_val,
        "max": max_val,
        "skew": skew_val,
        "excess_kurtosis": kurt_val
    }
    
    desc_table_rows.append({
        "Series": col,
        "Count": len(s),
        "Mean": round(mean_val, 4),
        "Std": round(std_val, 4),
        "Min": round(min_val, 4),
        "25%": round(p25_val, 4),
        "Median": round(median_val, 4),
        "75%": round(p75_val, 4),
        "Max": round(max_val, 4),
        "Skewness": round(skew_val, 4),
        "Excess_Kurtosis": round(kurt_val, 4)
    })

desc_df = pd.DataFrame(desc_table_rows)

# Augmented Dickey-Fuller (ADF)
adf_close = adfuller(data["Close"].dropna())
adf_ret = adfuller(data["Return"].dropna())

adf_results = {
    "Close": {
        "adf_stat": float(adf_close[0]),
        "p_value": float(adf_close[1]),
        "critical_values": {k: float(v) for k, v in adf_close[4].items()},
        "is_stationary": bool(adf_close[1] < 0.05)
    },
    "Return": {
        "adf_stat": float(adf_ret[0]),
        "p_value": float(adf_ret[1]),
        "critical_values": {k: float(v) for k, v in adf_ret[4].items()},
        "is_stationary": bool(adf_ret[1] < 0.05)
    }
}

# Ljung-Box Test (lags 5, 10, 20)
lb_ret = acorr_ljungbox(data["Return"].dropna(), lags=[5, 10, 20], return_df=True)
lb_abs = acorr_ljungbox(data["Abs_Return"].dropna(), lags=[5, 10, 20], return_df=True)

ljung_box_results = {
    "Return": {
        str(lag): {"stat": float(lb_ret.loc[lag, "lb_stat"]), "p_value": float(lb_ret.loc[lag, "lb_pvalue"])}
        for lag in [5, 10, 20]
    },
    "Abs_Return": {
        str(lag): {"stat": float(lb_abs.loc[lag, "lb_stat"]), "p_value": float(lb_abs.loc[lag, "lb_pvalue"])}
        for lag in [5, 10, 20]
    }
}

# ARCH-LM Test
arch_test = het_arch(data["Return"].dropna())
arch_results = {
    "lm_stat": float(arch_test[0]),
    "lm_p_value": float(arch_test[1]),
    "f_stat": float(arch_test[2]),
    "f_p_value": float(arch_test[3])
}

# Jarque-Bera Test
jb_ret = stats.jarque_bera(data["Return"].dropna())
jb_abs = stats.jarque_bera(data["Abs_Return"].dropna())

jb_results = {
    "Return": {
        "stat": float(jb_ret.statistic),
        "p_value": float(jb_ret.pvalue)
    },
    "Abs_Return": {
        "stat": float(jb_abs.statistic),
        "p_value": float(jb_abs.pvalue)
    }
}

eda_full_results = {
    "sample_period": {
        "start": str(data["Date"].min().date()),
        "end": str(data["Date"].max().date()),
        "total_sessions": int(len(data))
    },
    "descriptive_statistics": desc_stats,
    "adf_test": adf_results,
    "ljung_box_test": ljung_box_results,
    "arch_lm_test": arch_results,
    "jarque_bera_test": jb_results
}

with open("eda_results.json", "w") as f:
    json.dump(eda_full_results, f, indent=4)

desc_df.to_csv("eda_results.csv", index=False)
print("Saved eda_results.json and eda_results.csv successfully.")
