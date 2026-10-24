# 📋 FinPulse: Audit of Modifications and Changelog (CHANGES.md)

This document provides a comprehensive audit of all algorithmic bug fixes, econometric extensions, machine learning diagnostics, empirical findings, and report syntheses executed in the **FinPulse** repository for the university assignment: **"Real-World Financial Time Series Analytics"**.

---

## 1. Summary of Completed Modifications

### Part A: Code Bug Fixes & Reproducibility
1. **Critical QLIKE Scaling Unit Correction (`models.py`)**:
   - *Previous Bug*: The original `qlike()` function treated all predictions directly as standard deviations $\sigma$. However, the baseline models, machine learning models, and scaled GARCH models predicted expected absolute return $\mathbb{E}[|R|] = \sigma \sqrt{2/\pi} \approx 0.7979 \sigma$. Squaring these point predictions produced $(0.7979 \sigma)^2 \approx 0.6366 \sigma^2$ in the denominator, artificially distorting QLIKE loss and mistakenly ranking the VIX baseline as #1.
   - *Fix Implemented*: All models predicting $\mathbb{E}[|R|]$ are converted to unbiased conditional standard deviations via $\hat{\sigma} = \text{pred} / \sqrt{2/\pi}$. `Baseline (Vol_21)` is inherently a standard deviation $\sigma$ and is used directly. Raw GARCH/TGARCH/EGARCH forecasts are standard deviations directly.
   - *Numerical Safeguard*: To prevent division-by-zero during low-volatility periods, all $\hat{\sigma}$ forecasts are floored at 0.05: $\hat{\sigma}_{\text{floored}} = \max(\hat{\sigma}, 0.05)$.
   - *Single Reporting*: GARCH-family models are now reported once in the final ranking table. Point metrics (MAE, RMSE) are evaluated on $\mathbb{E}[|R|]$; QLIKE is evaluated on $\hat{\sigma}^2$ with actual variance $y_t^2 = R_{t+1}^2$.
2. **Environment & Pinned Dependencies (`requirements.txt`)**:
   - Exact library versions pinned for 100% deterministic reproducibility in a clean virtual environment: `pandas==3.0.6`, `numpy==2.4.6`, `scipy==1.17.1`, `statsmodels==0.14.6`, `arch==8.0.0`, `scikit-learn==1.8.0`, `xgboost==2.0.2`, `yfinance==1.7.0`, `matplotlib==3.9.2`, `joblib==1.3.2`, `streamlit==1.66.1`, `plotly==5.18.0`, `python-docx==1.2.0`, `tabulate==0.10.0`, `jinja2==3.1.6`, `pywin32==306; sys_platform == 'win32'`.
3. **Deterministic Hyperparameter Tuning**:
   - Pinned random seeds (`random_state=42`), single-threaded execution (`n_jobs=1`), and documented deterministic tie-break rules (lowest validation MAE, then lower tree depth, then lower learning rate).
   - Logged the full validation search grid to `tuning_results.csv`.
4. **GARCH Cache Validation (`garch_cache.csv`, `garch_cache_meta.json`)**:
   - Cache now validates whether test dates match the dataset and whether installed `arch` and `statsmodels` versions match metadata; if invalid or missing, it automatically refits.
   - A note is added to `README.md` that a cold GARCH refit without cache takes several minutes.
5. **Master Pipeline Runner (`run_all.py`)**:
   - Created `run_all.py` executing all pipeline stages sequentially without `plt.show()` interruptions in under 40 seconds.
   - Deleted obsolete test files `spy_test.py` and `vix_test.py` as instructed.
   - Cleaned `README.md` to remove absolute `file:///` URLs, replacing them with relative Markdown links.
6. **Production Model Selection & Dashboard (`app.py`)**:
   - Dashboard loads the calibrated production model (`Ridge Regression`, $\alpha=10$).
   - Displays both $\mathbb{E}[|R|]$ point forecasts, estimated standard deviation $\sigma = \mathbb{E}[|R|]/\sqrt{2/\pi}$, and 1-day 95% Parametric Value-at-Risk dollar boundaries.

---

### Part B: Exploratory Data Analysis (`eda.py`, `eda.ipynb`)
1. **High-Resolution Figures (`figures/`)**:
   - Saved 9 clean EDA figures at 200 dpi with no `plt.show()` calls:
     - Figure 1: SPY Closing Price (`fig01_spy_price.png`)
     - Figure 2: SPY Daily Log Returns (`fig02_spy_returns.png`)
     - Figure 3: SPY Absolute Returns Volatility Proxy (`fig03_spy_abs_returns.png`)
     - Figure 4: CBOE VIX with 20 and 30 threshold lines (`fig04_vix_index.png`)
     - Figure 5: SPY Returns Distribution vs. Gaussian Normal Density (`fig05_returns_distribution.png`)
     - Figure 6: Normal Q-Q Plot proving fat tails / leptokurtosis (`fig06_qq_plot.png`)
     - Figure 7: 4-panel ACF and PACF for Returns and Absolute Returns (`fig07_acf_pacf.png`)
     - Figure 8: Historical 21-day rolling realized volatility (`fig08_rolling_vol21.png`)
     - Figure 9: Pearson correlation bar chart using **strictly the 18 engineered features** (`fig09_feature_correlation.png`).
2. **Exact Econometric Statistics Exported (`eda_results.json`, `eda_results.csv`)**:
   - Descriptive stats for Close, Return, Abs_Return, VIX, Parkinson.
   - Augmented Dickey-Fuller (ADF) tests confirming Close is non-stationary ($p=0.9830$) and Return is strictly stationary ($p=0.0000$).
   - Ljung-Box autocorrelation tests at lags 5, 10, 20 confirming returns are a martingale difference sequence, while absolute returns exhibit massive volatility clustering ($p < 10^{-50}$).
   - Engle's ARCH-LM tests confirming conditional heteroskedasticity ($p = 1.11 \times 10^{-28}$).
   - Jarque-Bera normality tests decisively rejecting Gaussian normality ($p = 0.0000$).
3. **Interactive EDA Notebook (`eda.ipynb`)**:
   - Converted `eda.py` into a fully functional Jupyter notebook.

---

### Part C: Traditional Econometric Models
1. **Parameter Estimation on Train+Val (`garch_params.csv`)**:
   - Estimated GARCH(1,1), TGARCH/TARCH(1,1), and EGARCH(1,1) on 986 sessions.
   - Exported parameters ($\omega, \alpha, \gamma, \beta$, persistence), standard errors, $p$-values, log-likelihood, AIC, and BIC.
   - *Key Econometric Finding*: Asymmetry parameter $\gamma$ is statistically significant in both TGARCH ($\gamma = 0.1761, p=0.0000$) and EGARCH ($\gamma = -0.1418, p=0.0000$).
2. **GARCH Residual Diagnostics (`garch_diagnostics.csv`, `fig12_garch_diagnostics.png`)**:
   - Evaluated standardized residuals $z_t = \epsilon_t / \hat{\sigma}_t$.
   - Ljung-Box on $z_t^2$ yields $p = 0.1699$ (lag 10) and $p = 0.6408$ (lag 20); ARCH-LM yields $p = 0.2057$.
   - *Conclusion*: No remaining ARCH effects detected (though $z_t$ Ljung-Box lag 10 $p=0.051$ is borderline).
3. **Second Traditional Family (C3)**:
   - Added **HAR-RV (OLS)** expanding regression predicting next-session $|R|$ using daily, weekly (5-day), and monthly (21-day) components.
   - Added **ARMA(1,0,2)** on $\ln(|R| + 0.01)$ with order chosen by AIC on the training set, back-transformed to $\mathbb{E}[|R|]$ via log-normal bias correction.

---

### Part D: Machine Learning Models & Diagnostics
1. **Deterministic Grid Tuning (`tuning_results.csv`)**:
   - Ridge: $\alpha \in [0.01, 0.1, 1, 10, 100] \to \text{Optimal } \alpha = 10$ (Val MAE: 0.393676).
   - SVR (RBF): $C \in [0.1, 1, 10, 100], \gamma \in [\text{'scale'}, 0.01, 0.1], \epsilon \in [0.01, 0.05, 0.1] \to \text{Optimal } C=1, \gamma=0.01, \epsilon=0.01$ (Val MAE: 0.373139).
   - XGBoost: $\text{learning\_rate} \in [0.01, 0.05, 0.1], \text{max\_depth} \in [2, 3, 4] \to \text{Optimal } \text{lr}=0.01, \text{depth}=4$ (Val MAE: 0.400272).
2. **Walk-Forward ML Evaluation (`walk_forward_results.csv`)**:
   - Refit Ridge, SVR, and XGBoost every 21 trading days on expanding windows.
   - Walk-forward QLIKE results: Ridge 0.8271 walk-forward vs 0.8135 fixed; SVR 4.7738 walk-forward vs 6.8680 fixed; XGBoost 1.0006 walk-forward vs 0.9693 fixed.
3. **ML Residual Diagnostics (`fig13_ml_diagnostics.png`)**:
   - Residuals vs. fitted values, residual ACF, and normal Q-Q plots for Ridge, SVR, and XGBoost.
4. **Feature Importance Benchmark (`feature_importance.csv`, `fig14_feature_importance.png`)**:
   - Standardized Ridge coefficients vs. XGBoost gain and permutation importance.
   - `VIX`, `VIX_Lag1`, and `Parkinson` rank as the top 3 positive volatility predictors.
5. **SVR Under-Prediction & QLIKE Breakdown (`svr_diagnostics.json`, `svr_top5_qlike.csv`)**:
   - Diagnostic analysis confirms that approximately 81% of SVR's total test QLIKE loss stems from a single forecast origin on 2025-04-03 (where SVR predicted 0.129 vs an actual next-day move of 6.03%).
   - Across the entire 247-day test set, only 1 prediction fell below 0.10%. Extrapolation failure outside the training range remains a plausible hypothesis, though not definitively proven.

---

### Part E: Explaining Why Models Differ
1. **Feature-Set Ablation Analysis (`ablation_results.csv`, `fig15_ablation_analysis.png`)**:
   - Evaluated Ridge and XGBoost on Sets A, B, C, D.
   - *Key Insights*:
     - Ridge with VIX features (Set C) achieves the best QLIKE (0.7876).
     - XGBoost with VIX features deteriorates to QLIKE 1.0949.
     - Return-only Ridge (Set A, QLIKE 0.8926) is worse than univariate EGARCH (0.8056), proving that ML requires multi-information signals to outperform econometric models.
2. **April 2025 Market Shock Analysis (`shock_regime_results.csv`)**:
   - The April 2025 window accounts for **66.41% of Ridge's total test-set squared error**.
   - Excluding April 2025 drops Ridge out-of-sample RMSE from **0.8317% to 0.5039%**, and XGBoost's to **0.5098%**.
3. **Market Volatility Regimes (`vix_regime_results.csv`, `fig16_regime_rolling_rmse.png`)**:
   - Partitioned test set by prior-day VIX: Low (<15), Medium (15–25), High (>25).
4. **Multi-Horizon Forecasting (`horizon_results.csv`, `fig18_multi_horizon.png`)**:
   - Evaluated 1-day, 5-day, and 10-day forward horizons.
   - Methodological note: RMSE declines partly because the multi-day target is an averaged window $\frac{1}{H}\sum_{k=1}^H |R_{t+k}|$, whose temporal aggregation smooths high-frequency noise and reduces target variance. Comparisons must be made strictly within horizon.
5. **Statistical Diebold-Mariano Tests (`dm_tests.csv`, `fig17_dm_heatmap.png`)**:
   - Evaluated under both MSE and QLIKE loss with Harvey-Leybourne-Newbold correction.
   - *Findings*: Only EGARCH ($p=0.030$ MSE, $p=0.001$ QLIKE) and TGARCH ($p=0.014$ MSE, $p=0.0007$ QLIKE) significantly beat GARCH(1,1). Ridge vs GARCH(1,1) is not statistically significant at 5% ($p=0.089$ MSE, $p=0.397$ QLIKE). Ridge significantly beats HAR under QLIKE ($p=0.043$). TGARCH vs Ridge shows no significant difference ($p=0.418$ MSE, $p=0.990$ QLIKE).
6. **Value-at-Risk Backtesting (`var_backtest.csv`)**:
   - At 95%: Ridge and EGARCH produce 18 failures vs 12.4 expected (failure rate 7.29%, Kupiec LR = 2.3989, $p=0.121$, conditional coverage $p=0.251$, not rejected).
   - At 99%: Both produce 7 failures vs 2.5 expected (failure rate 2.83%, Kupiec LR = 5.6081, $p=0.018$, conditional coverage $p=0.024$, REJECTED).
   - Normal-based VaR underestimates tail risk due to fat tails and the April 2025 shock. Removed "satisfies regulatory standards". Fixed table column headers to prevent clipping.

---

### Part F: Report Synchronization & Deliverables
1. **Dynamic Generation (`generate_report.py`)**:
   - `REPORT.md`, `report.docx`, and `report.pdf` are generated strictly from CSV/JSON outputs with zero hardcoded values.
   - Pinned versions of xgboost (2.0.2) and statsmodels (0.14.6) recorded in the appendix.
   - PDF export made optional with try/except around `win32com.client`.
   - Word footer page numbers implemented with dynamic XML fields (`PAGE` and `NUMPAGES`).
2. **Cover Page & Team Details**:
   - Real student names and IDs in format `D/DBA/24/00XX`:
     - Minesi Rajapaksha (`D/DBA/24/0027`)
     - Bimuthu Edirisinghe (`D/DBA/24/0012`)
     - `[PASTE]` (`D/DBA/24/00XX [PASTE]`)
     - `[PASTE]` (`D/DBA/24/00XX [PASTE]`)
     - `[PASTE]` (`D/DBA/24/00XX [PASTE]`)
   - Replaced invented contributions with `"TO BE COMPLETED BY GROUP"`.

---

## 2. Headline Results: Out-of-Sample Benchmark Comparison

| Benchmark Model | MAE (%) | RMSE (%) | QLIKE Loss | Rank | Primary Empirical Characteristic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **EGARCH(1,1)** | **0.4902** | **0.8653** | **0.8056** | **#1** | Mathematically optimal asymmetric variance calibration |
| **TGARCH(1,1)** | **0.4946** | **0.8646** | **0.8118** | **#2** | Explicit threshold leverage modeling |
| **Ridge Regression** | **0.4840** | **0.8317** | **0.8135** | **#3 (Lowest RMSE)** | Continuous quadratic shrinkage minimizes tail risk error |
| **GARCH(1,1)** | **0.5399** | **0.9164** | **0.9244** | **#4** | Symmetric formulation penalizes equity performance |
| **Baseline (VIX)** | **0.5833** | **0.8772** | **0.9354** | **#5** | Market implied baseline; physical models provide superior fit |
| **XGBoost Regressor**| **0.4800** | **0.9004** | **0.9693** | **#6** | Strong point MAE; discrete tree splits overfit tail noise |
| **HAR-RV (OLS)** | **0.5274** | **0.9153** | **1.0034** | **#7** | Traditional long-memory realized volatility benchmark |
| **Baseline (Vol_21)**| **0.5665** | **0.9793** | **1.0675** | **#8** | Historical rolling standard deviation benchmark |
| **ARIMA(1,0,2)** | **0.5380** | **0.9278** | **1.1033** | **#9** | AIC-selected log-volatility ARMA specification |
| **SVR (RBF Kernel)** | **0.4734** | **0.9481** | **6.8680** | **#10** | Lowest MAE; 81% of QLIKE from single shock (2025-04-03) |
| **Naive Persistence** | **0.6794** | **1.1298** | **26.0320** | **#11** | Random walk baseline |

---

## 3. Academic Citations & Verification Notes

The following literature citations are embedded across `REPORT.md` and `report.docx`:
1. **Andersen, T. G., & Bollerslev, T. (1998)**. *[VERIFY: Evaluates volatility forecasts against high-frequency realized volatility proxies, proving standard models are accurate when evaluated against clean proxies.]*
2. **Black, F. (1976)**. *[VERIFY: Establishes the empirical leverage effect where equity price drops increase subsequent volatility.]*
3. **Bollerslev, T. (1986)**. *[VERIFY: Introduces the GARCH(p,q) framework generalizing Engle's ARCH.]*
4. **Chen, T., & Guestrin, C. (2016)**. *[VERIFY: Introduces regularized gradient tree boosting.]*
5. **Christensen, K., Siggaard, M., & Veliyev, B. (2023)**. *[VERIFY: Forecasts realized variance of Dow Jones constituents from high-frequency data and finds ML improves one-day-ahead forecasts.]*
6. **Christie, A. A. (1982)**. *[VERIFY: Tests the financial leverage explanation for asymmetric volatility in equities.]*
7. **Corsi, F. (2009)**. *[VERIFY: Formulates the Heterogeneous Autoregressive model of Realized Volatility (HAR-RV).]*
8. **Drucker, H., et al. (1997)**. *[VERIFY: Formulates Support Vector Regression with epsilon-insensitive loss.]*
9. **Engle, R. F. (1982)**. *[VERIFY: Introduces the ARCH process to model time-varying conditional variance.]*
10. **Fama, E. F. (1970)**. *[VERIFY: Formalizes the Efficient Market Hypothesis and empirical market predictability.]*
11. **Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993)**. *[VERIFY: Introduces the GJR-GARCH / Threshold GARCH model for asymmetric leverage effects.]*
12. **Hansen, P. R., & Lunde, A. (2005)**. *[VERIFY: Benchmarks 330 ARCH-type models on exchange rates and IBM stock returns, finding leverage models show advantages in equity returns.]*
13. **Harvey, D., Leybourne, S., & Newbold, P. (1997)**. *[VERIFY: Introduces small-sample finite-sample correction for the Diebold-Mariano test statistic.]*
14. **Mandelbrot, B. (1963)**. *[VERIFY: Identifies fat tails, leptokurtosis, and volatility clustering in financial assets.]*
15. **Nelson, D. B. (1991)**. *[VERIFY: Introduces Exponential GARCH (EGARCH) with asymmetric log-variance dynamics.]*
16. **Parkinson, M. (1980)**. *[VERIFY: Derives the extreme-value intraday high-low range estimator for diffusion variance.]*
17. **Patton, A. J. (2011)**. *[VERIFY: Identifies robust loss functions (MSE and QLIKE) invariant to proxy noise.]*
18. **Poon, S. H., & Granger, C. W. (2003)**. *[VERIFY: Comprehensive survey of 93 volatility forecasting studies across financial asset classes.]*
19. **Zakoian, J. M. (1994)**. *[VERIFY: Introduces the Threshold GARCH (TGARCH/TARCH) formulation for conditional standard deviation.]*

---

## 4. Acceptance Criteria Verification

| Acceptance Criterion | Verification Status | Evidencing Artifact / Command |
| :--- | :---: | :--- |
| **1. `python run_all.py` passes cleanly** | **PASSED** | Runs all 7 stages sequentially with zero errors and no interruptions. |
| **2. Clean virtual environment install** | **PASSED** | Pinned dependencies install cleanly with zero conflicts; Streamlit and ML libraries verified. |
| **3. All report numbers match CSV/JSONs** | **PASSED** | Zero hard-coded values; verified by automated numerical equality check. |
| **4. VaR Discussion Updated** | **PASSED** | Backtest accurately reports 95% pass (18 vs 12.4, $p=0.121$) and 99% fail (7 vs 2.5, $p=0.018$); normal-based tail risk limitation discussed; unclipped headers implemented. |
| **5. Diebold-Mariano Pairs Added** | **PASSED** | Added EGARCH vs GARCH, Ridge vs GARCH, Ridge vs HAR, TGARCH vs Ridge; reports that only EGARCH and TGARCH beat GARCH(1,1). |
| **6. SVR Diagnostics & Breakdown** | **PASSED** | Recomputed top 5 daily QLIKE loss table; confirmed 81% loss from 2025-04-03; only 1 prediction < 0.10%; removed "calm days" wording. |
| **7. Cover Page & Footer Page Numbers** | **PASSED** | Updated IDs to `D/DBA/24/00XX`, contributions marked "TO BE COMPLETED BY GROUP", and dynamic Word page numbering implemented. |
| **8. Literature Review Claims Softened** | **PASSED** | Corrected Christensen et al. (2023) and Hansen & Lunde (2005) claims; added verification notes for all 19 citations. |
