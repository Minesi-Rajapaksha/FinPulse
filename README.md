# 📈 FinPulse: Real-World Financial Time Series Analytics

FinPulse is an end-to-end quantitative financial time-series forecasting, econometric diagnostic, and risk analytics platform developed for the university assignment: **"Real-World Financial Time Series Analytics"**. 

It investigates the short-term predictability of equity market volatility (next-session SPDR S&P 500 ETF Trust / `SPY` price movements) using historical price action, econometric conditional heteroskedasticity models (GARCH family, HAR-RV, ARMA), machine learning algorithms (Ridge, SVR, XGBoost), and forward-looking market-implied uncertainty (CBOE Volatility Index / `^VIX`).

---

## 📑 Full Academic Report Documents
The complete academic research report is provided in three synchronized formats:
- **Markdown Source**: [`REPORT.md`](REPORT.md)
- **Formatted Microsoft Word Document**: [`report.docx`](report.docx)
- **Exported Adobe PDF Document**: [`report.pdf`](report.pdf)
- **Interactive EDA Jupyter Notebook**: [`eda.ipynb`](eda.ipynb)
- **Complete Pipeline Changelog & Audit**: [`CHANGES.md`](CHANGES.md)

---

## 🚀 Key Empirical Highlights
- **Financial Problem**: S&P 500 (`SPY`) next-session volatility magnitude forecasting for dynamic portfolio risk management and options market-making.
- **Dataset**: 1,233 contiguous trading sessions (February 3, 2021 – December 30, 2025) merging SPY OHLCV with CBOE VIX without lookahead bias.
- **Validation Design**: Strict chronological time-based splitting (60% Train: 739 sessions, 20% Validation: 247 sessions, 20% Test: 247 sessions).
- **Models Benchmarked (11 Models Across 4 Paradigms)**:
  1. *Baselines*: Naive Persistence ($|R_t|$), 21-Day Realized Volatility (`Vol_21`), and Implied Volatility (`VIX / sqrt(252)`).
  2. *Traditional Econometric GARCH*: Symmetric GARCH(1,1), Asymmetric Threshold GARCH (TGARCH/TARCH), and Exponential GARCH (EGARCH), re-estimated via expanding rolling windows.
  3. *Second Traditional Family*: Corsi's HAR-RV expanding regression and AIC-selected ARMA(1,0,2) on log-volatility with bias correction.
  4. *Machine Learning*: Regularized $L_2$ Ridge Regression ($\alpha=10$), Support Vector Regression (SVR with RBF kernel), and Extreme Gradient Boosting (XGBoost).
- **Corrected QLIKE Resolution**: Correcting the variance conversion unit bug establishes that **EGARCH achieves the optimal volatility loss (QLIKE = 0.8056)**, followed closely by **TGARCH (0.8118)** and **Ridge (0.8135)**. The VIX baseline ranks 5th (0.9354), showing that econometric and regularized machine learning models provide superior volatility calibrations relative to raw implied volatility.
- **Tail Risk Error Minimization**: **Ridge Regression achieves the lowest overall RMSE (0.8317)** of all 11 models due to its smooth quadratic regularization plane that dampens large squared errors during extreme market sell-offs.
- **Shock Resilience (April 2025 Window)**: The April 2025 shock (VIX peak 52.33 on April 8, -6.03% on April 4, +9.99% on April 9) accounts for **66.41% of total test squared error**. Excluding April 2025, Ridge's out-of-sample RMSE drops to **0.5039%**.
- **Information Advantage vs. Algorithms**: Feature ablation confirms that ML's point accuracy advantage over traditional GARCH is driven by access to **VIX and Parkinson intraday range signals**, rather than algorithmic complexity alone.
- **Production Deployment**: Ridge Regression is selected and deployed to the interactive Streamlit engine in [`app.py`](app.py).

---

## 📁 Repository Structure
```
├── app.py                      # Interactive Streamlit dashboard for real-time volatility forecasting
├── combine_data.py             # Data retrieval and offline cache preservation for SPY and ^VIX
├── preprocessing.py            # Feature engineering pipeline (18 volatility & lag predictors)
├── eda.py                      # Exploratory data analysis, econometric tests, and Figures 1-9
├── eda.ipynb                   # Executable Jupyter notebook for exploratory data analysis
├── models.py                   # Model estimation, deterministic tuning, walk-forward, Figures 10-14
├── analysis.py                 # Feature ablation, shock window, VIX regimes, DM tests, Figures 15-18
├── build_report_tables.py      # Automated compilation of all statistics into Markdown & LaTeX tables
├── generate_report.py          # Document synthesis generating REPORT.md, report.docx, and report.pdf
├── run_all.py                  # One-click master runner executing the full reproducible pipeline
├── requirements.txt            # Exact pinned package dependencies for full reproducibility
├── FinPulse_Raw_Data.csv       # Raw merged market data (2021–2025)
├── FinPulse_Processed_Data.csv # Cleaned feature-engineered dataset (1,233 rows)
├── garch_cache.csv             # Cached rolling GARCH/TGARCH/EGARCH forecasts (247 test days)
├── garch_cache_meta.json       # Version and date validation metadata for GARCH cache
├── predictions_test.csv        # Out-of-sample test predictions for all 11 evaluated models
├── model_comparison.csv        # Benchmark ranking table across all models (MAE, RMSE, QLIKE, DM)
├── model_info.json             # Production deployment metadata, parameters, and conversion formulas
├── garch_params.csv            # Estimated GARCH, TGARCH, EGARCH parameters, SEs, AIC, BIC
├── garch_diagnostics.csv       # Ljung-Box and ARCH-LM tests on standardized GARCH residuals
├── tuning_results.csv          # Complete hyperparameter validation search grids for Ridge, SVR, XGB
├── walk_forward_results.csv    # Fixed-fit vs. 21-day walk-forward expanding ML evaluation
├── ablation_results.csv        # Feature ablation benchmarks isolating VIX and Parkinson information
├── shock_regime_results.csv    # Performance inside vs. outside the April 2025 stress shock window
├── vix_regime_results.csv      # Out-of-sample metrics categorized across VIX volatility regimes
├── top10_errors.csv            # Largest 10 test-set forecasting errors per model
├── horizon_results.csv         # Multi-horizon term-structure forecasting (1-Day, 5-Day, 10-Day)
├── dm_tests.csv                # Pairwise Diebold-Mariano tests with Harvey-Leybourne-Newbold correction
├── var_backtest.csv            # Regulatory 1-day 95% and 99% Value-at-Risk backtest results
├── feature_importance.csv      # Standardized Ridge weights vs. XGBoost gain and permutation importance
├── svr_diagnostics.json        # SVR under-prediction percentiles evidencing its extreme QLIKE loss
├── best_model.pkl              # Serialized production model (Ridge Regression)
├── scaler.pkl                  # Serialized StandardScaler
├── feature_cols.pkl            # Serialized feature column names
├── ridge_model.pkl             # Backward-compatible Ridge model artifact
├── figures/                    # High-resolution (200 dpi) visual analytics figures (Figures 1 to 18)
│   ├── fig01_spy_price.png
│   ├── fig02_spy_returns.png
│   ├── fig03_spy_abs_returns.png
│   ├── fig04_vix_index.png
│   ├── fig05_returns_distribution.png
│   ├── fig06_qq_plot.png
│   ├── fig07_acf_pacf.png
│   ├── fig08_rolling_vol21.png
│   ├── fig09_feature_correlation.png
│   ├── fig10_actual_vs_predicted.png
│   ├── fig11_model_comparison_bars.png
│   ├── fig12_garch_diagnostics.png
│   ├── fig13_ml_diagnostics.png
│   ├── fig14_feature_importance.png
│   ├── fig15_ablation_analysis.png
│   ├── fig16_regime_rolling_rmse.png
│   ├── fig17_dm_heatmap.png
│   └── fig18_multi_horizon.png
├── REPORT.md                   # Comprehensive academic evaluation report
├── report.docx                 # Fully formatted academic Word document with embedded figures
├── report.pdf                  # Exported PDF report
├── CHANGES.md                  # Detailed changelog, old vs. new results, open issues, citation audit
└── Group Assignment.pdf        # University assignment specification
```

---

## 💻 Reproducibility & Running the Pipeline

### 1. Environment Setup
Install the exact pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Full Master Pipeline (One-Click)
To re-run the entire pipeline from scratch, execute:
```bash
python run_all.py
```
This sequentially runs `combine_data.py` -> `preprocessing.py` -> `eda.py` -> `models.py` -> `analysis.py` -> `build_report_tables.py` -> `generate_report.py`, regenerating every table, figure, and report document in approximately 40 seconds using cached GARCH rolling forecasts.

*(Note: A cold GARCH refit without cache takes several minutes to re-estimate expanding-window models across 247 test sessions. Cached forecasts in `garch_cache.csv` enable rapid verification.)*

### 3. Individual Script Execution
If running components modularly:
```bash
# 1. Feature Preprocessing
python preprocessing.py

# 2. Exploratory Data Analysis & Econometric Diagnostics (Figures 1-9)
python eda.py

# 3. Model Benchmark & Walk-Forward Training (Figures 10-14)
python models.py

# 4. In-Depth Analysis: Ablation, Shocks, Regimes, Multi-Horizon, DM (Figures 15-18)
python analysis.py

# 5. Build Tables & Generate Report Documents (MD, DOCX, PDF)
python build_report_tables.py
python generate_report.py
```

### 4. Interactive Web Application
Launch the real-time Streamlit dashboard:
```bash
streamlit run app.py
```
The application loads the calibrated production model (`best_model.pkl`), presents $E[|R|]$ point forecasts, converts to conditional standard deviation $\sigma = E[|R|]/\sqrt{2/\pi}$, and provides real-time 95% Parametric Value-at-Risk dollar boundaries.
