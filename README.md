# 📈 FinPulse: Real-World Financial Time Series Analytics

FinPulse is an end-to-end financial time-series forecasting and risk analytics platform developed for the Real-World Financial Time Series Analytics assignment. It forecasts the short-term magnitude of equity market volatility (next-session SPDR S&P 500 ETF Trust / `SPY` price movement) using historical price action, technical features, and market-implied uncertainty (`^VIX`).

---

## 📑 Full Academic Report
The complete, comprehensive project report covering the introduction, research question, data sources, exploratory data analysis, econometric tests, model architectures, hyperparameter tuning, empirical benchmark performance, model choosing criteria, and business interpretations is available in **[`REPORT.md`](file:///d:/GitHubProjects/FinPulse/REPORT.md)**.

---

## 🚀 Key Project Highlights
- **Financial Problem**: S&P 500 (`SPY`) next-session volatility magnitude forecasting for risk management and option positioning.
- **Dataset**: 1,233 daily observations from Yahoo Finance (February 2021 to December 2025) merging SPY OHLCV with CBOE VIX.
- **Validation Design**: Strict chronological time-based splitting (60% Train, 20% Validation, 20% Test) to eliminate data leakage and lookahead bias.
- **Models Benchmarked**:
  1. *Baselines*: Naive Persistence ($\hat{\sigma}_{t+1} = |R_t|$), Historical Rolling Volatility (`Vol_21`), and Implied Volatility (`VIX / sqrt(252)` scaled by $\sqrt{2/\pi}$)
  2. *Traditional Econometric*: GARCH(1,1), TGARCH(1,1), EGARCH(1,1) evaluated both raw and scaled by $\sqrt{2/\pi}$ with expanding-window re-estimation
  3. *Machine Learning*: Ridge Regression ($\alpha=10$), Support Vector Regression (SVR RBF: $C=1, \gamma=0.01, \epsilon=0.01$), and XGBoost Regressor
- **Statistical Testing**: Diebold-Mariano tests under squared-error loss against the best-scoring model and between top candidates.
- **Production Model**: Ridge Regression is selected and saved as `best_model.pkl` with metadata in `model_info.json`, powering the interactive Streamlit engine in [`app.py`](file:///d:/GitHubProjects/FinPulse/app.py).

---

## 📁 Repository Structure
```
├── app.py                      # Interactive Streamlit dashboard for real-time volatility forecasting
├── combine_data.py             # Data retrieval from Yahoo Finance for SPY and ^VIX
├── preprocessing.py            # Feature engineering pipeline (18 volatility & lag predictors)
├── eda.py                      # Exploratory data analysis & econometric tests (ADF, Ljung-Box, ARCH-LM)
├── models.py                   # Model training, SVR tuning, expanding-window evaluation, DM tests & exports
├── FinPulse_Raw_Data.csv       # Raw merged market data (2021–2025)
├── FinPulse_Processed_Data.csv # Cleaned feature-engineered dataset
├── garch_cache.csv             # Cached rolling GARCH/TGARCH/EGARCH forecasts
├── model_comparison.csv        # Comprehensive out-of-sample benchmark comparison table
├── actual_vs_predicted_top3.png# Plot of actual vs predicted next-session volatility for top 3 models
├── model_metrics_barchart.png  # Performance metrics bar chart across all evaluated models
├── best_model.pkl              # Serialized selected production model (Ridge Regression)
├── scaler.pkl                  # Serialized StandardScaler
├── feature_cols.pkl            # Serialized feature column names
├── model_info.json             # Selected model metadata, parameters, and test metrics
├── Group Assignment.pdf        # Course assignment specification
├── REPORT.md                   # Comprehensive academic and financial evaluation report
└── requirements.txt            # Python dependencies
```

---

## 💻 Running the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Preprocessing & Feature Engineering
```bash
python combine_data.py
python preprocessing.py
```

### 3. Run Exploratory Data Analysis & Tests
```bash
python eda.py
```

### 4. Train, Compare, and Export Models
```bash
python models.py
```
*This evaluates all baseline, GARCH, and ML models, runs Diebold-Mariano significance tests, exports `model_comparison.csv`, generates visualization charts (`actual_vs_predicted_top3.png`, `model_metrics_barchart.png`), and serializes `best_model.pkl` and `model_info.json`.*

### 5. Launch Interactive Web Application
```bash
streamlit run app.py
```
