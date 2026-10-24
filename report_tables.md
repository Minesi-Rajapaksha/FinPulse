### Table 1 Descriptive Stats

| Series     |   Count |     Mean |   Std. Dev. |      Min |      25% |   Median |      75% |      Max |   Skewness |   Excess Kurtosis |
|:-----------|--------:|---------:|------------:|---------:|---------:|---------:|---------:|---------:|-----------:|------------------:|
| Close      |    1233 | 468.057  |     93.9091 | 338.538  | 391.892  | 428.823  | 546.981  | 685.029  |     0.7071 |           -0.7616 |
| Return     |    1233 |   0.0533 |      1.0766 |  -6.0327 |  -0.4533 |   0.0731 |   0.628  |   9.9863 |     0.1402 |            8.2208 |
| Abs_Return |    1233 |   0.7598 |      0.7643 |   0      |   0.2352 |   0.5606 |   1.0619 |   9.9863 |     3.1085 |           22.0242 |
| VIX        |    1233 |  19.2473 |      5.252  |  11.86   |  15.57   |  17.97   |  21.77   |  52.33   |     1.2923 |            2.5391 |
| Parkinson  |    1233 |   0.0072 |      0.0048 |   0.001  |   0.0041 |   0.006  |   0.0087 |   0.0641 |     3.3244 |           24.8435 |

---

### Table 2 Econometric Diagnostics

| Test Name                              | Null Hypothesis (H0)              |   Test Statistic |     p-value | Statistical Conclusion                    |
|:---------------------------------------|:----------------------------------|-----------------:|------------:|:------------------------------------------|
| Augmented Dickey-Fuller (ADF) - Close  | Unit Root (Non-Stationary)        |           0.4428 | 0.983       | Fail to Reject H0 (Non-Stationary)        |
| Augmented Dickey-Fuller (ADF) - Return | Unit Root (Non-Stationary)        |         -21.8916 | 0           | Reject H0 (Strictly Stationary)           |
| Ljung-Box Q (Lag 5) - Return           | No Serial Autocorrelation         |           7.0452 | 0.2173      | Fail to Reject H0 (Martingale Difference) |
| Ljung-Box Q (Lag 10) - Return          | No Serial Autocorrelation         |           9.1892 | 0.5142      | Fail to Reject H0 (Martingale Difference) |
| Ljung-Box Q (Lag 20) - Return          | No Serial Autocorrelation         |          27.0547 | 0.1337      | Fail to Reject H0 (Martingale Difference) |
| Ljung-Box Q (Lag 5) - |Return|         | No Serial Autocorrelation         |         255.115  | 4.3913e-53  | Reject H0 (Strong Volatility Clustering)  |
| Ljung-Box Q (Lag 10) - |Return|        | No Serial Autocorrelation         |         411.906  | 2.7482e-82  | Reject H0 (Strong Volatility Clustering)  |
| Ljung-Box Q (Lag 20) - |Return|        | No Serial Autocorrelation         |         567.354  | 2.1389e-107 | Reject H0 (Strong Volatility Clustering)  |
| Engle's ARCH-LM (LM Statistic)         | No Conditional Heteroskedasticity |         157.404  | 1.1122e-28  | Reject H0 (ARCH Effects Present)          |
| Engle's ARCH-LM (F-Statistic)          | No Conditional Heteroskedasticity |          17.903  | 9.0239e-31  | Reject H0 (ARCH Effects Present)          |
| Jarque-Bera Test - Return              | Gaussian Normal Distribution      |        3443.86   | 0           | Reject H0 (Fat Tails / Leptokurtic)       |
| Jarque-Bera Test - |Return|            | Gaussian Normal Distribution      |       26688.7    | 0           | Reject H0 (Extreme Positive Skew)         |

---

### Table 3 GARCH Parameters

| Model       |    omega |   alpha | gamma (Asymmetry)   |   beta |   Persistence |   Log-Likelihood |     AIC |     BIC |
|:------------|---------:|--------:|:--------------------|-------:|--------------:|-----------------:|--------:|--------:|
| GARCH(1,1)  |  0.02161 |  0.0761 | -                   | 0.9029 |        0.9789 |         -1340.78 | 2689.56 | 2709.13 |
| TGARCH(1,1) |  0.04172 |  0      | 0.1761              | 0.8895 |        0.9776 |         -1314.5  | 2639    | 2663.47 |
| EGARCH(1,1) | -0.00198 |  0.0414 | -0.1418             | 0.9783 |        0.9783 |         -1315.46 | 2640.93 | 2665.4  |

---

### Table 4 GARCH Residual Diagnostics

| Test                        |   Statistic |   p_value |
|:----------------------------|------------:|----------:|
| Ljung-Box on z_t (Lag 10)   |     18.2321 |    0.0512 |
| Ljung-Box on z_t (Lag 20)   |     28.5907 |    0.0961 |
| Ljung-Box on z_t^2 (Lag 10) |     14.068  |    0.1699 |
| Ljung-Box on z_t^2 (Lag 20) |     17.1875 |    0.6408 |
| ARCH-LM on z_t (LM stat)    |     13.3315 |    0.2057 |
| ARCH-LM on z_t (F stat)     |      1.347  |    0.2067 |

---

### Table 5 Hyperparameter Grid

| Model   | Parameters                       |   Val_MAE | Optimal        |
|:--------|:---------------------------------|----------:|:---------------|
| Ridge   | alpha=0.01                       |  0.393841 | No             |
| Ridge   | alpha=0.1                        |  0.393839 | No             |
| Ridge   | alpha=1                          |  0.393822 | No             |
| Ridge   | alpha=10                         |  0.393676 | Yes (Selected) |
| Ridge   | alpha=100                        |  0.395014 | No             |
| SVR     | C=0.1, gamma=scale, epsilon=0.01 |  0.37357  | No             |
| SVR     | C=0.1, gamma=scale, epsilon=0.05 |  0.373173 | No             |
| SVR     | C=0.1, gamma=scale, epsilon=0.1  |  0.374002 | No             |
| SVR     | C=0.1, gamma=0.01, epsilon=0.01  |  0.377943 | No             |
| SVR     | C=0.1, gamma=0.01, epsilon=0.05  |  0.378717 | No             |
| SVR     | C=0.1, gamma=0.01, epsilon=0.1   |  0.378389 | No             |
| SVR     | C=0.1, gamma=0.1, epsilon=0.01   |  0.377137 | No             |
| SVR     | C=0.1, gamma=0.1, epsilon=0.05   |  0.376436 | No             |
| SVR     | C=0.1, gamma=0.1, epsilon=0.1    |  0.376996 | No             |
| SVR     | C=1, gamma=scale, epsilon=0.01   |  0.381541 | No             |
| SVR     | C=1, gamma=scale, epsilon=0.05   |  0.380688 | No             |
| SVR     | C=1, gamma=scale, epsilon=0.1    |  0.378409 | No             |
| SVR     | C=1, gamma=0.01, epsilon=0.01    |  0.373139 | Yes (Selected) |
| SVR     | C=1, gamma=0.01, epsilon=0.05    |  0.373282 | No             |
| SVR     | C=1, gamma=0.01, epsilon=0.1     |  0.373809 | No             |
| SVR     | C=1, gamma=0.1, epsilon=0.01     |  0.380243 | No             |
| SVR     | C=1, gamma=0.1, epsilon=0.05     |  0.377787 | No             |
| SVR     | C=1, gamma=0.1, epsilon=0.1      |  0.376472 | No             |
| SVR     | C=10, gamma=scale, epsilon=0.01  |  0.408038 | No             |
| SVR     | C=10, gamma=scale, epsilon=0.05  |  0.403553 | No             |
| SVR     | C=10, gamma=scale, epsilon=0.1   |  0.399451 | No             |
| SVR     | C=10, gamma=0.01, epsilon=0.01   |  0.391394 | No             |
| SVR     | C=10, gamma=0.01, epsilon=0.05   |  0.389576 | No             |
| SVR     | C=10, gamma=0.01, epsilon=0.1    |  0.38847  | No             |
| SVR     | C=10, gamma=0.1, epsilon=0.01    |  0.42302  | No             |
| SVR     | C=10, gamma=0.1, epsilon=0.05    |  0.422071 | No             |
| SVR     | C=10, gamma=0.1, epsilon=0.1     |  0.422108 | No             |
| SVR     | C=100, gamma=scale, epsilon=0.01 |  0.504219 | No             |
| SVR     | C=100, gamma=scale, epsilon=0.05 |  0.495815 | No             |
| SVR     | C=100, gamma=scale, epsilon=0.1  |  0.488409 | No             |
| SVR     | C=100, gamma=0.01, epsilon=0.01  |  0.408551 | No             |
| SVR     | C=100, gamma=0.01, epsilon=0.05  |  0.406452 | No             |
| SVR     | C=100, gamma=0.01, epsilon=0.1   |  0.401606 | No             |
| SVR     | C=100, gamma=0.1, epsilon=0.01   |  0.504011 | No             |
| SVR     | C=100, gamma=0.1, epsilon=0.05   |  0.494376 | No             |
| SVR     | C=100, gamma=0.1, epsilon=0.1    |  0.485932 | No             |
| XGBoost | lr=0.01, depth=2                 |  0.403668 | No             |
| XGBoost | lr=0.01, depth=3                 |  0.402093 | No             |
| XGBoost | lr=0.01, depth=4                 |  0.400272 | Yes (Selected) |
| XGBoost | lr=0.05, depth=2                 |  0.400766 | No             |
| XGBoost | lr=0.05, depth=3                 |  0.408164 | No             |
| XGBoost | lr=0.05, depth=4                 |  0.405508 | No             |
| XGBoost | lr=0.1, depth=2                  |  0.410678 | No             |
| XGBoost | lr=0.1, depth=3                  |  0.412862 | No             |
| XGBoost | lr=0.1, depth=4                  |  0.404951 | No             |

---

### Table 6 Model Benchmark

|   Rank | Model             |      MAE |     RMSE |     QLIKE | MAE_vs_Naive   | RMSE_vs_Naive   |   DM_p_value |
|-------:|:------------------|---------:|---------:|----------:|:---------------|:----------------|-------------:|
|      1 | EGARCH            | 0.490201 | 0.865333 |  0.805575 | -27.85%        | -23.41%         |       1      |
|      2 | TGARCH            | 0.494643 | 0.864552 |  0.811795 | -27.20%        | -23.48%         |       0.8094 |
|      3 | Ridge             | 0.48396  | 0.831712 |  0.813462 | -28.77%        | -26.38%         |       0.9494 |
|      4 | GARCH(1,1)        | 0.539878 | 0.916353 |  0.924442 | -20.54%        | -18.89%         |       0.0012 |
|      5 | Baseline (VIX)    | 0.583278 | 0.877231 |  0.935427 | -14.15%        | -22.35%         |       0.0986 |
|      6 | XGBoost           | 0.479951 | 0.900352 |  0.969306 | -29.36%        | -20.31%         |       0.3272 |
|      7 | HAR               | 0.527449 | 0.915291 |  1.00336  | -22.37%        | -18.99%         |       0.0102 |
|      8 | Baseline (Vol_21) | 0.56654  | 0.979281 |  1.0675   | -16.61%        | -13.32%         |       0.0216 |
|      9 | ARIMA(1, 0, 2)    | 0.538012 | 0.927835 |  1.10331  | -20.81%        | -17.88%         |       0.0172 |
|     10 | SVR               | 0.473359 | 0.948116 |  6.86797  | -30.33%        | -16.08%         |       0.2767 |
|     11 | Naive             | 0.679408 | 1.12979  | 26.032    | +0.00%         | +0.00%          |       0.0003 |

---

### Table 7 Walk Forward Results

| Model   |   Fixed_MAE |   WalkForward_MAE |   Fixed_RMSE |   WalkForward_RMSE |   Fixed_QLIKE |   WalkForward_QLIKE |
|:--------|------------:|------------------:|-------------:|-------------------:|--------------:|--------------------:|
| Ridge   |      0.484  |            0.4839 |       0.8317 |             0.8312 |        0.8135 |              0.8271 |
| SVR     |      0.4734 |            0.4733 |       0.9481 |             0.9415 |        6.868  |              4.7738 |
| XGBoost |      0.48   |            0.4847 |       0.9004 |             0.9082 |        0.9693 |              1.0006 |

---

### Table 8 Ablation Results

| Feature_Set               | Model       |    MAE |   RMSE |   QLIKE |
|:--------------------------|:------------|-------:|-------:|--------:|
| Univariate GARCH(1,1)     | GARCH(1,1)  | 0.5399 | 0.9164 |  0.9244 |
| Univariate EGARCH(1,1)    | EGARCH(1,1) | 0.4902 | 0.8653 |  0.8056 |
| HAR-RV (OLS)              | HAR         | 0.5274 | 0.9153 |  1.0034 |
| A (Return History Only)   | Ridge       | 0.5157 | 0.8986 |  0.8926 |
| A (Return History Only)   | XGBoost     | 0.4996 | 0.8957 |  0.9109 |
| B (Return + Parkinson)    | Ridge       | 0.499  | 0.8543 |  0.8929 |
| B (Return + Parkinson)    | XGBoost     | 0.4903 | 0.8753 |  0.8151 |
| C (Return + VIX Features) | Ridge       | 0.4863 | 0.8488 |  0.7876 |
| C (Return + VIX Features) | XGBoost     | 0.4849 | 0.9139 |  1.0949 |
| D (All 18 Features)       | Ridge       | 0.484  | 0.8317 |  0.8135 |
| D (All 18 Features)       | XGBoost     | 0.4831 | 0.8978 |  0.937  |

---

### Table 9 April Shock Analysis

| Model             |   Total_RMSE |   April_RMSE |   Non_April_RMSE |   April_SqErr_Share_Pct |   Total_MAE |   April_MAE |   Non_April_MAE |
|:------------------|-------------:|-------------:|-----------------:|------------------------:|------------:|------------:|----------------:|
| Naive             |       1.1298 |       2.9651 |           0.7603 |                   58.56 |      0.6794 |      2.0516 |          0.5519 |
| Baseline (Vol_21) |       0.9793 |       2.7268 |           0.5977 |                   65.92 |      0.5665 |      1.9563 |          0.4374 |
| Baseline (VIX)    |       0.8772 |       2.3365 |           0.5777 |                   60.32 |      0.5833 |      1.5858 |          0.4901 |
| GARCH(1,1)        |       0.9164 |       2.5892 |           0.543  |                   67.88 |      0.5399 |      1.7989 |          0.4229 |
| TGARCH            |       0.8646 |       2.4329 |           0.5166 |                   67.33 |      0.4946 |      1.6221 |          0.3899 |
| EGARCH            |       0.8653 |       2.4424 |           0.5139 |                   67.73 |      0.4902 |      1.5787 |          0.3891 |
| HAR               |       0.9153 |       2.596  |           0.5379 |                   68.39 |      0.5274 |      1.7415 |          0.4146 |
| Ridge             |       0.8317 |       2.3246 |           0.5039 |                   66.41 |      0.484  |      1.5826 |          0.3819 |
| SVR               |       0.9481 |       2.7948 |           0.5066 |                   73.88 |      0.4734 |      1.7203 |          0.3575 |
| XGBoost           |       0.9004 |       2.6119 |           0.502  |                   71.55 |      0.48   |      1.6183 |          0.3742 |

---

### Table 10 VIX Regimes

| Model             | Regime                              |   Days |    MAE |   RMSE |   QLIKE |
|:------------------|:------------------------------------|-------:|-------:|-------:|--------:|
| Naive             | Low Volatility (VIX < 15)           |     21 | 0.299  | 0.3641 |  9.4389 |
| Naive             | Medium Volatility (15 <= VIX <= 25) |    206 | 0.599  | 0.8479 | 28.6775 |
| Naive             | High Volatility / Panic (VIX > 25)  |     20 | 1.9067 | 2.867  | 16.2065 |
| Baseline (Vol_21) | Low Volatility (VIX < 15)           |     21 | 0.2527 | 0.2964 | -0.4751 |
| Baseline (Vol_21) | Medium Volatility (15 <= VIX <= 25) |    206 | 0.5081 | 0.7773 |  0.9812 |
| Baseline (Vol_21) | High Volatility / Panic (VIX > 25)  |     20 | 1.4976 | 2.3513 |  3.5763 |
| Baseline (VIX)    | Low Volatility (VIX < 15)           |     21 | 0.4119 | 0.469  |  0.0527 |
| Baseline (VIX)    | Medium Volatility (15 <= VIX <= 25) |    206 | 0.5389 | 0.7213 |  0.8696 |
| Baseline (VIX)    | High Volatility / Panic (VIX > 25)  |     20 | 1.2206 | 1.9782 |  2.5407 |
| GARCH(1,1)        | Low Volatility (VIX < 15)           |     21 | 0.2999 | 0.3437 | -0.3076 |
| GARCH(1,1)        | Medium Volatility (15 <= VIX <= 25) |    206 | 0.4787 | 0.7121 |  0.8332 |
| GARCH(1,1)        | High Volatility / Panic (VIX > 25)  |     20 | 1.4215 | 2.2414 |  3.1578 |
| TGARCH            | Low Volatility (VIX < 15)           |     21 | 0.2762 | 0.3154 | -0.3969 |
| TGARCH            | Medium Volatility (15 <= VIX <= 25) |    206 | 0.4394 | 0.6798 |  0.7427 |
| TGARCH            | High Volatility / Panic (VIX > 25)  |     20 | 1.2934 | 2.0898 |  2.793  |
| EGARCH            | Low Volatility (VIX < 15)           |     21 | 0.2907 | 0.339  | -0.3364 |
| EGARCH            | Medium Volatility (15 <= VIX <= 25) |    206 | 0.4377 | 0.6771 |  0.7184 |
| EGARCH            | High Volatility / Panic (VIX > 25)  |     20 | 1.2399 | 2.0986 |  2.9029 |
| HAR               | Low Volatility (VIX < 15)           |     21 | 0.3043 | 0.3394 | -0.328  |
| HAR               | Medium Volatility (15 <= VIX <= 25) |    206 | 0.4688 | 0.7205 |  0.9181 |
| HAR               | High Volatility / Panic (VIX > 25)  |     20 | 1.3658 | 2.2087 |  3.28   |
| Ridge             | Low Volatility (VIX < 15)           |     21 | 0.2458 | 0.2877 | -0.5363 |
| Ridge             | Medium Volatility (15 <= VIX <= 25) |    206 | 0.4398 | 0.6915 |  0.7973 |
| Ridge             | High Volatility / Panic (VIX > 25)  |     20 | 1.1889 | 1.8792 |  2.3967 |
| SVR               | Low Volatility (VIX < 15)           |     21 | 0.2137 | 0.2476 | -0.7819 |
| SVR               | Medium Volatility (15 <= VIX <= 25) |    206 | 0.4196 | 0.7234 |  7.4014 |
| SVR               | High Volatility / Panic (VIX > 25)  |     20 | 1.3002 | 2.3764 |  9.4058 |
| XGBoost           | Low Volatility (VIX < 15)           |     21 | 0.2282 | 0.2668 | -0.5935 |
| XGBoost           | Medium Volatility (15 <= VIX <= 25) |    206 | 0.4317 | 0.6971 |  0.8282 |
| XGBoost           | High Volatility / Panic (VIX > 25)  |     20 | 1.2412 | 2.2207 |  4.0634 |

---

### Table 11 Multi Horizon Results

| Horizon   | Model   |    MAE |   RMSE |   QLIKE |
|:----------|:--------|-------:|-------:|--------:|
| 1-Day     | Ridge   | 0.4847 | 0.8332 |  0.8144 |
| 1-Day     | XGBoost | 0.484  | 0.8995 |  0.9387 |
| 1-Day     | HAR     | 0.5257 | 0.9158 |  1.0043 |
| 1-Day     | EGARCH  | 0.492  | 0.8671 |  0.807  |
| 5-Day     | Ridge   | 0.2631 | 0.4629 |  0.508  |
| 5-Day     | XGBoost | 0.2677 | 0.5145 |  0.5331 |
| 5-Day     | HAR     | 0.3102 | 0.5389 |  0.6162 |
| 5-Day     | EGARCH  | 0.2914 | 0.5204 |  0.5747 |
| 10-Day    | Ridge   | 0.2668 | 0.4531 |  0.5861 |
| 10-Day    | XGBoost | 0.2518 | 0.4635 |  0.5837 |
| 10-Day    | HAR     | 0.2923 | 0.4784 |  0.6081 |
| 10-Day    | EGARCH  | 0.2895 | 0.4682 |  0.5561 |

---

### Table 12 Diebold Mariano Tests

| Model_1   | Model_2        | Loss_Function       |   DM_Stat |   p_value | Significant_05   | Favored_Model   |
|:----------|:---------------|:--------------------|----------:|----------:|:-----------------|:----------------|
| EGARCH    | GARCH(1,1)     | Squared Error (MSE) |   -2.1886 |    0.0296 | True             | EGARCH          |
| EGARCH    | GARCH(1,1)     | QLIKE Loss          |   -3.2757 |    0.0012 | True             | EGARCH          |
| TGARCH    | GARCH(1,1)     | Squared Error (MSE) |   -2.4807 |    0.0138 | True             | TGARCH          |
| TGARCH    | GARCH(1,1)     | QLIKE Loss          |   -3.4407 |    0.0007 | True             | TGARCH          |
| Ridge     | GARCH(1,1)     | Squared Error (MSE) |   -1.7101 |    0.0885 | False            | No Difference   |
| Ridge     | GARCH(1,1)     | QLIKE Loss          |   -0.8484 |    0.397  | False            | No Difference   |
| Ridge     | HAR            | Squared Error (MSE) |   -1.579  |    0.1156 | False            | No Difference   |
| Ridge     | HAR            | QLIKE Loss          |   -2.0368 |    0.0427 | True             | Ridge           |
| TGARCH    | Ridge          | Squared Error (MSE) |    0.8119 |    0.4177 | False            | No Difference   |
| TGARCH    | Ridge          | QLIKE Loss          |   -0.0128 |    0.9898 | False            | No Difference   |
| EGARCH    | Ridge          | Squared Error (MSE) |    0.7199 |    0.4723 | False            | No Difference   |
| EGARCH    | Ridge          | QLIKE Loss          |   -0.0636 |    0.9494 | False            | No Difference   |
| Ridge     | XGBoost        | Squared Error (MSE) |   -0.9999 |    0.3183 | False            | No Difference   |
| Ridge     | XGBoost        | QLIKE Loss          |   -1.1335 |    0.2581 | False            | No Difference   |
| EGARCH    | TGARCH         | Squared Error (MSE) |    0.0689 |    0.9451 | False            | No Difference   |
| EGARCH    | TGARCH         | QLIKE Loss          |   -0.2415 |    0.8094 | False            | No Difference   |
| EGARCH    | Baseline (VIX) | Squared Error (MSE) |   -0.4319 |    0.6662 | False            | No Difference   |
| EGARCH    | Baseline (VIX) | QLIKE Loss          |   -1.6582 |    0.0986 | False            | No Difference   |
| Ridge     | Baseline (VIX) | Squared Error (MSE) |   -1.6015 |    0.1106 | False            | No Difference   |
| Ridge     | Baseline (VIX) | QLIKE Loss          |   -0.9046 |    0.3665 | False            | No Difference   |
| XGBoost   | Baseline (VIX) | Squared Error (MSE) |    0.4191 |    0.6755 | False            | No Difference   |
| XGBoost   | Baseline (VIX) | QLIKE Loss          |    0.1785 |    0.8585 | False            | No Difference   |
| EGARCH    | Naive          | Squared Error (MSE) |   -2.8166 |    0.0052 | True             | EGARCH          |
| EGARCH    | Naive          | QLIKE Loss          |   -3.6724 |    0.0003 | True             | EGARCH          |
| Ridge     | Naive          | Squared Error (MSE) |   -2.545  |    0.0115 | True             | Ridge           |
| Ridge     | Naive          | QLIKE Loss          |   -3.6686 |    0.0003 | True             | Ridge           |
| XGBoost   | Naive          | Squared Error (MSE) |   -2.2795 |    0.0235 | True             | XGBoost         |
| XGBoost   | Naive          | QLIKE Loss          |   -3.6494 |    0.0003 | True             | XGBoost         |

---

### Table 6b SVR Top5 QLIKE

|   Rank | Date       |   SVR Pred (%) |   Actual Move (%) |   Daily QLIKE | Share of Total QLIKE (%)   |
|-------:|:-----------|---------------:|------------------:|--------------:|:---------------------------|
|      1 | 2025-04-03 |         0.1294 |            6.0327 |     1380.73   | 81.39%                     |
|      2 | 2025-04-08 |         0.793  |            9.9863 |      100.956  | 5.95%                      |
|      3 | 2025-04-21 |         0.3296 |            2.5685 |       36.8966 | 2.18%                      |
|      4 | 2025-10-09 |         0.4341 |            2.74   |       24.145  | 1.42%                      |
|      5 | 2025-04-09 |         0.7689 |            4.4808 |       21.5465 | 1.27%                      |

---

### Table 13 VaR Backtesting

| Model          | Conf   |   Exp Fail |   Act Fail | Fail %   |   Kupiec LR |   Kupiec p |   Christ LR |   Christ p |   Cond Cov p |
|:---------------|:-------|-----------:|-----------:|:---------|------------:|-----------:|------------:|-----------:|-------------:|
| Ridge          | 95%    |       12.4 |         18 | 7.29%    |      2.3989 |     0.1214 |      0.364  |     0.5463 |       0.2512 |
| Ridge          | 99%    |        2.5 |          7 | 2.83%    |      5.6081 |     0.0179 |      1.8246 |     0.1768 |       0.0243 |
| EGARCH         | 95%    |       12.4 |         18 | 7.29%    |      2.3989 |     0.1214 |      1.9383 |     0.1639 |       0.1143 |
| EGARCH         | 99%    |        2.5 |          7 | 2.83%    |      5.6081 |     0.0179 |      1.8246 |     0.1768 |       0.0243 |
| Baseline (VIX) | 95%    |       12.4 |          9 | 3.64%    |      1.0518 |     0.3051 |      4.5766 |     0.0324 |       0.06   |
| Baseline (VIX) | 99%    |        2.5 |          5 | 2.02%    |      2.0185 |     0.1554 |      3.1313 |     0.0768 |       0.0762 |

---

### Table 14 Feature Importance

| Feature          |   Ridge Std Coef |   XGB Gain |   XGB Permutation MAE |
|:-----------------|-----------------:|-----------:|----------------------:|
| Abs_Return       |          -0.1196 |     0.0389 |               -0.0158 |
| VIX_Lag1         |           0.0994 |     0.146  |                0.0306 |
| VIX              |           0.0957 |     0.1623 |                0.0214 |
| Parkinson        |           0.0925 |     0.0412 |                0.0151 |
| Return_Lag5      |          -0.0815 |     0.0583 |                0.006  |
| AbsReturn_Lag1   |          -0.0506 |     0.0475 |                0.001  |
| Vol_21           |           0.0444 |     0.0425 |                0.0056 |
| Vol_5            |           0.0442 |     0.045  |                0.0086 |
| Return_Lag1      |          -0.0334 |     0.0434 |                0.0051 |
| VIX_Change_5D    |           0.0289 |     0.0419 |               -0.0021 |
| Vol_10           |           0.0281 |     0.0417 |                0.0045 |
| AbsReturn_Lag5   |           0.0237 |     0.0351 |                0.0006 |
| Return_Lag2      |           0.0207 |     0.0467 |               -0.0126 |
| Return           |          -0.0142 |     0.0346 |               -0.0015 |
| VIX_Change_1D    |          -0.0119 |     0.0559 |               -0.0157 |
| Return_Lag3      |          -0.0117 |     0.036  |                0.015  |
| Volume_Change_1D |          -0.0074 |     0.0418 |               -0.0052 |
| AbsReturn_Lag2   |          -0.0069 |     0.0412 |               -0.0031 |

---

