import pandas as pd

# Load our combined FinPulse dataset
data = pd.read_csv(
    "FinPulse_Raw_Data.csv",
    parse_dates=["Date"]
)

# First 5 rows
print(data.head())

# Dataset dimensions
print("\nShape:")
print(data.shape)

# Column data types
print("\nData types:")
print(data.dtypes)

# Missing values
print("\nMissing values:")
print(data.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:")
print(data.duplicated().sum())

import numpy as np

# Calculate daily log return
data["Return"] = 100 * np.log(data["Close"] / data["Close"].shift(1))

print("\nData with Return:")
print(data[["Date", "Close", "Return"]].head(10))

# Absolute return as volatility proxy
data["Abs_Return"] = data["Return"].abs()

print("\nReturn and Absolute Return:")
print(data[["Date", "Return", "Abs_Return"]].head(10))

# Next-day absolute return = forecasting target
data["Target"] = data["Abs_Return"].shift(-1)

print("\nCurrent information and next-day target:")
print(data[["Date", "Return", "Abs_Return", "Target"]].head(10))

# Rolling volatility features
data["Vol_5"] = data["Return"].rolling(window=5).std()
data["Vol_10"] = data["Return"].rolling(window=10).std()
data["Vol_21"] = data["Return"].rolling(window=21).std()

print("\nRolling Volatility:")
print(
    data[
        ["Date", "Return", "Vol_5", "Vol_10", "Vol_21"]
    ].head(25)
)

# Lagged return features
data["Return_Lag1"] = data["Return"].shift(1)
data["Return_Lag2"] = data["Return"].shift(2)
data["Return_Lag3"] = data["Return"].shift(3)
data["Return_Lag5"] = data["Return"].shift(5)

# Lagged absolute-return features
data["AbsReturn_Lag1"] = data["Abs_Return"].shift(1)
data["AbsReturn_Lag2"] = data["Abs_Return"].shift(2)
data["AbsReturn_Lag5"] = data["Abs_Return"].shift(5)

print("\nLagged Features:")
print(
    data[
        [
            "Date",
            "Return",
            "Return_Lag1",
            "Return_Lag2",
            "Return_Lag3",
            "Return_Lag5",
            "Abs_Return",
            "AbsReturn_Lag1"
        ]
    ].head(10)
)

# VIX-based features
data["VIX_Lag1"] = data["VIX"].shift(1)

# 1-day change in VIX
data["VIX_Change_1D"] = data["VIX"].diff(1)

# 5-day change in VIX
data["VIX_Change_5D"] = data["VIX"].diff(5)

print("\nVIX Features:")
print(
    data[
        [
            "Date",
            "VIX",
            "VIX_Lag1",
            "VIX_Change_1D",
            "VIX_Change_5D"
        ]
    ].head(10)
)

# Parkinson volatility measure
data["Parkinson"] = np.sqrt(
    (np.log(data["High"] / data["Low"]) ** 2) / (4 * np.log(2))
)

print("\nParkinson Volatility:")
print(
    data[
        ["Date", "High", "Low", "Parkinson"]
    ].head(10)
)

# Daily percentage change in trading volume
data["Volume_Change_1D"] = data["Volume"].pct_change() * 100

print("\nVolume Features:")
print(
    data[
        ["Date", "Volume", "Volume_Change_1D"]
    ].head(10)
)

# Remove rows with missing values created by lagging/rolling calculations
data_clean = data.dropna().copy()

print("\nShape before cleaning:")
print(data.shape)

print("\nShape after cleaning:")
print(data_clean.shape)

print("\nMissing values after cleaning:")
print(data_clean.isnull().sum())

data_clean.to_csv("FinPulse_Processed_Data.csv", index=False)