import pandas as pd
import matplotlib.pyplot as plt

# Load processed dataset
data = pd.read_csv(
    "FinPulse_Processed_Data.csv",
    parse_dates=["Date"]
)

# Basic summary statistics
print("\nSummary Statistics:")
print(data[["Close", "Return", "Abs_Return", "VIX"]].describe())

plt.figure(figsize=(12, 5))

plt.plot(data["Date"], data["Close"])

plt.title("SPY Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")

plt.show()

plt.figure(figsize=(12, 5))

plt.plot(data["Date"], data["Return"])

plt.title("SPY Daily Log Returns")
plt.xlabel("Date")
plt.ylabel("Return (%)")

plt.show()

plt.figure(figsize=(12, 5))

plt.plot(data["Date"], data["Abs_Return"])

plt.title("SPY Absolute Returns")
plt.xlabel("Date")
plt.ylabel("Absolute Return (%)")

plt.show()

plt.figure(figsize=(12, 5))

plt.plot(data["Date"], data["VIX"])

plt.title("VIX")
plt.xlabel("Date")
plt.ylabel("VIX")

plt.show()

plt.figure(figsize=(8, 5))

plt.hist(data["Return"], bins=50)

plt.title("Distribution of SPY Daily Returns")
plt.xlabel("Return (%)")
plt.ylabel("Frequency")

plt.show()

print("\nSkewness:")
print(data["Return"].skew())

print("\nKurtosis:")
print(data["Return"].kurt())

from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(data["Return"].dropna())

print("\nADF Test:")
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

from statsmodels.stats.diagnostic import acorr_ljungbox

ljung_box = acorr_ljungbox(
    data["Return"].dropna(),
    lags=[10],
    return_df=True
)

print("\nLjung-Box Test:")
print(ljung_box)

from statsmodels.stats.diagnostic import het_arch

arch_test = het_arch(data["Return"].dropna())

print("\nARCH Test:")
print("LM Statistic:", arch_test[0])
print("LM p-value:", arch_test[1])




#py eda.py

