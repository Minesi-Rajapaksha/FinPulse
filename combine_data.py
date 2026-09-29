import yfinance as yf
import pandas as pd

# Download SPY
spy = yf.download(
    "SPY",
    start="2021-01-01",
    end="2026-01-01"
)

# Download VIX
vix = yf.download(
    "^VIX",
    start="2021-01-01",
    end="2026-01-01"
)

# Remove the extra ticker level from the column names
spy.columns = spy.columns.get_level_values(0)
vix.columns = vix.columns.get_level_values(0)

# Keep only VIX closing value
vix_close = vix[["Close"]].copy()

# Rename Close to VIX
vix_close.columns = ["VIX"]

# Combine using the Date index
data = spy.join(vix_close, how="left")

print(data.head())
print(data.tail())
print(data.shape)
print(data.columns)

# Save
data.to_csv("FinPulse_Raw_Data.csv")