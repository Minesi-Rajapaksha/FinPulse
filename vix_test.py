import yfinance as yf

vix = yf.download(
    "^VIX",
    start="2021-01-01",
    end="2026-01-01"
)

print(vix.head())
print(vix.tail())
print(vix.shape)
print(vix.columns)

vix.to_csv("VIX_2021_2025.csv")