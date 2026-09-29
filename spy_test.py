import yfinance as yf

spy = yf.download(
    "SPY",
    start="2021-01-01",
    end="2026-01-01"
)

print(spy.head())

print(spy.columns)
print(spy.shape)
print(spy.tail())

spy.to_csv("SPY_2021_2025.csv")