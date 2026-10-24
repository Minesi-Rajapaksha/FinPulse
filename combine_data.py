import os
import sys
import pandas as pd
import yfinance as yf

RAW_DATA_FILE = "FinPulse_Raw_Data.csv"

def get_data(force_download=False):
    if not force_download and os.path.exists(RAW_DATA_FILE):
        print(f"Loading existing raw market dataset from '{RAW_DATA_FILE}'...")
        data = pd.read_csv(RAW_DATA_FILE, index_col=0, parse_dates=True)
        print(f"Loaded {len(data)} rows successfully. Pass --download to force re-fetch.")
        return data

    print("Downloading historical SPY and ^VIX series from Yahoo Finance (2021-01-01 to 2026-01-01)...")
    spy = yf.download("SPY", start="2021-01-01", end="2026-01-01", progress=False)
    vix = yf.download("^VIX", start="2021-01-01", end="2026-01-01", progress=False)

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    vix_close = vix[["Close"]].copy()
    vix_close.columns = ["VIX"]

    data = spy.join(vix_close, how="left")
    data.to_csv(RAW_DATA_FILE)
    print(f"Downloaded and saved {len(data)} rows to '{RAW_DATA_FILE}'.")
    return data

if __name__ == "__main__":
    force = "--download" in sys.argv or "--force" in sys.argv
    df = get_data(force_download=force)
    print("Dataset dimensions:", df.shape)
    print("Columns:", list(df.columns))