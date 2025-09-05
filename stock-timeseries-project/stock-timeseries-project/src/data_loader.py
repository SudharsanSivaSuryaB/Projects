from __future__ import annotations
import pandas as pd
from datetime import datetime
from typing import Optional
import os

def load_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load OHLCV data using yfinance. Returns a DataFrame with:
    Date, Open, High, Low, Close, Adj Close, Volume
    """
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")
    df = df.reset_index()  # ensure Date column
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def load_from_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df
