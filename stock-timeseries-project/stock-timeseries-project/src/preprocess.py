from __future__ import annotations
import pandas as pd
import numpy as np

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Keep essential columns
    cols = [c for c in ["Date","Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    df = df[cols]
    # Sort and drop duplicates
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])
    # Fill missing numerics via forward fill then back fill
    num_cols = [c for c in df.columns if c != "Date"]
    df[num_cols] = df[num_cols].ffill().bfill()
    # Ensure positive volume
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].clip(lower=0)
    return df

def train_test_split_series(df: pd.DataFrame, test_size: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= test_size:
        raise ValueError("Not enough rows to create a test split")
    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()

def make_supervised(series: np.ndarray, lookback: int = 30) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)
