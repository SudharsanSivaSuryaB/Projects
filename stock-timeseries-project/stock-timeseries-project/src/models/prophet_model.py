from __future__ import annotations
import pandas as pd
import numpy as np

def fit_prophet(train_df: pd.DataFrame):
    """
    Expects train_df with columns: Date, Close
    """
    from prophet import Prophet
    df = train_df[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df)
    return m

def forecast(m, horizon: int, last_date: pd.Timestamp) -> pd.DataFrame:
    future = m.make_future_dataframe(periods=horizon, freq="D")
    fcst = m.predict(future)
    return fcst[fcst["ds"] > last_date]
