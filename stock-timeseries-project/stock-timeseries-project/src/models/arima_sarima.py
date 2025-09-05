from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

def fit_arima(train_close: pd.Series, order=(5,1,0)) -> SARIMAX:
    model = SARIMAX(train_close, order=order, enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp=False)
    return res

def fit_sarima(train_close: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,12)) -> SARIMAX:
    model = SARIMAX(train_close, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp=False)
    return res

def forecast(model_fit, steps: int) -> np.ndarray:
    fc = model_fit.forecast(steps=steps)
    return np.asarray(fc)
