from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

def build_lstm(input_len: int) -> "keras.Model":
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except Exception as e:
        raise RuntimeError("TensorFlow is required for LSTM: pip install tensorflow") from e

    model = keras.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def fit_lstm(series: np.ndarray, lookback: int = 30, epochs: int = 10, batch_size: int = 32) -> Tuple[object, int, float, float]:
    """
    series: 1D numpy array of Close prices (train only)
    Returns: (model, lookback, train_mean, train_std) for scaling and forecasting
    """
    import numpy as np
    series = np.asarray(series).astype(float)
    # scale
    mean, std = series.mean(), series.std() if series.std() != 0 else 1.0
    series_std = (series - mean) / std

    # supervised
    X, y = [], []
    for i in range(lookback, len(series_std)):
        X.append(series_std[i-lookback:i])
        y.append(series_std[i])
    X, y = np.array(X)[...,None], np.array(y)  # add feature dim

    model = build_lstm(lookback)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, lookback, mean, std

def forecast_lstm(model, history: np.ndarray, steps: int, lookback: int, mean: float, std: float) -> np.ndarray:
    import numpy as np
    hist = np.asarray(history).astype(float)
    series_std = (hist - mean) / (std if std != 0 else 1.0)
    window = series_std[-lookback:].copy()
    preds = []
    for _ in range(steps):
        x = window.reshape(1, lookback, 1)
        yhat_std = model.predict(x, verbose=0)[0,0]
        yhat = yhat_std * (std if std != 0 else 1.0) + mean
        preds.append(yhat)
        window = np.concatenate([window[1:], [yhat_std]])
    return np.array(preds)
