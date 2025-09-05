from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_close(df: pd.DataFrame, out_path: str):
    plt.figure()
    plt.plot(df["Date"], df["Close"])
    plt.title("Close Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_train_test(train: pd.DataFrame, test: pd.DataFrame, out_path: str):
    plt.figure()
    plt.plot(train["Date"], train["Close"], label="Train")
    plt.plot(test["Date"], test["Close"], label="Test")
    plt.title("Train/Test Split")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
