# src/features/technical_indicators.py
"""
Generate technical indicators and processed ML-ready features.
Reads: data/raw/*.csv
Writes: data/processed/*.csv
"""

import pandas as pd
import numpy as np
import os
import yaml
from tqdm import tqdm


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------
# Helper Functions
# ------------------------

def add_returns(df):
    """Add simple returns and log returns."""
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_volatility(df, window=14):
    """Rolling standard deviation of log returns."""
    df["volatility"] = df["log_return"].rolling(window).std()
    return df


def add_sma(df, window=14):
    df[f"sma_{window}"] = df["close"].rolling(window).mean()
    return df


def add_ema(df, window=14):
    df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
    return df


def add_rsi(df, window=14):
    """Relative Strength Index (RSI)."""
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df):
    """MACD indicator (12, 26, 9)."""
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger(df, window=20):
    sma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()

    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    return df


def add_shifted_targets(df, horizon=6):
    """
    Add shifted close price for future predictions.
    e.g., if horizon=6, predict 6 candles (~24 hours ahead for 4h interval).
    """
    df[f"target_close_shift_{horizon}"] = df["close"].shift(-horizon)
    
    # future return label
    df[f"future_return_{horizon}"] = (
        df[f"target_close_shift_{horizon}"] - df["close"]
    ) / df["close"]

    return df


# ------------------------
# MAIN PROCESS
# ------------------------

def process_file(input_path, output_path):
    df = pd.read_csv(input_path)

    # Convert timestamps if saved as string
    df["open_time"] = pd.to_datetime(df["open_time"])
    df["close_time"] = pd.to_datetime(df["close_time"])

    # Sort just in case
    df = df.sort_values("open_time")

    # Add indicators
    df = add_returns(df)
    df = add_volatility(df)
    df = add_sma(df, 14)
    df = add_sma(df, 50)
    df = add_ema(df, 14)
    df = add_ema(df, 50)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_shifted_targets(df, horizon=6)   # 6 * 4h = 24 hours future prediction

    # Drop rows with NaN from rolling windows
    df = df.dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✔] Processed saved → {output_path}")


def main():
    cfg = load_config()
    symbols = cfg["coins"]
    interval = cfg["interval"]

    raw_path = "data/raw"
    processed_path = "data/processed"

    for sym in tqdm(symbols, desc="Processing Indicators"):
        input_file = os.path.join(raw_path, f"{sym}_{interval}_raw.csv")
        output_file = os.path.join(processed_path, f"{sym}_{interval}_processed.csv")

        if os.path.exists(input_file):
            process_file(input_file, output_file)
        else:
            print(f"[!] Raw file missing for {sym}")


if __name__ == "__main__":
    main()
