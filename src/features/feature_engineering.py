import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    df["volatility"] = df["return"].rolling(14).std()

    df["sma_14"] = df["close"].rolling(14).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_14"] = df["close"].ewm(span=14).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    return df
