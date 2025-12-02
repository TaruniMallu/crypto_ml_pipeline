# src/ingestion/fetch_data.py
"""
Fetch OHLCV klines from Binance REST API and save to CSV.
Usage:
    python fetch_data.py --symbol BTCUSDT --interval 4h --limit 1000
"""

import requests
import pandas as pd
import argparse
import os
from datetime import datetime
from dateutil import tz
from tqdm import tqdm
import yaml

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol: str, interval: str = "4h", limit: int = 1000):
    """
    Fetches klines (OHLCV) for a symbol and returns a pandas DataFrame.
    Binance returns:
    [ open_time, open, high, low, close, volume, close_time, ... ]
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # columns according to Binance API
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)

    # convert numeric columns
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # convert timestamps (ms) to human readable in UTC and local timezone
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    return df

def save_raw(df: pd.DataFrame, symbol: str, interval: str, data_path: str):
    os.makedirs(data_path, exist_ok=True)
    filename = f"{symbol}_{interval}_raw.csv"
    filepath = os.path.join(data_path, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved raw data to: {filepath}")

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=False, help="Symbol e.g. BTCUSDT")
    parser.add_argument("--interval", type=str, required=False, help="Interval e.g. 4h")
    parser.add_argument("--limit", type=int, required=False, default=1000)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = [args.symbol] if args.symbol else cfg.get("coins", [])
    interval = args.interval if args.interval else cfg.get("interval", "4h")
    limit = args.limit if args.limit else cfg.get("limit", 1000)
    data_path = cfg.get("data_path", "data/raw")

    for sym in tqdm(symbols, desc="Fetching symbols"):
        print(f"Fetching {sym} | interval={interval} | limit={limit}")
        df = fetch_klines(sym, interval=interval, limit=limit)
        save_raw(df, sym, interval, data_path)

if __name__ == "__main__":
    main()
