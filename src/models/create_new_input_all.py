# src/models/create_new_input_all.py

import pandas as pd
import json
from pathlib import Path
import yaml

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

coins = cfg.get("coins", [])
interval = cfg.get("interval", "4h")

for coin in coins:
    labeled_file = Path(f"data/processed_labeled/{coin}_{interval}_labeled.csv")

    # NEW: features inside coin-specific folder
    features_file = Path(f"saved_models/{coin}/features.json")

    new_file = Path(f"data/new/{coin}_{interval}_new.csv")

    # Validate required files
    if not labeled_file.exists():
        print(f"[!] Missing labeled file for {coin}, skipping...")
        continue

    if not features_file.exists():
        print(f"[!] Missing features.json for {coin} → expected at {features_file}, skipping...")
        continue

    # Load labeled data
    df = pd.read_csv(labeled_file)

    # Load chosen features
    with open(features_file, "r") as f:
        feature_cols = json.load(f)

    # Extract only the feature columns
    df_new = df[feature_cols]

    # Ensure output directory exists
    new_file.parent.mkdir(parents=True, exist_ok=True)

    # Save
    df_new.to_csv(new_file, index=False)
    print(f"[✔] New input CSV saved → {new_file}")
