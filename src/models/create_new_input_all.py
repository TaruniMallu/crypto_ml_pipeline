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
    features_file = Path(f"saved_models/{coin}_features.json")
    new_file = Path(f"data/new/{coin}_{interval}_new.csv")

    if not labeled_file.exists() or not features_file.exists():
        print(f"[!] Missing files for {coin}, skipping...")
        continue

    # Load labeled data
    df = pd.read_csv(labeled_file)

    # Load features
    with open(features_file, "r") as f:
        feature_cols = json.load(f)

    # Keep all rows and only feature columns
    df_new = df[feature_cols]

    # Ensure directory exists
    new_file.parent.mkdir(parents=True, exist_ok=True)

    # Save new CSV
    df_new.to_csv(new_file, index=False)
    print(f"[✔] New input CSV saved → {new_file}")
