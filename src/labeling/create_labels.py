# src/labeling/create_labels.py
"""
Create pump/crash/neutral labels based on future returns.
Reads: data/processed/*.csv
Writes: data/processed_labeled/*.csv
"""

import pandas as pd
import os
import yaml
from tqdm import tqdm


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_labels(df, horizon=6, pump_thr=0.03, crash_thr=-0.03):
    """
    Creates:
        - label: 0 neutral, 1 pump, 2 crash
        - label_binary: 1 (pump), -1 (crash), 0 (neutral)
    """
    target_col = f"future_return_{horizon}"

    df["label"] = 0  # neutral default
    df.loc[df[target_col] > pump_thr, "label"] = 1       # pump
    df.loc[df[target_col] < crash_thr, "label"] = 2      # crash

    # also binary label if needed
    df["label_binary"] = 0
    df.loc[df[target_col] > pump_thr, "label_binary"] = 1
    df.loc[df[target_col] < crash_thr, "label_binary"] = -1

    return df


def process_file(input_path, output_path, horizon=6):
    df = pd.read_csv(input_path)

    df = create_labels(df, horizon=horizon)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✔] Labeled file saved → {output_path}")


def main():
    cfg = load_config()
    symbols = cfg["coins"]
    interval = cfg["interval"]

    processed_path = "data/processed"
    labeled_path = "data/processed_labeled"

    for sym in tqdm(symbols, desc="Creating Labels"):
        input_file = os.path.join(processed_path, f"{sym}_{interval}_processed.csv")
        output_file = os.path.join(labeled_path, f"{sym}_{interval}_labeled.csv")

        if os.path.exists(input_file):
            process_file(input_file, output_file)
        else:
            print(f"[!] Processed file missing for {sym}")


if __name__ == "__main__":
    main()
