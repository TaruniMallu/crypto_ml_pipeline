# src/models/predict_model.py

import argparse
import glob
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import joblib
import numpy as np

def main(args):
    coin = args.coin
    interval = "4h"

    # Load input file
    if not args.input:
        labeled_dir = Path("data/processed_labeled")
        pattern = labeled_dir / f"{coin}_{interval}_labeled.csv"
        files = sorted(glob.glob(str(pattern)), reverse=True)
        if not files:
            raise FileNotFoundError(f"No labeled CSV found for {coin}")
        input_file = files[0]
        print(f"[i] Using latest labeled CSV: {input_file}")
    else:
        input_file = args.input

    df_new = pd.read_csv(input_file)
    print(f"[i] Loaded {len(df_new)} rows from {input_file}")

    # NEW: coin-specific folder
    coin_dir = Path(f"saved_models/{coin}")

    # Load features
    features_path = coin_dir / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file missing: {features_path}")
    with open(features_path, "r") as f:
        feature_cols = json.load(f)

    # Prepare features
    X_new = df_new[feature_cols].values

    # Load scaler
    scaler_path = coin_dir / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file missing: {scaler_path}")
    scaler = joblib.load(scaler_path)
    X_new_scaled = scaler.transform(X_new)

    # Load all models inside coin folder
    models = {}
    for model_name in ["rf", "xgb", "lgbm"]:
        model_path = coin_dir / f"{model_name}.joblib"
        if model_path.exists():
            models[model_name] = joblib.load(model_path)
        else:
            print(f"[!] {model_name} model not found for {coin}")

    if not models:
        raise RuntimeError("No models found. Cannot predict.")

    # Generate predictions
    predictions = {}
    for name, model in models.items():
        df_new[f"pred_{name}"] = model.predict(X_new_scaled)
        predictions[name] = df_new[f"pred_{name}"].values

    # Majority vote
    all_preds = np.vstack(list(predictions.values())).T
    majority_preds = []

    for row in all_preds:
        counts = Counter(row)
        majority_label = min([lbl for lbl, cnt in counts.items() if cnt == max(counts.values())])
        majority_preds.append(majority_label)

    df_new["pred_majority"] = majority_preds

    # Summary
    print("\n[i] Prediction summary:")
    labels_map = {0: "neutral", 1: "pump", 2: "crash"}
    for lbl, count in df_new["pred_majority"].value_counts().sort_index().items():
        print(f"  {labels_map[lbl]}: {count}")

    # Save output
    output_dir = Path("data/predictions")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{coin}_{interval}_predictions.csv"
    df_new.to_csv(output_file, index=False)

    print(f"\n[✔] Predictions saved → {output_file}")
    print("\n[i] Sample predictions:")
    print(df_new.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", required=True)
    parser.add_argument("--input", required=False)
    args = parser.parse_args()
    main(args)
