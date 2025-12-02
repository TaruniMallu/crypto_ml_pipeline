# src/models/predict_model.py
"""
Predict pump/crash/neutral for a given coin using trained ML models.

Usage:
    python predict_model.py --coin BTCUSDT --input data/new/BTCUSDT_4h_new.csv
    python predict_model.py --coin BTCUSDT   # uses latest labeled CSV if input not provided
"""

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
    interval = "4h"  # or load from config.yaml if needed

    # If no input file is provided, pick the latest labeled CSV
    if not args.input:
        labeled_dir = Path("data/processed_labeled")
        pattern = labeled_dir / f"{coin}_{interval}_labeled.csv"
        files = sorted(glob.glob(str(pattern)), reverse=True)
        if not files:
            raise FileNotFoundError(f"No labeled CSV found for {coin} in {labeled_dir}")
        input_file = files[0]
        print(f"[i] No input provided. Using latest labeled CSV: {input_file}")
    else:
        input_file = args.input

    # Load data
    df_new = pd.read_csv(input_file)
    print(f"[i] Loaded {len(df_new)} rows from {input_file}")

    # Load features
    features_path = Path(f"saved_models/{coin}_features.json")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file missing: {features_path}")
    with open(features_path, "r") as f:
        feature_cols = json.load(f)

    # Prepare features
    X_new = df_new[feature_cols].values
    scaler_path = Path(f"saved_models/{coin}_scaler.joblib")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file missing: {scaler_path}")
    scaler = joblib.load(scaler_path)
    X_new_scaled = scaler.transform(X_new)

    # Load models
    models = {}
    for model_name in ["rf", "xgb", "lgbm"]:
        model_path = Path(f"saved_models/{coin}_{model_name}.joblib")
        if model_path.exists():
            models[model_name] = joblib.load(model_path)
        else:
            print(f"[!] Model {model_name} not found for {coin}, skipping.")

    if not models:
        raise RuntimeError("No models loaded. Cannot predict.")

    # Make predictions
    predictions = {}
    for name, model in models.items():
        pred_col = f"pred_{name}"
        df_new[pred_col] = model.predict(X_new_scaled)
        predictions[name] = df_new[pred_col].values

    # Majority vote
    all_preds = np.vstack(list(predictions.values())).T  # shape: (n_samples, n_models)
    majority_preds = []
    for row in all_preds:
        counts = Counter(row)
        majority_label = min([lbl for lbl, cnt in counts.items() if cnt == max(counts.values())])
        majority_preds.append(majority_label)
    df_new["pred_majority"] = majority_preds

    # Summary of predictions
    print("\n[i] Prediction summary (majority vote):")
    summary_counts = df_new["pred_majority"].value_counts().sort_index()
    labels_map = {0: "neutral", 1: "pump", 2: "crash"}
    for lbl, count in summary_counts.items():
        print(f"  {labels_map.get(lbl, lbl)}: {count}")

    # Save predictions to new CSV
    output_dir = Path("data/predictions")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{coin}_{interval}_predictions.csv"
    df_new.to_csv(output_file, index=False)
    print(f"[✔] Predictions saved → {output_file}")

    # Show head
    print("\n[i] Sample predictions:")
    print(df_new.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, required=True, help="Coin symbol (e.g. BTCUSDT)")
    parser.add_argument("--input", type=str, help="CSV file with new data (optional)")
    args = parser.parse_args()
    main(args)
