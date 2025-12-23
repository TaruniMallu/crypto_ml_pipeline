# src/models/train_model.py
"""
Train ML models for multi-class crypto pump/crash detection.

Generates per-coin folders:
 saved_models/<coin>/rf.joblib
 saved_models/<coin>/xgb.joblib
 saved_models/<coin>/lgbm.joblib
 saved_models/<coin>/scaler.joblib
 saved_models/<coin>/features.json
 saved_models/<coin>/metrics.json

Usage:
    python train_model.py --coin BTCUSDT
    python train_model.py  # trains for all coins in config.yaml
"""

import os
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import joblib

# optional libs
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# oversampling
try:
    from imblearn.over_sampling import RandomOverSampler
except Exception:
    RandomOverSampler = None


# ---------------------------
# LOAD CONFIG
# ---------------------------

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_labeled_csv(coin, interval):
    fp = Path("data/processed_labeled") / f"{coin}_{interval}_labeled.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing labeled file: {fp}")
    return pd.read_csv(fp)


# ---------------------------
# FEATURE SELECTION
# ---------------------------

def choose_features(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ["open_time", "close_time", "label", "label_binary", "target_close_shift_6"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude_cols]
    return features


# ---------------------------
# TIME BASED SPLIT
# ---------------------------

def time_series_split(df, test_size=0.2, val_size=0.1):
    n = len(df)
    test_n = int(n * test_size)
    val_n = int(n * val_size)
    train_end = n - test_n - val_n

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:train_end + val_n].copy()
    test_df = df.iloc[train_end + val_n:].copy()
    return train_df, val_df, test_df


def prepare_xy(df, feature_cols, label_col="label"):
    X = df[feature_cols].values
    y = df[label_col].values.astype(int)
    return X, y


# ---------------------------
# TRAIN AND EVAL
# ---------------------------

def train_and_evaluate(models_to_train, X_train, y_train, X_val, y_val, scaler, feature_cols, coin, coin_dir):
    results = {}

    classes = np.unique(y_train)

    for name in models_to_train:
        if name == "rf":
            print("[*] Training RandomForest...")
            cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_train)))
            model = RandomForestClassifier(
                n_estimators=200,
                n_jobs=-1,
                class_weight=cw,
                random_state=42
            )
            model.fit(X_train, y_train)

        elif name == "xgb":
            if xgb is None:
                print("[!] XGBoost missing, skipping.")
                continue

            print("[*] Training XGBoost...")
            model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                n_jobs=-1,
                random_state=42
            )
            model.fit(X_train, y_train)

        elif name == "lgbm":
            if lgb is None:
                print("[!] LightGBM missing, skipping.")
                continue

            print("[*] Training LightGBM...")
            model = lgb.LGBMClassifier(n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)

        else:
            print(f"[!] Unknown model {name}. Skipping.")
            continue

        # Validation eval
        y_pred = model.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

        acc = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average="macro")
        cm = confusion_matrix(y_val, y_pred).tolist()

        # Save validation results
        results[name] = {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "classification_report": report,
            "confusion_matrix": cm
        }

        # Save the model
        out_path = coin_dir / f"{name}.joblib"
        joblib.dump(model, out_path)
        print(f"[✔] Saved {name} → {out_path}")

    # Save scaler
    joblib.dump(scaler, coin_dir / "scaler.joblib")

    # Save feature list
    with open(coin_dir / "features.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"[✔] Saved scaler + features for {coin}")

    return results


# ---------------------------
# MAIN
# ---------------------------

def main(args):
    cfg = load_config()
    coins = cfg.get("coins", [])
    interval = cfg.get("interval", "4h")

    selected_coins = [args.coin] if args.coin else coins
    models_to_train = args.models.split(",") if args.models else ["rf", "xgb", "lgbm"]

    base_dir = Path("saved_models")
    base_dir.mkdir(exist_ok=True)

    for coin in selected_coins:
        print(f"\n====== TRAINING {coin} ======")

        # Create per-coin folder
        coin_dir = base_dir / coin
        coin_dir.mkdir(exist_ok=True)

        df = load_labeled_csv(coin, interval)
        df = df.sort_values("open_time").reset_index(drop=True)

        feature_cols = choose_features(df)
        print(f"[i] Using {len(feature_cols)} features")

        train_df, val_df, test_df = time_series_split(df, test_size=0.15, val_size=0.10)

        # Prepare data
        X_train_raw, y_train = prepare_xy(train_df, feature_cols)
        X_val_raw, y_val = prepare_xy(val_df, feature_cols)
        X_test_raw, y_test = prepare_xy(test_df, feature_cols)

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        # Train + validate
        results = train_and_evaluate(
            models_to_train,
            X_train, y_train,
            X_val, y_val,
            scaler,
            feature_cols,
            coin,
            coin_dir
        )

        # Test evaluation
        for name in results.keys():
            model_path = coin_dir / f"model_{name}.joblib"
            model = joblib.load(model_path)

            y_test_pred = model.predict(X_test)

            acc_test = accuracy_score(y_test, y_test_pred)
            f1_test = f1_score(y_test, y_test_pred, average="macro")
            rep_test = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
            cm_test = confusion_matrix(y_test, y_test_pred).tolist()

            results[name]["test_eval"] = {
                "accuracy": float(acc_test),
                "f1_macro": float(f1_test),
                "classification_report": rep_test,
                "confusion_matrix": cm_test
            }

        # Save metrics
        metrics_path = coin_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[✔] Saved metrics → {metrics_path}")

    print("\n🎉 All models trained successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, help="Coin symbol (BTCUSDT). If omitted, trains all.")
    parser.add_argument("--models", type=str, help="Comma-separated: rf,xgb,lgbm")
    args = parser.parse_args()
    main(args)
