# src/models/train_model.py
"""
Train ML models for multi-class crypto pump/crash detection.

Generates:
 - saved_models/<coin>_rf.joblib
 - saved_models/<coin>_xgb.joblib
 - saved_models/<coin>_lgbm.joblib
 - saved_models/<coin>_scaler.joblib
 - saved_models/<coin>_features.json
 - saved_models/<coin>_metrics.json

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

from sklearn.model_selection import train_test_split, TimeSeriesSplit
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


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_labeled_csv(coin, interval):
    fp = Path("data/processed_labeled") / f"{coin}_{interval}_labeled.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing labeled file: {fp}")
    df = pd.read_csv(fp)
    return df


def choose_features(df, exclude_cols=None):
    """
    Heuristic selection:
    - Remove time columns and raw text / irrelevant columns
    - Keep numeric indicators and lag/return columns
    """
    if exclude_cols is None:
        exclude_cols = ["open_time", "close_time", "label", "label_binary", "target_close_shift_6"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude_cols]
    # ensure we don't accidentally include true-future columns (already excluded above)
    return features


def time_series_split(df, test_size=0.2, val_size=0.1):
    """
    Simple time-based split:
    - Train: first (1 - test_size - val_size)
    - Val: next val_size
    - Test: last test_size
    """
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


def handle_imbalance(X_train, y_train, method="class_weight"):
    """
    method:
      - "class_weight": rely on classifier's class_weight (we compute weights)
      - "oversample": RandomOverSampler (requires imblearn)
    Returns X_res, y_res, sample_weights (or None)
    """
    if method == "oversample":
        if RandomOverSampler is None:
            raise ImportError("imblearn not installed. Install imbalanced-learn or choose class_weight.")
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_train, y_train)
        return X_res, y_res, None
    else:
        # compute sample weights via class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weight_map = {cls: w for cls, w in zip(classes, class_weights)}
        sample_weights = np.array([weight_map[int(lbl)] for lbl in y_train])
        return X_train, y_train, sample_weights


def train_and_evaluate(models_to_train, X_train, y_train, X_val, y_val, scaler, feature_cols, coin):
    results = {}
    saved_models = {}

    # Optionally, compute class weights for classifiers that accept class_weight
    classes = np.unique(y_train)

    for name in models_to_train:
        if name == "rf":
            print("[*] Training RandomForest...")
            # compute class weights dict
            cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_train)))
            model = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight=cw, random_state=42)
            model.fit(X_train, y_train)
        elif name == "xgb":
            if xgb is None:
                print("[!] xgboost not installed, skipping XGBoost.")
                continue
            print("[*] Training XGBoost...")
            # using sklearn API
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1, random_state=42)
            # xgboost can use scale_pos_weight but multi-class needs different handling; use class weights via sample_weight
            model.fit(X_train, y_train)
        elif name == "lgbm":
            if lgb is None:
                print("[!] lightgbm not installed, skipping LightGBM.")
                continue
            print("[*] Training LightGBM...")
            model = lgb.LGBMClassifier(n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)
        else:
            print(f"[!] Unknown model name: {name}. Skipping.")
            continue

        # Evaluate on validation
        y_val_pred = model.predict(X_val)
        y_val_proba = None
        try:
            y_val_proba = model.predict_proba(X_val)
        except Exception:
            pass

        report = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
        acc = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred, average="macro")

        cm = confusion_matrix(y_val, y_val_pred).tolist()

        results[name] = {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "classification_report": report,
            "confusion_matrix": cm
        }

        # Save the model
        model_path = Path("saved_models") / f"{coin}_{name}.joblib"
        joblib.dump(model, model_path)
        saved_models[name] = str(model_path)
        print(f"[✔] Saved {name} model to {model_path}")

    # Save scaler and features
    scaler_path = Path("saved_models") / f"{coin}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    features_path = Path("saved_models") / f"{coin}_features.json"
    with open(features_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"[✔] Saved scaler and features for {coin}")

    return results, saved_models


def main(args):
    cfg = load_config()
    coins = cfg.get("coins", [])
    interval = cfg.get("interval", "4h")

    selected_coins = [args.coin] if args.coin else coins
    models_to_train = args.models.split(",") if args.models else ["rf", "xgb", "lgbm"]

    os.makedirs("saved_models", exist_ok=True)

    for coin in selected_coins:
        print(f"\n=== Processing {coin} ===")
        df = load_labeled_csv(coin, interval)
        # sort (just to be safe)
        df = df.sort_values("open_time").reset_index(drop=True)

        # Choose features
        feature_cols = choose_features(df)
        print(f"[i] Number of features chosen: {len(feature_cols)}")

        # Train/val/test split (time-series)
        train_df, val_df, test_df = time_series_split(df, test_size=0.15, val_size=0.10)
        print(f"[i] Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

        # Prepare X/y
        X_train_raw, y_train = prepare_xy(train_df, feature_cols)
        X_val_raw, y_val = prepare_xy(val_df, feature_cols)
        X_test_raw, y_test = prepare_xy(test_df, feature_cols)

        # Scale features (fit on train only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        # Handle imbalance (choose method via args)
        method = args.imbalance_method
        if method == "oversample":
            print("[i] Using RandomOverSampler to balance classes on training data")
            if RandomOverSampler is None:
                raise ImportError("Install imbalanced-learn to use oversampling.")
            ros = RandomOverSampler(random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        else:
            print("[i] Using class_weight technique (no oversampling)")

        # Train models and save
        results, saved = train_and_evaluate(models_to_train, X_train, y_train, X_val, y_val, scaler, feature_cols, coin)

        # Save metrics + test evaluation
        metrics_out = {
            "coin": coin,
            "models_trained": list(results.keys()),
            "validation_metrics": results
        }

        # Evaluate saved models on test set and append
        for name, model_path in saved.items():
            model = joblib.load(model_path)
            y_test_pred = model.predict(X_test)
            acc_test = accuracy_score(y_test, y_test_pred)
            f1_test = f1_score(y_test, y_test_pred, average="macro")
            report_test = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
            cm_test = confusion_matrix(y_test, y_test_pred).tolist()
            metrics_out["validation_metrics"][name]["test_eval"] = {
                "accuracy": float(acc_test),
                "f1_macro": float(f1_test),
                "classification_report": report_test,
                "confusion_matrix": cm_test
            }

        # Save metrics JSON
        metrics_path = Path("saved_models") / f"{coin}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_out, f, indent=2)
        print(f"[✔] Saved metrics to {metrics_path}")

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, help="Coin symbol (e.g. BTCUSDT). If omitted, trains on all coins in config.")
    parser.add_argument("--models", type=str, help="Comma-separated models: rf,xgb,lgbm")
    parser.add_argument("--imbalance_method", type=str, default="class_weight", help="class_weight or oversample")
    args = parser.parse_args()
    main(args)
