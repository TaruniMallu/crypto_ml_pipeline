import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from src.features.feature_engineering import add_features

LABELS = {0: "neutral", 1: "pump", 2: "crash"}

def predict(coin, candles):
    coin_dir = Path(f"saved_models/{coin}")

    if not coin_dir.exists():
        raise ValueError(f"No saved models found for coin '{coin}'")

    # load artifacts
    scaler = joblib.load(coin_dir / "scaler.joblib")
    features = json.load(open(coin_dir / "features.json"))

    models = {}
    for name in ["rf", "xgb", "lgbm"]:
        path = coin_dir / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)

    if not models:
        raise RuntimeError("No models found")

    # convert candles → dataframe
    df = pd.DataFrame([c.dict() for c in candles])

    # compute features
    df_feat = add_features(df)

    # fallback if add_features produces empty dataframe
    if df_feat.dropna().empty:
        print("Not enough candles for advanced features, using raw values instead")
        df_feat = df.copy()  # fallback to raw OHLCV
        # add missing feature columns filled with 0
        for f in features:
            if f not in df_feat.columns:
                df_feat[f] = 0

    # ensure all required features are present in correct order
    X = scaler.transform(df_feat.reindex(columns=features, fill_value=0).values)

    # predict with all available models
    votes = {}
    preds = []
    for name, model in models.items():
        p = int(model.predict(X)[-1])  # last candle
        votes[name] = p
        preds.append(p)

    majority = Counter(preds).most_common(1)[0][0]
    confidence = preds.count(majority) / len(preds)

    return {
        "signal": LABELS[majority],
        "confidence": confidence,
        "model_votes": votes
    }
