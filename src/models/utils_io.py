import os
import json
import joblib

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_model(coin, model_name, model_obj):
    dir_path = f"saved_models/{coin}"
    ensure_dir(dir_path)
    file_path = f"{dir_path}/{model_name}.pkl"
    joblib.dump(model_obj, file_path)
    return file_path

def save_scaler(coin, scaler_obj):
    dir_path = f"saved_models/{coin}"
    ensure_dir(dir_path)
    file_path = f"{dir_path}/scaler.pkl"
    joblib.dump(scaler_obj, file_path)
    return file_path

def save_feature_list(coin, features):
    dir_path = f"saved_models/{coin}"
    ensure_dir(dir_path)
    save_json(features, f"{dir_path}/features.json")

def save_metrics(coin, metrics_dict):
    dir_path = f"saved_models/{coin}"
    ensure_dir(dir_path)
    save_json(metrics_dict, f"{dir_path}/metrics.json")
