from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import uuid
import os

app = FastAPI(title="Crypto ML Prediction API")

class PredictRequest(BaseModel):
    coin: str
    interval: str = "4h"

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(req: PredictRequest):
    run_id = str(uuid.uuid4())[:8]

    # Step 1: fetch latest candles
    from src.utils.api import fetch_latest_klines

    output_csv = f"data/new/{req.coin}_4h_new.csv"
    fetch_latest_klines(req.coin, interval=req.interval, save_path=output_csv)

    # Step 2: run prediction on new data
    cmd = [
        "python",
        "src/models/predict_model.py",
        "--coin", req.coin,
        "--input", output_csv
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()

    if process.returncode != 0:
        return {
            "status": "error",
            "error": err.decode()
        }

    return {
        "status": "success",
        "coin": req.coin,
        "run_id": run_id
    }
