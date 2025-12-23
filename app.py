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

    cmd = [
        "python",
        "src/models/predict_model.py",
        "--coin", req.coin
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ
    )
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
