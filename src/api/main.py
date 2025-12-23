from fastapi import FastAPI, HTTPException
from src.api.schemas import PredictRequest, PredictResponse
from src.api.predictor import predict

app = FastAPI(title="Crypto Pump/Crash Predictor")

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    try:
        result = predict(
            coin=req.coin,
            candles=req.candles
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
