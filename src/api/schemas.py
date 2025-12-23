from pydantic import BaseModel
from typing import List, Dict

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictRequest(BaseModel):
    coin: str
    candles: List[Candle]

class PredictResponse(BaseModel):
    signal: str
    confidence: float
    model_votes: Dict[str, int]
