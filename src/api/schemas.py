from pydantic import BaseModel
from typing import List

class SymptomRequest(BaseModel):
    text: str

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[DiseasePrediction]
    language: str
    advice: str
