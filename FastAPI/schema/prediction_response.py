from model.predict import model
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Dict

class PredictionResponse(BaseModel):
    prediction: Annotated[Literal["low", "medium", "high"], Field(description="Prediction from the model", examples=["low"])]
    confidence_score: Annotated[float, Field(description="Confidence Score of the prediction", examples=[0.8])]
    confidence_probs: Annotated[Dict[str, float], Field("Confidence Scores of the prediction classes", examples=[{"medium": 0, "low": 0.8, "high": 0.2}])]

