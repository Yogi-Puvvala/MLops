from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import model, MODEL_VERSION

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Insurance Premium Prediction API"}

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "version": MODEL_VERSION,
        "model loaded": model is not None
    }

@app.post('/predict', response_model=PredictionResponse)
def predict_premium(data: UserInput):

    input_df = pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }])

    try:
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0] 
        confidence_score = round(max(probs), 2)
        class_labels = model.classes_
        confidence_probs = {label: round(prob, 2) for label, prob in zip(class_labels, probs)}

        return JSONResponse(
            status_code=200,
            content={
                'predicted_category': prediction,
                "confidence_score": confidence_score,
                "confidence_probs": confidence_probs 
            }
        )

    
    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))



