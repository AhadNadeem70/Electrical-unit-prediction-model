from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and features
model = joblib.load("trained_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


app = FastAPI(title="LightGBM Energy Prediction API")

# Define input format
class EnergyInput(BaseModel):
    frequency: int
    voltagePhaseR: float
    currentPhaseR: float
    powerfactorPhaseR: float
    realpowerphaseR: float
    totalLineCurrent: float
    totalRealPower: float
    hour: int
    day: int
    month: int
 

@app.get("/")
def home():
    return {"message": "LightGBM Energy Prediction API is running!"}

@app.post("/predict")
def predict(data: EnergyInput):
    df = pd.DataFrame([data.dict()])[feature_columns]
    prediction = model.predict(df)
    return {"prediction": float(prediction[0])}
