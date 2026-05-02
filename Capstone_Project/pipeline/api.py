from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

app = FastAPI(title="Fraud Detection API")

# Load model globally on startup to save time per request
model = XGBClassifier()
try:
    model.load_model('xgb_model.json')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model: {e}")

class Transaction(BaseModel):
    step: int
    type: str # 'TRANSFER' or 'CASH_OUT'
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int = 0

class PredictionResponse(BaseModel):
    isFraud: int
    fraudProbability: float

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):
    try:
        # Compute derived features
        errorBalanceOrig = transaction.newbalanceOrig + transaction.amount - transaction.oldbalanceOrg
        errorBalanceDest = transaction.oldbalanceDest + transaction.amount - transaction.newbalanceDest
        is_transfer = 1 if transaction.type.upper() == 'TRANSFER' else 0
        
        # Build the feature vector in exact order expected by the model
        feature_dict = {
            'step': [transaction.step],
            'amount': [transaction.amount],
            'oldbalanceOrg': [transaction.oldbalanceOrg],
            'newbalanceOrig': [transaction.newbalanceOrig],
            'oldbalanceDest': [transaction.oldbalanceDest],
            'newbalanceDest': [transaction.newbalanceDest],
            'errorBalanceOrig': [errorBalanceOrig],
            'errorBalanceDest': [errorBalanceDest],
            'is_transfer': [is_transfer]
        }
        
        # Create DataFrame
        df = pd.DataFrame(feature_dict)
        
        # Predict probability
        prob = model.predict_proba(df)[0][1]
        
        # Determine strict prediction
        is_fraud = 1 if prob >= 0.5 else 0
        
        return PredictionResponse(
            isFraud=is_fraud,
            fraudProbability=float(prob)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
