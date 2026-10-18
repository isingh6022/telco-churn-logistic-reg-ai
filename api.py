# FastAPI service for the churn model.

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from model.all_names import (Gender, YesNo, SeniorFlag, PhoneAddOn, InternetType, InternetAddOn,
                             ContractType, PaymentType)
from model.models    import MODEL_PATH, predictProba

app = FastAPI(title='Telco Churn Inference API')

# Load the model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f'Failed to load model: {e}')

# Expected JSON input: one customer, with the same field names and values as the csv.
# The allowed values of each categorical field are the Literal types in all_names.
# This will be given a UI so that the requests can be sent.
class CustomerRequest(BaseModel):
    gender          : Gender
    SeniorCitizen   : SeniorFlag
    Partner         : YesNo
    Dependents      : YesNo
    tenure          : int   = Field(ge=0, description='months with the company')
    PhoneService    : YesNo
    MultipleLines   : PhoneAddOn
    InternetService : InternetType
    OnlineSecurity  : InternetAddOn
    OnlineBackup    : InternetAddOn
    DeviceProtection: InternetAddOn
    TechSupport     : InternetAddOn
    StreamingTV     : InternetAddOn
    StreamingMovies : InternetAddOn
    Contract        : ContractType
    PaperlessBilling: YesNo
    PaymentMethod   : PaymentType
    MonthlyCharges  : float = Field(ge=0)
    TotalCharges    : float = Field(ge=0)

    model_config = ConfigDict(json_schema_extra={'example': {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No', 'tenure': 2,
        'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 95.5, 'TotalCharges': 190.0,
    }})

# JSON output
class PredictionResponse(BaseModel):
    will_churn       : bool
    churn_probability: float


# Controllers
@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.post('/predict', response_model=PredictionResponse)
def predict_churn(data: CustomerRequest):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded. Run "python -m model.train" first.')

    # one row of raw values; predictProba encodes and scales it the same way as in training
    features = pd.DataFrame([data.model_dump()])

    try:
        probability = float(predictProba(model, features)[0])
        return PredictionResponse(will_churn=probability >= 0.5, churn_probability=round(probability, 4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
