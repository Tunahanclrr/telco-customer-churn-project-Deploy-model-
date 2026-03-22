from fastapi import FastAPI, Request
import pandas as pd
import joblib
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="api/templates")

with open("models/lgbm_churn_model.pkl", "rb") as f:
    model = joblib.load(f)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")



@app.post("/predict")
async def predict(features: ModelInput):
    input_data = pd.DataFrame([features.model_dump()])
    proba = model.predict_proba(input_data)[:, 1][0]
    return {"churn_probability": float(proba)}
    

class ModelInput(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: int
    PaymentMethod: str
    MonthlyCharges: float


@app.post("/predict")
async def predict(features: ModelInput):

    input_data = pd.DataFrame([features.model_dump()])

    print("Received input data:")
    print(input_data)

    # Tahmin
    proba = model.predict_proba(input_data)[:, 1][0]

    print("Predicted probability of churn:", proba)
    print("-" * 40)

    return {
        "churn_probability": float(proba)
    }