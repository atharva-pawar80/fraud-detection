from fastapi import FastAPI
from pydantic import BaseModel
import joblib


model = joblib.load('fraud_detection.pkl')


app = FastAPI()


class Transaction(BaseModel):

    Time : float
    V_features : list[float]
    Amount : float


@app.post("/predict")
def predict(transacrion: Transaction):

    row =[transacrion.Time] + transacrion.V_features +[transacrion.Amount]

    prediction = model.predcit([row][0])

    return {"is_fraud": bool(predict)}
