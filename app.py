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
def predict(transaction: Transaction):

    row =[transaction.Time] + transaction.V_features +[transaction.Amount]

    prediction = model.predict([row])[0]

    return {"is_fraud": bool(prediction)}



