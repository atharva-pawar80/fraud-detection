from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.responses import HTMLResponse


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

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Fraud Detection API</title>
        <style>
            body { font-family: sans-serif; max-width: 700px; margin: 50px auto; background: #12161C; color: #E8ECEF; }
            h2 { color: #FF7A45; }
            textarea { width: 100%; height: 200px; background: #1B212A; color: #E8ECEF; border: 1px solid #2A323D; border-radius: 6px; padding: 10px; font-family: monospace; }
            button { background: #FF7A45; color: #14100D; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; cursor: pointer; margin-top: 10px; margin-right: 10px; }
            #result { margin-top: 20px; padding: 15px; background: #1B212A; border-radius: 6px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h2>Fraud Detection API</h2>
        <p>Paste a transaction JSON below, or load an example.</p>
        <textarea id="input"></textarea><br>
        <button onclick="loadExample('legit')">Load Legit Example</button>
        <button onclick="loadExample('fraud')">Load Fraud Example</button>
        <button onclick="submitTransaction()">Check Transaction</button>
        <div id="result"></div>

        <script>
        const examples = {
            legit: {"Time": 136131.0, "V_features": [-0.224816, 1.821721, 0.131439, 3.348614, 2.582168, 1.628348, 1.247632, 0.055309, -1.925839, 0.424950, 1.761240, -0.522738, -0.486871, -2.660399, -0.181033, 0.386488, 2.047155, -0.182370, -1.443739, 0.026941, -0.285439, -0.430957, -0.036161, -1.723991, -1.039929, -0.213176, -0.208386, -0.207902], "Amount": 0.9},
            fraud: {"Time": 102625.0, "V_features": [-4.221221, 2.871121, -5.888716, 6.890952, -3.404894, -1.154394, -7.739928, 2.851363, -2.507569, -5.110728, 5.350890, -9.299807, 2.793140, -6.106552, -2.106947, -6.250629, -13.566325, -4.192780, 0.510570, -0.227882, 1.620591, 1.567947, -0.578007, -0.059045, -1.829169, -0.072429, 0.136734, -0.599848], "Amount": 7.59}
        };

        function loadExample(type) {
            document.getElementById('input').value = JSON.stringify(examples[type], null, 2);
        }

        async function submitTransaction() {
            const raw = document.getElementById('input').value;
            const resultDiv = document.getElementById('result');
            try {
                const data = JSON.parse(raw);
                resultDiv.innerText = "Checking...";
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                resultDiv.innerText = JSON.stringify(result, null, 2);
            } catch (err) {
                resultDiv.innerText = "Error: " + err;
            }
        }
        </script>
    </body>
    </html>
    """

