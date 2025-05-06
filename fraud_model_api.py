from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

# Load your trained model (replace with actual model path)
model = pickle.load(open("xgb_final.pkl", "rb"))

# Define a FastAPI app
app = FastAPI(title="Fraud Detection API")

# Define expected request format
class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        data = np.array(request.features).reshape(1, -1)
        prediction = model.predict(data)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)