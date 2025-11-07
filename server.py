
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model/model.pkl")

@app.get("/predict")
def predict(price: int):
    return {"predicted": int(model.predict(np.array([[price]]))[0])}
