from fastapi import FastAPI
from utils.model import load_model

app = FastAPI()

model = load_model()

@app.get("/predict")
def predict(temp: float, hour: int):
    return {"prediction": model.predict([[temp, hour]])[0]}