from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import pycaret
from pycaret.datasets import get_data
from pycaret.regression import *
import pandas as pd
from joblib import load
from  data_preprocessing import preprocessing
import numpy as np

df = pd.read_csv("./clean_data.csv")
setup(
    data=df,
    target="price",
)


app = FastAPI()
model = load("./best_price_model.joblib")


class PropertyInput(BaseModel):
    locality_name: str
    postal_code: int
    price : int
    type_of_property: str
    subtype_of_property: str
    number_of_rooms: int
    living_area: int
    equipped_kitchen: int
    furnished: float
    open_fire: float
    terrace: float
    garden: float
    number_of_facades: int
    swimming_pool: float
    state_of_building: str
    garden_surface: float
    terrace_surface: int

class PropertyOutput(BaseModel):
    prediction: float

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Price prediction API is running"}

@app.post("/predict", response_model=PropertyOutput)
def predict(data: PropertyInput):
    df = preprocessing(pd.DataFrame([data.model_dump()]))

    feature_names = load("./model_features.joblib") #load the feature names used during training
    X_dummy = df.reindex(columns=feature_names, fill_value=0) # reindex dummy to have all training features, fill missing with 0
    preds = model.predict(X_dummy)

    scaler = load("./price_scaler.joblib")
    y_scaled = preds.reshape(-1, 1)
    real_price = scaler.inverse_transform(y_scaled)[:, 0]

    return {"prediction": float(real_price[0])}
