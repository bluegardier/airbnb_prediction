from fastapi import APIRouter, Body
import pandas as pd
from pycaret.regression import load_model, predict_model

model = load_model("../../data/model/model")
router = APIRouter()


@router.post('/prediction')
async def predict_price(data: dict = Body(...)):
    data_input = pd.json_normalize(data)
    price = predict_model(estimator=model, data=data_input)
    return {
        "price_estimation": price['Label'][0]
    }

