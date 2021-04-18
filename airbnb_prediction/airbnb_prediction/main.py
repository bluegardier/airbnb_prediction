import pandas as pd
from pandas import json_normalize
from pycaret.regression import load_model, predict_model
from flask import Flask, request, jsonify
f

model = load_model("../../data/model/model")

app = Flask(__name__)


@app.route('/')
def home():
    return "Standing By"


@app.route('/prediction', methods=['POST'])
def predict_price():
    data = request.get_json()
    data_input = pd.json_normalize(data)
    price = predict_model(estimator=model, data=data_input)
    return jsonify(price['Label'][0])


if __name__ == '__main__':
    app.run(port=5000, debug=True)
