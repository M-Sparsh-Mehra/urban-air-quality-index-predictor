import os
import joblib
import yaml
import numpy as np
import matplotlib.pyplot as mp
from pyESN import ESN
from src.config import POLLUTANTS, DATA_PROCESSED_PATH, FORECAST_HORIZON

import pandas as pd

def load_best_configs(config_path="/workspaces/urban-air-quality-index-predictor/config/best_esn_configs.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_esn_model(pollutant):
    model_path = f"models/esn_{pollutant}.pkl"
    return joblib.load(model_path)

def predict_ahead(esn, data, horizon=FORECAST_HORIZON):
    inputs = data[:-horizon]
    outputs = []

    last_input = inputs[-1].reshape(1, -1)

    for _ in range(horizon):
        pred = esn.predict(last_input)
        outputs.append(pred.item())
        last_input = pred.reshape(1, -1)

    return outputs

def test_all_esns():
    df = pd.read_csv(DATA_PROCESSED_PATH)
    configs = load_best_configs()

    horizon = FORECAST_HORIZON
    time = df.shape[0]
    
    for pollutant in POLLUTANTS:
        print(f"Predicting for {pollutant}...")

        series = df[pollutant].values.reshape(-1, 1)
        model = load_esn_model(pollutant)

        preds = predict_ahead(model, series, horizon=horizon)
        actual = df[pollutant].values[-horizon:]

        # Plot
        mp.figure(figsize=(8, 4))
        mp.plot(range(len(actual)), actual, label="Actual")
        mp.plot(range(len(preds)), preds, label="Predicted", linestyle="--")
        mp.title(f"{pollutant} Prediction - Next {horizon} steps")
        mp.xlabel("Time Steps Ahead")
        mp.ylabel(pollutant)
        mp.legend()
        mp.grid(True)
        mp.tight_layout()
        mp.show()

if __name__ == "__main__":
    test_all_esns()
