"""
Tests the ESN models for all pollutants for any new instances.

"""


# imports
import os
import joblib
import yaml
import numpy as np
import matplotlib.pyplot as mp
from src.pyESN import ESN
from src.config import POLLUTANTS, DATA_PROCESSED_PATH, FORECAST_HORIZON,MODEL_DIR,LOOKBACK
import pandas as pd





def load_best_configs(config_path=r"D:\SPARSH\pollution data\urban-air-quality-index-predictor\config\best_esn_configs.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_esn_model(pollutant):
    model_path =  os.path.join(MODEL_DIR, f"esn_{pollutant.lower().replace('.', '')}.pkl")
    return joblib.load(model_path)

def predict_ahead(model_dict, data, horizon=FORECAST_HORIZON):
    
    esn = model_dict["esn"]
    lookback = model_dict.get("lookback", LOOKBACK)

     # Take the last 'lookback' points as input
    last_window = data[-lookback:]  # shape: (-1,)
    current_input = last_window.flatten().reshape(1, -1)  # shape: (1, lookback)

    # Predict the next 'horizon' steps
    preds = []
    preds = esn.predict(current_input)  # shape: (1, forecast_horizon)
    
    return preds

def test_all_esns():
    print("updated")
    df = pd.read_csv(DATA_PROCESSED_PATH)
    configs = load_best_configs()

    horizon = FORECAST_HORIZON
    time = df.shape[0]
    
    for pollutant in POLLUTANTS:
        print(f"Predicting for {pollutant}...")
        
        # feed all pollutants at same time
        series = df[POLLUTANTS].values

        # Load model
        model_dict = load_esn_model(pollutant)

        # Predict next N steps
        preds = predict_ahead(model_dict, series, horizon=FORECAST_HORIZON)

        # Get ground truth for comparison (actual next values)
        actual = series[-FORECAST_HORIZON:, POLLUTANTS.index(pollutant)]

        # Plotting
        mp.figure(figsize=(8, 4))
        mp.plot(range(FORECAST_HORIZON), actual, label="Actual")
        mp.plot(range(FORECAST_HORIZON), preds[0], label="Predicted", linestyle="--")
        mp.title(f"{pollutant} Forecast - Next {FORECAST_HORIZON} Steps")
        mp.xlabel("Time Step Ahead")
        mp.ylabel(pollutant)
        mp.legend()
        mp.grid(True)
        mp.tight_layout()
        mp.show()

if __name__ == "__main__":
    test_all_esns()
