import os
import joblib
import yaml
import numpy as np
import matplotlib.pyplot as mp
from src.pyESN import ESN
from src.config import POLLUTANTS, DATA_PROCESSED_PATH, FORECAST_HORIZON,MODEL_DIR,LOOKBACK

import pandas as pd

def load_best_configs(config_path="/workspaces/urban-air-quality-index-predictor/config/best_esn_configs.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_esn_model(pollutant):
    model_path =  os.path.join(MODEL_DIR, f"esn_{pollutant.lower().replace('.', '')}.pkl")
    return joblib.load(model_path)

def predict_ahead(model_dict, data, horizon=FORECAST_HORIZON):
    """Perform autoregressive forecasting for a pollutant."""
    esn = model_dict["esn"]
    scaler_x = model_dict["scaler_x"]
    scaler_y = model_dict["scaler_y"]
    lookback = model_dict.get("lookback", LOOKBACK)

     # Take the last 'lookback' points as input
    last_window = data[-lookback:]  # shape: (lookback,)
    current_input = last_window.reshape(1, -1)  # shape: (1, lookback)
    current_input = scaler_x.transform(current_input)  # transform only

    # Predict in one go (since ESN was trained that way)
    pred_scaled = esn.predict(current_input)  # shape: (1, forecast_horizon)
    preds = scaler_y.inverse_transform(pred_scaled).flatten()  # inverse transform output

    return preds

def test_all_esns():
    print("updated")
    df = pd.read_csv(DATA_PROCESSED_PATH)
    configs = load_best_configs()

    horizon = FORECAST_HORIZON
    time = df.shape[0]
    
    for pollutant in POLLUTANTS:
        print(f"Predicting for {pollutant}...")

        series = df[pollutant].values

        # Load model
        model_dict = load_esn_model(pollutant)

        # Predict next N steps
        preds = predict_ahead(model_dict, series, horizon=FORECAST_HORIZON)

        # Get ground truth for comparison (actual next values)
        actual = series[-FORECAST_HORIZON:]
  
       # Plotting
        mp.figure(figsize=(8, 4))
        mp.plot(range(FORECAST_HORIZON), actual, label="Actual")
        mp.plot(range(FORECAST_HORIZON), preds, label="Predicted", linestyle="--")
        mp.title(f"{pollutant} Forecast - Next {FORECAST_HORIZON} Steps")
        mp.xlabel("Time Step Ahead")
        mp.ylabel(pollutant)
        mp.legend()
        mp.grid(True)
        mp.tight_layout()
        mp.show()

if __name__ == "__main__":
    test_all_esns()
