"""
This module contains the code for hyperparameter tuning of XGBoost models.
saves best params to "config/best_xgb_config.yaml"
No need to save model, as we only use XGBoost once in final training.
"""

#imports
import xgboost as xgb
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from itertools import product
import joblib
from src.config import DATA_PROCESSED_PATH, MODEL_DIR, POLLUTANTS, FORECAST_HORIZON, LOOKBACK
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from src.AQI_pred import train_xg_aqi



# hyperparameter ranges
N_ESTIMATORS = [100, 200]
MAX_DEPTH = [3, 4, 7]
LEARNING_RATE = [0.1, 0.2]
SUBSAMPLE = [0.5, 0.7, 1]
"""
   'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 500],
    'subsample': [0.5, 0.7, 1],
"""


# generates grid of (n_estimators, max_depth, learning_rate)
param_grid = list(product(N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE))


# runs hyperparameter tuning for XGBoost
# saves best params to "config/best_xgb_config.yaml"
def run_xgb_hyperparams():
    df = pd.read_csv(DATA_PROCESSED_PATH)

    # initialize best metrics
    best_rmse = float("inf")
    best_r2 = float("-inf")
    best_params = None

    for n_est, depth, lr,subsample in param_grid:
        # train the model 
        model,metrics = train_xg_aqi(
                df=df,
                forecast_horizon=FORECAST_HORIZON,
                lookback=LOOKBACK,
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                subsample=subsample,
                return_model=True 
            )
        # get metrics       
        
        forecast_rmse = metrics["forecast_rmse"]
        forecast_r2 = metrics["forecast_r2"]
        if forecast_r2 > best_r2:
            best_rmse = forecast_rmse
            best_r2 = forecast_r2
            best_params = {
                    "n_estimators": n_est,
                    "max_depth": depth,
                    "learning_rate": lr,
                    "best_rmse": best_rmse,
                    "best_r2": best_r2
            }


    # save best config
    config_path =("D:/SPARSH/pollution data/urban-air-quality-index-predictor/config/best_xg_configs.yaml")
    with open(config_path, "w") as f:
        yaml.dump(best_params, f)

    print("best_params:", best_params)

