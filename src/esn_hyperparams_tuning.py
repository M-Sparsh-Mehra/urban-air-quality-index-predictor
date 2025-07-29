"""Runs hyperparameter tuning over preprocessed data
Also returns logs best params to "config/best_esn_configs.yaml" 
and best models are saved at "models/esn_<pollutant>.pkl" 
"""

# imports

import os
import yaml
import pandas as pd
import joblib
import mlflow
from itertools import product
from src.config import FORECAST_HORIZON,POLLUTANTS,DATA_RAW_PATH,DATA_PROCESSED_PATH,MODEL_DIR,LOOKBACK
from src.train_esn import train_esn_pollutant
from src.preprocessing import load_raw_data



# hyperparams
N_RESERVOIR_VALUES = [100, 200,50]
SPARSITY_VALUES = [0.1, 0.3,0.5]
SPECTRAL_RADIUS_VALUES = [0.9, 1.0, 0.6]



# grid
param_grid = list(product(N_RESERVOIR_VALUES, SPARSITY_VALUES, SPECTRAL_RADIUS_VALUES))

# function to run hyperparameter tuning
def run_hyperparams():
    mlflow.set_tracking_uri("file:///D:/SPARSH/pollution data/urban-air-quality-index-predictor/mlruns")
    mlflow.set_experiment("ESN_Hyperparam_Sweep")
    df = pd.read_csv(DATA_PROCESSED_PATH)

    # will log all hyperparams here
    best_configs = {}


    for pollutant in POLLUTANTS:
        # initialize
        best_rmse = float("inf")  # we need to minimize RMSE
        best_model = None
        best_params = None
        best_r2 = float("-inf")  # we need to maximize R2
        # start params tuning for each pollutant
        for n_res, spars, rho in param_grid:
            model, rmse, r2 = train_esn_pollutant(
                df=df,
                feature=pollutant,
                forecast_horizon=FORECAST_HORIZON,
                lookback=LOOKBACK,
                n_reservoir=n_res,
                sparsity=spars,
                spectral_radius=rho,
                return_model=True  # train_esn supports this
            )
            if r2 > best_r2:
                best_rmse = rmse
                best_r2 = r2
                # save best model and params
                best_model = model
                best_params = {
                    "n_reservoir": n_res,
                    "sparsity": spars,
                    "spectral_radius": rho,
                    "best_rmse": best_rmse,
                    "best_r2": best_r2
                    }
                
        # saves best model 
        save_path = os.path.join(MODEL_DIR, f"esn_{pollutant.lower().replace('.', '')}.pkl")
        joblib.dump(best_model, save_path)              

         # saves config
        best_configs[pollutant] = best_params
        print(f"Best for {pollutant}: {best_params}, RMSE={best_rmse:.4f}")


    # saves all best configs to YAML
    config_path = ("D:/SPARSH/pollution data/urban-air-quality-index-predictor/config/best_esn_configs.yaml")
    with open(config_path, "w") as f:
        yaml.dump(best_configs, f)    

    print("best configs saved !")





