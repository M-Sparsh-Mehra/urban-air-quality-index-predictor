import pandas as pd
import mlflow
from itertools import product
from src.config import FORECAST_HORIZON,POLLUTANTS
from src.train_esn import train_esn_pollutants
from src.preprocessing import load_raw_data

# hyperparams
N_RESERVOIR_VALUES = [100, 200, 300]
SPARSITY_VALUES = [0.1, 0.3, 0.5]
SPECTRAL_RADIUS_VALUES = [0.8, 0.9, 1.0]
ACTIVATION=["tanh","relu"]

# grid
param_grid = list(product(N_RESERVOIR_VALUES, SPARSITY_VALUES, SPECTRAL_RADIUS_VALUES,ACTIVATION))

#    function to run hyperparameter tuning
def run_hyperparams(csv_path):
    mlflow.set_experiment("ESN_Hyperparam_Sweep")
    df = load_raw_data(csv_path)

    for pollutant in POLLUTANTS:
        for n_res, spars, rho,act in param_grid:
            train_esn_pollutants(df, pollutant, FORECAST_HORIZON, n_res, spars, rho,act)




