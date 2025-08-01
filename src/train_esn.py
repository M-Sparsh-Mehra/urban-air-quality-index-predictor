""" Trains an ESN model for a given pollutant 
    From each time point, we want to predict the next forecast_horizon steps 
    Not doing recursive predictions. Instead, the model directly outputs all 5 future values at once

"""


# imports

import os
import numpy as np
import matplotlib.pyplot as mp
import mlflow
import joblib
from src.pyESN import ESN
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from src.preprocessing import load_raw_data
from src.config import DATA_RAW_PATH, MODEL_DIR,POLLUTANTS,FORECAST_HORIZON,LOOKBACK
import mlflow.sklearn


# trains on specefic feature in the pollutants list
# this happens for one feature at a time
# df--> preprocessed dataframe
# feature --> pollutant to train on
def train_esn_pollutant(df, feature, forecast_horizon=FORECAST_HORIZON,lookback=LOOKBACK,n_reservoir=200, sparsity=0.2, spectral_radius=0.6,return_model=False):

        idx = POLLUTANTS.index(feature)   #column index / index of feature iterator among POLLUTANTS
        print(f"Training ESNs for {feature} ")

        # data preparation 
        
        series = df[POLLUTANTS].values  #time series of all features
        X,Y=[],[] # initialise x and y

        # Prepare lookback input and horizon output
        # X will be the last lookback point, Y will be the next 'forecast_horizon' points
        for i in range(len(series) - lookback - forecast_horizon):
            X.append(series[i+lookback-1,:].flatten())
            Y.append(series[i+lookback:i+lookback+forecast_horizon,idx])
        
        
        X = np.array(X) # shape: (samples, lookback) 
        Y = np.array(Y)  # shape: (samples, forecast_horizon)
        # each sample is a flattened array of the last 'lookback' points

        # Train/test split
        trainlen = int(len(X) * 0.99)
        X_train, X_test = X[:trainlen], X[trainlen:]
        Y_train, Y_test = Y[:trainlen], Y[trainlen:]


        #set mlflows directry
        mlflow.set_tracking_uri("file:///D:/SPARSH/pollution data/urban-air-quality-index-predictor/mlruns")
        mlflow.set_experiment("ESN_Experiments")

        # Ensure no previous run is active
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=f"ESN_{feature}"):
                mlflow.set_tag("pollutant", feature)
                mlflow.log_params({
                "n_reservoir": n_reservoir,
                "sparsity": sparsity,
                "spectral_radius": spectral_radius})

        esn=ESN(
            n_inputs=X.shape[1],
            n_outputs=FORECAST_HORIZON,
            n_reservoir=n_reservoir,
            sparsity=sparsity,
            spectral_radius=spectral_radius,
            random_state=42)
        
        #fitting ESN
        esn.fit(X_train, Y_train)
        
        # predicting y
        Y_pred = esn.predict(X_test)

        # error metrics for test data
        rmse = root_mean_squared_error(Y_test[0], Y_pred[0])
        r2 = r2_score(Y_test[0], Y_pred[0])

        # error metrics on training data
        Y_pred_train = esn.predict(X_train)
        train_rmse = root_mean_squared_error(Y_train[0], Y_pred_train[0])
        train_r2 = r2_score(Y_train[0], Y_pred_train[0])

        #############################################################################
        # debug test block

        print("x_test",X_test)

        #################################################################################



        #plots
        # Plot test predictions
        mp.figure(figsize=(10, 4))
        mp.plot(Y_test[0].flatten(), label='Actual')
        mp.plot(Y_pred[0].flatten(), label='Predicted')
        mp.title(f"Test - {feature}")
        mp.text(0.01, 0.95, f"RMSE: {rmse:.3f}\nR2: {r2:.3f}", transform=mp.gca().transAxes)
        mp.xlabel("Time")
        mp.ylabel("Value")
        mp.legend()
        mp.tight_layout()
        mp.show()
        mp.savefig(f"test_plot_{feature}.png")
        mlflow.log_artifact(f"test_plot_{feature}.png")
        mp.close()

        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        # Save model
        model_dict = {
            "esn": esn,
            "forecast_horizon": forecast_horizon
        }

        

        save_path = os.path.join(MODEL_DIR, f"esn_{feature.lower().replace('.', '')}.pkl")
        joblib.dump(model_dict, save_path)
        mlflow.log_artifact(save_path)
        
        if return_model:
            return esn, rmse, r2