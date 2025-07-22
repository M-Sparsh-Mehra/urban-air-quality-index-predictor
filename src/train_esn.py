""" Trains an ESN model for a given pollutant """


import os
import numpy as np
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
def train_esn_pollutant(df, feature, forecast_horizon=FORECAST_HORIZON,lookback=LOOKBACK,n_reservoir=200, sparsity=0.2, spectral_radius=0.95,activation="tanh",return_model=False):
        print("change logged2")
        series = df[feature].values   #time series of that specefic feature
        
        X,Y=[],[] # initialise x and y 
        
        # Prepare lookback input and horizon output
        for i in range(len(series) - lookback - forecast_horizon):
            X.append(series[i:i+lookback])
            Y.append(series[i+lookback:i+lookback+forecast_horizon])
        
        
        X = np.array(X) # shape: (samples, lookback)
        Y = np.array(Y)  # shape: (samples, forecast_horizon)
        
    
        #scaling
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_x.fit_transform(X)
        Y_scaled = scaler_y.fit_transform(Y)
            
        # Train/test split
        trainlen = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:trainlen], X_scaled[trainlen:]
        Y_train, Y_val = Y_scaled[:trainlen], Y_scaled[trainlen:]

        # mlflows 
        # if an active run exists, end it

        #set mlflows directry
        mlflow.set_tracking_uri("file:/workspaces/urban-air-quality-index-predictor/mlruns")
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
            random_state=0)
        
        #fitting ESN
        esn.fit(X_train, Y_train)
        
        # predicting y
        Y_pred = esn.predict(X_val)

        # error metrics
        rmse = root_mean_squared_error(Y_val, Y_pred)
        r2 = r2_score(Y_val, Y_pred)


        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_r2", r2)

        # Save model + scalers
        model_dict = {
            "esn": esn,
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "forecast_horizon": forecast_horizon
        }

        

        save_path = os.path.join(MODEL_DIR, f"esn_{feature.lower().replace('.', '')}.pkl")
        joblib.dump(model_dict, save_path)
        mlflow.log_artifact(save_path)
        
        if return_model:
            return esn, rmse