import os
import numpy as np
import joblib
from pyESN import ESN
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from src.preprocessing import load_raw_data
from src.config import DATA_RAW_PATH, MODEL_DIR,POLLUTANTS
import mlflow
import mlflow.sklearn

# trains on specefic feature in the pollutants list
def train_esn_pollutant(df, feature, forecast_horizon=6, n_reservoir=200, sparsity=0.2, spectral_radius=0.95,activation="tanh"):
        series = df[feature].values   #time series of that specefic feature
        n = len(series) - forecast_horizon
        X = series[:n].reshape(-1, 1) # X will be my first n series values
        Y = series[forecast_horizon:n+forecast_horizon].reshape(-1, 1) # Y will be all next (forecast horizon) values that have to be predicted
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_x.fit_transform(X)
        Y_scaled = scaler_y.fit_transform(Y)
            
        # Train/test split
        trainlen = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:trainlen], X_scaled[trainlen:]
        Y_train, Y_val = Y_scaled[:trainlen], Y_scaled[trainlen:]

        # mlflows 
        with mlflow.start_run(run_name=f"ESN_{feature}"):
                mlflow.set_tag("pollutant", feature)
                mlflow.log_params({
                "n_reservoir": n_reservoir,
                "sparsity": sparsity,
                "spectral_radius": spectral_radius})

        params = { 
                "n_reservoir":200 ,
                "sparsity":0.2 ,
                "spectral_radius": 0.95,
                "leak_rate": 1.0,
                "activation":"tanh"}

        esn=ESN(
            n_inputs=1,
            n_outputs=1,
            n_reservoir=n_reservoir,
            sparsity=sparsity,
            spectral_radius=spectral_radius,
            activation=activation,
            random_state=0)
        
        esn.fit(X_train, Y_train)
        Y_pred = esn.predict(X_val)
        
        rmse = root_mean_squared_error(Y_val, Y_pred, squared=False)
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
        
