"""
    Trains an XGBoost model to predict air quality index (AQI) based on pollutant data.
    We used ESN to predict pollutants for the first test point's forecast horizon i.e. the next FORECAST_HORIZON predicted pollutant vectors
    We pass those predicted pollutant vectors to the trained XGBoost model and predict next FORECAST_HORIZON AQI values accordingly
"""

# imports

import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import joblib
from src.config import DATA_RAW_PATH, MODEL_DIR,POLLUTANTS,FORECAST_HORIZON,LOOKBACK
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score





def train_xg_aqi(df, forecast_horizon=FORECAST_HORIZON,lookback=LOOKBACK,n_estimators=200, max_depth=4, learning_rate=0.1, return_model=False):
    
    # data preparation
    X = df[POLLUTANTS].values
    Y = df["AQI"].values



    # Prepare lookback input and horizon output
    def make_lookback_data(series, lookback):
        X, Y = [], []
        for i in range(len(series) - lookback - forecast_horizon):
            X.append(series[i+lookback-1,:-1])# all pollutants except AQI
            Y.append(series[i+lookback-1,-1]) # AQI is the last column
        return np.array(X), np.array(Y)

    series = df[POLLUTANTS + ["AQI"]].values
    X, Y = make_lookback_data(series, lookback=LOOKBACK)
    # Train/test split
    # Use entire available training data (same cutoff as ESN)
    trainlen = int(len(X) * 0.99)
    X_train, y_train = X[:trainlen], Y[:trainlen]
    X_test, y_test = X[trainlen:], Y[trainlen:]

    # Recalculate the real forecast horizon available
    actual_forecast_horizon = len(df) - trainlen

    # Use the smaller of the requested horizon or the available data
    forecast_horizon = min(FORECAST_HORIZON, actual_forecast_horizon)


    # Initialize and train the XGBoost model
    # Train regressor
    model = xgb.XGBRegressor(n_estimators=n_estimators, 
                             max_depth=max_depth, 
                             learning_rate=learning_rate,
                            random_state=42)
    model.fit(X_train, y_train)



    # call the model trained for pollutants we will make a matrix of the next FORECAST_HORIZON predicted pollutants
    # using that we can predict aqi for those next instances
    predicted_pollutants = []

    #  last input (just before test) is used as starting point
    #last_lookback_input = df[POLLUTANTS].values[trainlen + LOOKBACK - 1, :].reshape(1, -1)

    for pollutant in POLLUTANTS:
        model_path = f"{MODEL_DIR}/esn_{pollutant.lower().replace('.', '')}.pkl"
        esn_dict = joblib.load(model_path)
        esn = esn_dict["esn"]
        
        # Predicts horizon
        pred = esn.predict(X_test)
        predicted_pollutants.append(pred[0])
        # this is how we did during training

    # Stack predictions 
    predicted_pollutants = np.stack(predicted_pollutants, axis=1)  # shape: (forecast_horizon,num_pollutants)

    # evaluate the model
    y_train_pred = model.predict(X_train)
    # 1. Real pollutants → XGBoost
    X_test=X_test[1:] #remove first row because its not included in the predicted pollutants (because predicted poll needs first row to predict forward ahead isntances)
    y_test= y_test[1:] #remove first row because its not included in the predicted pollutants (because predicted poll needs first row to predict forward ahead isntances)
    xg_test_pred = model.predict(X_test)
    
    # 2. Predicted pollutants → XGBoost
    y_forecast_pred = model.predict(predicted_pollutants[:lookback])  # predict using the predicted pollutants
    
    # Evaluate metrics

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
        
    xg_test_rmse = root_mean_squared_error(y_test, xg_test_pred)
    xg_test_r2 = r2_score(y_test, xg_test_pred)

    forecast_rmse = root_mean_squared_error(y_test, y_forecast_pred)
    forecast_r2 = r2_score(y_test, y_forecast_pred)

    print(f"[XGBoost AQI Model]")
    print(f"Train RMSE: {train_rmse:.3f}, R2: {train_r2:.3f}")
    print(f"Test RMSE: {xg_test_rmse:.3f}, R2: {xg_test_r2:.3f}")
    print(f"Forecast RMSE: {forecast_rmse:.3f}, R2: {forecast_r2:.3f}")

    # plots predictions 
    mp.figure(figsize=(10, 4))
    mp.plot(y_test, label="Actual AQI", marker='o')
    mp.plot(xg_test_pred, label="Predicted AQI", marker='x')
    mp.plot(y_forecast_pred, label="Forecasted AQI", linestyle='--')
    mp.title("XGBoost AQI Prediction")
    mp.xlabel("Time")
    mp.ylabel("AQI")
    mp.legend()
    mp.tight_layout()
    mp.show()


        