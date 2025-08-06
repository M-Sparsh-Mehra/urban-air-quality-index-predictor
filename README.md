# urban-air-quality-index-predictor

This project presents a robust end to end machine learning pipeline for **multi-step Air Quality Index (AQI) forecasting** using:


- 📈 Pollutant Forecasting using Echo State Networks (ESNs) for dynamic, multi-step ahead predictions of critical pollutants (PM2.5, PM10, NO2).

- 📊 AQI Estimation via XGBoost Regressor, on forecasted pollutant levels to accurately predict future AQI values.

- ⚙️ Hyperparameter Optimization for ESNs and XGBoost to maximize predictive performance across pollutants.

- 🔁 Configurable Pipeline, allowing flexible tuning of lookback windows and forecast horizons.

- 🧪 Experiment Tracking using MLflow, ensuring reproducibility and comprehensive model evaluation.


# Pollutant Forecasting with ESN
Each selected pollutant was modeled using a *dedicated Echo State Network*, and was tuned via hyperparameter search.
<p align="center"> 
  <img src="Results/plots_esn/test_plot_PM2.5.png" width="400"> 
  <img src="Results/plots_esn/test_plot_PM10.png" width="400"> 
  <img src="Results/plots_esn/test_plot_NO2.png" width="400"> 
</p>

## 🎯 AQI Regression via XGBoost
Using the forecasted pollutants as features , I trained an XGBoost regressor to predict the corresponding AQIs.

<p align="center"> 
  <img src="Results/plots_xg/plot_xgbosst_aqi_pred.png" width="400"> 
</p>

## 📈 Performance Metrics

### Pollutants forecasting
| Pollutant          | Best R2  | Best RMSE| 
|--------------------|----------|----------|
| PM 2.5             | 0.995    |  0.0005  |
| PM 10              | 0.997    |  0.0020  |
| NO2                | 0.968    |  0.0058  |


### Corresponding AQI prediction

| Metric  |Best value |
|---------|-----------|
| R2      | 0.951     |
| RMSE    | 4.63      |

## 🔮 Future Work

I am aiming to build a complete air quality forecasting system. Possible extensions include:

- **📊 Dashboard Integration:**  
  Develop a real-time interactive dashboard using **Dash**, or **Streamlit**, to visualize pollutant forecasts, AQI predictions, and trends across different horizons.

- **🌐 Web Interface & API Deployment:**  
  Wrap the entire pipeline into a REST API (using **FastAPI** or **Flask**) for easy integration with front-end web or mobile applications.