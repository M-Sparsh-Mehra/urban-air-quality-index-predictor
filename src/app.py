import time
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from config import MODEL_DIR,POLLUTANTS, FORECAST_HORIZON
import os
import sys


# adds project root to sys.path so 'src' can be imported during pickle load
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# load models

@st.cache_resource
def load_models():

    pm25_model = joblib.load(os.path.join(MODEL_DIR, "esn_pm25.pkl"))
    pm10_model = joblib.load(os.path.join(MODEL_DIR, "esn_pm10.pkl"))
    no2_model = joblib.load(os.path.join(MODEL_DIR, "esn_no2.pkl"))
    aqi_model = joblib.load(os.path.join(MODEL_DIR, "aqi_regressor.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "pollutant_scaler.pkl"))
    return pm25_model, pm10_model, no2_model, aqi_model, scaler

pm25_model, pm10_model, no2_model, aqi_model, scaler = load_models()


# forecast function 

def forecast_pollutant(model,pollutant_name,current_pollutants):
    """
    model: trained ESN for a single pollutant
    current_pollutants: list or array [PM2.5_now, PM10_now, NO2_now]
    """
    # scale input
    scaled_input = scaler.transform([current_pollutants])
    prediction_scaled = model.predict(scaled_input)[0]  # (forecast_horizon,)
    # inverse transform only this pollutant's predictions
    idx = POLLUTANTS.index(pollutant_name)
    dummy = np.zeros((len(prediction_scaled), len(POLLUTANTS)))
    dummy[:, idx] = prediction_scaled
    pred_unscaled = scaler.inverse_transform(dummy)[:, idx]

    return pred_unscaled,prediction_scaled 
# return both bcause we will need scaled for xgboost prediction and unscaled for display


# Streamlit UI

st.set_page_config(page_title="Air-Quality-Index Predictor", layout="wide")
st.title(" Air Quality Index (AQI) Predictor")
st.markdown("""
    This application predicts the Air Quality Index (AQI) based on current pollutant levels.
    Enter the current values for PM2.5, PM10, and NO₂ to get a 10-day forecast.
""")


# sidebar Inputs
st.sidebar.info("Forecast AQI 10 days ahead using Machine Learning")
pm25_now = st.sidebar.number_input("Current PM2.5", min_value=0.0, value=50.0)
pm10_now = st.sidebar.number_input("Current PM10", min_value=0.0, value=80.0)
no2_now = st.sidebar.number_input("Current NO₂", min_value=0.0, value=40.0)

if st.sidebar.button("Run Forecast"):
    # forecast pollutants
    current_values = [pm25_now, pm10_now, no2_now]
    pm25_forecast_unscaled,pm25_forecast_scaled = forecast_pollutant(pm25_model, "PM2.5", current_values)
    pm10_forecast_unscaled,pm10_forecast_scaled = forecast_pollutant(pm10_model, "PM10", current_values)
    no2_forecast_unscaled,no2_forecast_scaled = forecast_pollutant(no2_model, "NO2", current_values)

    with st.spinner('Predicting AQI...'):
        time.sleep(3)

    # prepare dataframe
    forecast_df_scaled = pd.DataFrame({
        "Step": range(1, FORECAST_HORIZON + 1),
        "PM2.5": pm25_forecast_scaled,
        "PM10": pm10_forecast_scaled,
        "NO2": no2_forecast_scaled
    })

    forecast_df_unscaled = pd.DataFrame({
        "Step": range(1, FORECAST_HORIZON + 1),
        "PM2.5": pm25_forecast_unscaled,
        "PM10": pm10_forecast_unscaled,
        "NO2": no2_forecast_unscaled
    })

    # predict AQI for each step
    X_aqi = forecast_df_scaled[["PM2.5", "PM10", "NO2"]].values
    aqi_forecast = aqi_model.predict(X_aqi)
    forecast_df_scaled["Predicted AQI"] = aqi_forecast

    # display charts
    col1, col2 = st.columns(2)
    with col1:
        fig_pollutants = px.line(forecast_df_unscaled, x="Step", y=["PM2.5", "PM10", "NO2"], title="Pollutant Forecast")
        st.plotly_chart(fig_pollutants, use_container_width=True)
    with col2:
        fig_aqi = px.line(forecast_df_scaled, x="Step", y="Predicted AQI", title="AQI Forecast")
        st.plotly_chart(fig_aqi, use_container_width=True)

    # show table
    st.dataframe(forecast_df_unscaled)

    # download button
    csv = forecast_df_unscaled.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", csv, "aqi_forecast.csv", "text/csv")
