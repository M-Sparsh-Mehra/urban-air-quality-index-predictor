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

# Define models directory based on project root
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
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
# centers everything
st.markdown("""
    <style>
    /* Center all main elements */
    .block-container {
        text-align: center;
    }
    
    /* Center markdown text */
    .stMarkdown {
        text-align: center;
    }

    /* Center dataframes / tables */
    table {
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Center images */
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Air-Quality-Index Predictor", layout="wide")
st.title(" Air Quality Index (AQI) Predictor")
st.markdown("""
    This application predicts the Air Quality Index (AQI) based on current pollutant levels.
    Enter the current values for PM2.5, PM10, and NO₂ to get a 10-day forecast.
""")



# Input fields on main page
with st.container():
    st.subheader("Enter Current Pollutant Levels")
    col1, col2, col3 = st.columns(3)
    with col1:
        pm25_now = st.number_input("Current PM2.5", min_value=0.0, value=55.21)
    with col2:
        pm10_now = st.number_input("Current PM10", min_value=0.0, value=128.02)
    with col3:
        no2_now = st.number_input("Current NO₂", min_value=0.0, value=17.64)

if st.button("Run Forecast", use_container_width=True):


    # forecast pollutants
    current_values = [pm25_now, pm10_now, no2_now]
    pm25_forecast_unscaled,pm25_forecast_scaled = forecast_pollutant(pm25_model, "PM2.5", current_values)
    pm10_forecast_unscaled,pm10_forecast_scaled = forecast_pollutant(pm10_model, "PM10", current_values)
    no2_forecast_unscaled,no2_forecast_scaled = forecast_pollutant(no2_model, "NO2", current_values)

    with st.spinner('Predicting AQI...'):
        time.sleep(3)

    # prepare dataframe
    forecast_df_scaled = pd.DataFrame({
        "Days ahead": range(1, FORECAST_HORIZON + 1),
        "PM2.5": pm25_forecast_scaled,
        "PM10": pm10_forecast_scaled,
        "NO2": no2_forecast_scaled
    })

    forecast_df_unscaled = pd.DataFrame({
        "Days ahead": range(1, FORECAST_HORIZON + 1),
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
        fig_pollutants = px.line(forecast_df_unscaled, x="Days ahead", y=["PM2.5", "PM10", "NO2"], title="Pollutant Forecast")
        st.plotly_chart(fig_pollutants, use_container_width=True)
    with col2:
        fig_aqi = px.line(forecast_df_scaled, x="Days ahead", y="Predicted AQI", title="AQI Forecast")
        st.plotly_chart(fig_aqi, use_container_width=True)

    forecast_df_unscaled["Predicted AQI"] = aqi_forecast

    # show table
    records = forecast_df_unscaled.to_dict(orient="records")
    with st.expander("Forecast Data", expanded=True):
        st.dataframe(records, use_container_width=True)

    # download button
    csv = forecast_df_unscaled.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", csv, "aqi_forecast.csv", "text/csv")
    
    st.markdown(
    """
    ---
    **Made with ❤️ by [Sparsh](https://github.com/M-Sparsh-Mehra/urban-air-quality-index-predictor)**
    """
     )
