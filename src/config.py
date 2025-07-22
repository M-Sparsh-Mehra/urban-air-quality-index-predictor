import os

# general paths
DATA_RAW_PATH = "/workspaces/urban-air-quality-index-predictor/data/raw/"
DATA_PROCESSED_PATH = "/workspaces/urban-air-quality-index-predictor/data/processed/processed_data.csv"
PREDICTIONS_PATH = "/workspaces/urban-air-quality-index-predictor/data/predictions/"
MODEL_DIR = "/workspaces/urban-air-quality-index-predictor/models/"

features=["StationId", "Date", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI", "AQI_Bucket"]

#These are the standard criteria pollutants used by agencies
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "CO", "SO2", "O3", "NH3"]
#target 
TARGET_COLUMN = "AQI"



FORECAST_HORIZON = 6  # steps ahead to predict ; i want this to be tunable
LOOKBACK=12 # history window
