import os
import pandas as pd
from src.config import DATA_RAW_PATH,POLLUTANTS,TARGET_COLUMN,features


def load_raw_data(filename):
    filepath=os.path.join(DATA_RAW_PATH, filename) ## this way we can later call any other raw file as well and need to change in config only
    df=pd.read_csv(filepath)
    #converting date
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    #sort by date
    df = df.sort_values(by=["Date"])

    # cleaning whole df
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # drops rows with no date or AQI
    df.dropna(subset=["Date", "AQI"], inplace=True)

    # interpolate missing pollutant values
    df[features] = df[features].interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(method='ffill')

    return df


