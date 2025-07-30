import os
import pandas as pd
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from src.config import DATA_RAW_PATH,POLLUTANTS,TARGET_COLUMN,features,DATA_PROCESSED_PATH,STATION


# just loads the data as it is and returns as a dataframe
#takes csv file name as input 
def load_raw_data(filename):
    filepath = os.path.join(DATA_RAW_PATH, filename)
    df = pd.read_csv(filepath)
    return df

# does the preprocessing
# input is a df
#op is a preprocessed df 
def process_data(df):

    df = df[df["StationId"] == STATION] # filter by station

    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.sort_values(by=["Date"])

    # drops unecesarry columns
    for col in features:
        if col in set(POLLUTANTS + ["Date", TARGET_COLUMN]):
            continue
        else:
            df=df.drop([col],axis=1)

    #converts all data inside df into numericals
    for col in POLLUTANTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    
    #drop NaN in date or target
    df.dropna(subset=["Date", "AQI"], inplace=True)

    df[POLLUTANTS] = df[POLLUTANTS].interpolate(method='linear', limit_direction='both').bfill() .ffill()

    # Apply MinMax scaling
    scaler = MinMaxScaler() 
    df[POLLUTANTS] = scaler.fit_transform(df[POLLUTANTS])

    # an unrealistic outlier with AQI>2000 was observed during EDA
    df = df[df[TARGET_COLUMN] <= 500]


    return df

   
# saves preprocessed data to mentioned file path in config.py....the file is in data/preocessed
def save_processed_data(df, path=DATA_PROCESSED_PATH):
    #saves processed dataframe to CSV
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)







# runs the complete pipeline
# more often we'll just need to call this
#returns processed dataframe
def run_preprocessing(filename): 
    raw_df = load_raw_data(filename)
    processed_df = process_data(raw_df) 
    save_processed_data(processed_df)  # save to file
    return processed_df  # return for use in notebook 




# Entry point for CLI or notebook
if __name__ == "__main__":
    run_preprocessing()
