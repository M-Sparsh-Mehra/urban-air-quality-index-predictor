import os

# general paths
DATA_RAW_PATH = r"D:\SPARSH\pollution data\urban-air-quality-index-predictor\data\raw"
DATA_PROCESSED_PATH = r"D:\SPARSH\pollution data\urban-air-quality-index-predictor\data\processed\processed_data.csv"
PREDICTIONS_PATH = r"D:\SPARSH\pollution data\urban-air-quality-index-predictor\data\predictions"
MODEL_DIR = r"D:\SPARSH\pollution data\urban-air-quality-index-predictor\models"

features=["StationId", "Date", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI", "AQI_Bucket"]

#These are the standard criteria pollutants used by agencies
POLLUTANTS = ["PM2.5", "PM10", "NO2"] #, "NO", "CO", "SO2", "O3", "NH3"] #AFTER EDA, WE ONLY USE PM2.5, PM10, NO2


#station
STATION= 'DL001'  # default station for testing
# list of all stations in the dataset
STATIONS = ['AP001', 'AP005', 'AS001', 'BR005', 'BR006', 'BR007', 'BR008',
       'BR009', 'BR010', 'CH001', 'DL001', 'DL002', 'DL003', 'DL004',
       'DL005', 'DL006', 'DL007', 'DL008', 'DL009', 'DL010', 'DL011',
       'DL012', 'DL013', 'DL014', 'DL015', 'DL016', 'DL017', 'DL018',
       'DL019', 'DL020', 'DL021', 'DL022', 'DL023', 'DL024', 'DL025',
       'DL026', 'DL027', 'DL028', 'DL029', 'DL030', 'DL031', 'DL032',
       'DL033', 'DL034', 'DL035', 'DL036', 'DL037', 'DL038', 'GJ001',
       'HR011', 'HR012', 'HR013', 'HR014', 'JH001', 'KA002', 'KA003',
       'KA004', 'KA005', 'KA006', 'KA007', 'KA008', 'KA009', 'KA010',
       'KA011', 'KL002', 'KL004', 'KL007', 'KL008', 'MH005', 'MH006',
       'MH007', 'MH008', 'MH009', 'MH010', 'MH011', 'MH012', 'MH013',
       'MH014', 'ML001', 'MP001', 'MZ001', 'OD001', 'OD002', 'PB001',
       'RJ004', 'RJ005', 'RJ006', 'TG001', 'TG002', 'TG003', 'TG004',
       'TG005', 'TG006', 'TN001', 'TN002', 'TN003', 'TN004', 'TN005',
       'UP012', 'UP013', 'UP014', 'UP015', 'UP016', 'WB007', 'WB008',
       'WB009', 'WB010', 'WB011', 'WB012', 'WB013']


#target 
TARGET_COLUMN = "AQI"



FORECAST_HORIZON = 10  # steps ahead to predict ; i want this to be tunable
LOOKBACK=5 # history window
