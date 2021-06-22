import pandas as pd
import tensorflow as tf
import functools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class forecast_LSTM():

	'''
	model: LSTM Model trained
	data:  Ercot Hourly power consumption history (Kwh)
    region: Region of power consumption - [Coast, West, North, East, Far West, South, South Central]

	This class make a weekly power demand forecast with an LSTM Model pre-trained.
	'''
    def __init__(self, model,data,region):
        self.data = data.copy()
        self.model = model
        self.region = region
        
        self.preprocess()
        self.forecast()
        
    def preprocess(self):
        
        # Convert CST & CDT to UTC
        self.data['Hour_Ending']= pd.to_datetime(self.data['Hour_Ending'],utc= True)
        # Datetimeindex
        self.data.set_index("Hour_Ending", inplace=True)
        self.data = self.data.tz_convert(tz='US/Central')
        self.data = self.data.tz_localize(tz=None)
        
    def forecast(self):
        window = 24*7
        train_test = self.data.tail(window)[self.region].values
        scaler = StandardScaler()
        scaler.fit(self.data[self.region].values.reshape(-1,1))
        train_t_scaled = scaler.fit_transform(train_test.reshape(-1,1))
        forcast = []
        for idx in range(0,24*7,24):
            pred = self.model_forecast(self.model, train_t_scaled[idx:window + idx][...,np.newaxis], 1).reshape(-1,1)[:24]
            forcast = np.append(forcast,pred)
            train_t_scaled = np.append(train_t_scaled,pred)
            
        self.forecast = scaler.inverse_transform(forcast)
        
    # transforma os dados para previsões
    def model_forecast(self,model, series, window_size):
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(32).prefetch(1)
        forecast = model.predict(ds)
        return(forecast)