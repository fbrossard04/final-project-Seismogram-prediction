import streamlit
import os
import time
from obspy import read, Stream, Trace
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from datetime import timedelta

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from keras.models import load_model

# Load your trained model
#model = load_model('fine_tuned_model.h5')

def get_data(st):
    """Take a stream, fill gap with interpolate  and return  data"""
    st =  st.merge(method=1, fill_value='interpolate')
    tr = st[0]
    data = tr.data.astype(np.float32).reshape(-1, 1)
    return data

def fit_scaler(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = data.reshape(-1, 1)
    scaler.fit(data)
    return scaler

def scale_data(data, scaler):
    data = data.reshape(-1, 1)
    scaled_data = scaler.transform(data)
    return scaled_data

def inverse_scaler(scaled_data, scaler):
    restored_data = scaler.inverse_transform(scaled_data)
    return restored_data

def get_stream(network, station_code, location, channel, starttime, endtime):
    try:
        client = Client("IRIS")

        st = client.get_waveforms(network, station_code, location, channel, starttime, endtime)
    except Exception as e:
      print(f"Error fetching data for {network} {station_code}: {e}")
      st = None
    return st

# Streamlit app
streamlit.title('Final Project')

#seconds = (look_back/20)
starttime = UTCDateTime(2023, 9, 10)
endtime = starttime + timedelta(seconds=350)

network = "MX"
station_code = "MOIG"
channel = 'BHZ'
location = ""
client = Client("IRIS")

st = get_stream(network, station_code, location, channel, starttime, endtime)
trace = st[0]

# Plot the seismogram using Matplotlib
fig = plt.figure(figsize=(10, 6))
plt.plot(trace.times("timestamp"), trace.data, label='Seismogram Data')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Seismogram Plot')
plt.legend()
streamlit.pyplot(fig)

# Credit Section
streamlit.header('Credits')
streamlit.write('This project was developed using obspy')
#streamlit.write('https://www.kaggle.com/datasets/gpiosenka/100-bird-species')
streamlit.write('Project developped Francois Brossard')
