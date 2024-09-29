import streamlit as st
import os
import time
from obspy import read, Trace
from obspy import Stream as stream
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
import joblib

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

        stream = client.get_waveforms(network, station_code, location, channel, starttime, endtime)
    except Exception as e:
      print(f"Error fetching data for {network} {station_code}: {e}")
      stream = None
    return stream

def prediction_output(window, model, look_back, batch_size):
    #reseting the state
    model.reset_states()
    
    # Normalise and  Reshape the initial window
    window = scale_data(window, scaler)
    window = np.reshape(window, (batch_size, look_back, 1))
    
    #Prediction
    predicted_value = model.predict(window, verbose = 0)  
    
    #Returning prediction to original format
    predictions = inverse_scaler(predicted_value, scaler)
    return predictions

def prepare_window_prediction(data, look_back, output, batch_size):
    window_range = look_back*batch_size
    window = data[-window_range:]
    return window

def plot_prediction(trace, predictions):
    fig = plt.figure(figsize=(10, 6))
    
    # Plot the seismogram data
    plt.plot(trace.times("matplotlib"), trace.data, label='Seismogram Data')
    
    # Plot the predictions at the end of the seismogram data
    end_time = trace.times("matplotlib")[-1]
    prediction_times = [end_time + i for i in range(1, len(predictions[0]) + 1)]
    plt.plot(prediction_times, predictions[0], label='Predictions')
    
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Seismogram Plot')
    plt.legend()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
def update_data():
    seconds = ((look_back * batch_size) / 20)
    endtime = UTCDateTime(datetime.utcnow())
    starttime = endtime - timedelta(seconds=seconds)
    
    st = get_stream(network, station_code, location, channel, starttime, endtime)
    trace = st[0]
    data = get_data(st)
    
    window, predictions = prepare_window_prediction(data, look_back, output, batch_size)
    predictions = prediction_output(window, model, look_back, batch_size)
    
    plot_prediction(trace, predictions)
    st.write("Data updated at:", time.strftime("%Y-%m-%d %H:%M:%S"))

# Create a placeholder
placeholder = st.empty()

# Run the update function every 30 seconds
while True:
    with placeholder.container():
        update_data()
    time.sleep(30)

scaler = joblib.load('scaler.save')

model = load_model('model_quick_stateful_i2000_o600_epoch20.hdf5')

look_back = 2000
output = 600
batch_size = 4

# Streamlit app
st.title('Final Project')

seconds = ((look_back*batch_size)/20)
starttime = endtime - delta(seconds=seconds)
endtime = UTCDateTime(datetime.utcnow())

network = "MX"
station_code = "MOIG"
channel = 'BHZ'
location = ""
client = Client("IRIS")

# Create a placeholder
placeholder = st.empty()

# Run the update function every 30 seconds
while True:
    with placeholder.container():
        update_data()
    time.sleep(30)
    
st = get_stream(network, station_code, location, channel, starttime, endtime)
trace = st[0]

data = get_data(st)
window = data[-look_back:]
pred = prediction_output(window, model, look_back)
print(pred)


# Plot the seismogram using Matplotlib
fig = plt.figure(figsize=(10, 6))
plt.plot(trace.times("matplotlib"), trace.data, label='Seismogram Data')
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
