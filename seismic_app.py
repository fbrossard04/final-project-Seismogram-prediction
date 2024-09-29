import streamlit
from pages import page1
from pages import About

streamlit.set_page_config(page_title="Multipage App")

streamlit.sidebar.title("Navigation")
streamlit.sidebar.success("Select a page above.")


import os
import time
from obspy import read, Trace
from obspy import Stream as stream
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from keras.models import load_model
import joblib
from time import sleep


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
    
def prepare_window(data, look_back, output, batch_size):
    window_range = look_back*batch_size
    window = data[:window_range]
    actual = data[window_range:window_range+output]
    return window, actual
    
def prepare_window_prediction(data, look_back, output, batch_size):
    window_range = look_back*batch_size
    window = data[-window_range:]
    return window

def plot_prediction(trace, predictions):
    fig = plt.figure(figsize=(10, 6))
    
    # Calculate time points for predictions (assuming equal intervals)
    time_steps = len(trace.data)
    prediction_times = range(time_steps)  # Adjust this based on your prediction logic

    # Plot the seismogram data
    plt.plot(range(time_steps), trace.data, label='Seismogram Data')

    # Plot the predictions
    plt.plot(range(time_steps,time_steps+len(predictions[0])), predictions[0], label='Predictions')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Seismogram Plot')
    plt.legend()
    
    # Display the plot in Streamlit
    streamlit.pyplot(fig)

    
def update_data():
    look_back = 1200
    batch_size = 4
    seconds = ((look_back * batch_size) / 10)
    endtime = UTCDateTime(datetime.utcnow())
    starttime = endtime - timedelta(seconds=seconds)
    network = "MX"
    station_code = "MOIG"
    channel = 'BHZ'
    location = ""
    client = Client("IRIS")
    st = get_stream(network, station_code, location, channel, starttime, endtime)
    trace = st[0]
    data = get_data(st)
    
    window = prepare_window_prediction(data, look_back, output, batch_size)
    predictions = prediction_output(window, model, look_back, batch_size)
    
    plot_prediction(trace, predictions)

def update_data1():
    look_back = 1200
    batch_size = 4
    seconds = ((look_back * batch_size) / 10)
    endtime = UTCDateTime(datetime.utcnow())
    starttime = endtime - timedelta(seconds=seconds)
    network = "MX"
    station_code = "MOIG"
    channel = 'BHZ'
    location = ""
    client = Client("IRIS")
    st = get_stream(network, station_code, location, channel, starttime, endtime)
    trace = st[0]
    data = get_data(st)
    window, actual = prepare_window(data, look_back, output, batch_size)
    predictions = prediction_output(window, model, look_back, batch_size)
    plot_seismogram(window, actual, predictions, look_back)
    

def plot_seismogram(window, actual, predictions, look_back):
    fig = plt.figure(figsize=(10, 6))

    # Plot the window
    window_plot = window[-look_back:]
    plt.plot(window_plot, label='window')

    # Plot the actual data at the end of the window
    plt.plot(range(len(window_plot), len(window_plot) + len(actual)), actual, label='actual')

    # Plot the predictions at the end of the window
    plt.plot(range(len(window_plot), len(window_plot) + len(predictions[0])), predictions[0], label='predictions')

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Seismogram Plot')
    plt.legend()
    streamlit.pyplot(fig)


scaler = joblib.load('scaler.save')

model = load_model('my_model_checkpoints/model_stateful_i1200_o600_batch.hdf5')

look_back = 1200
output = 600
batch_size = 4

# Streamlit app
streamlit.title('Final Project')

network = "MX"
station_code = "MOIG"
channel = 'BHZ'
location = ""
client = Client("IRIS")

# Create a placeholder
placeholder = streamlit.empty()

# Run the update function every 30 seconds
while True:
    with placeholder.container():
        update_data()
    time.sleep(30)
    streamlit.experimental_rerun()



# Credit Section
streamlit.header('Credits')
streamlit.write('This project was developed using obspy')
streamlit.write('Project developped Francois Brossard')
