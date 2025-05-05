import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
import plotly.graph_objects as go

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store, fetch_hourly_rides, fetch_predictions
from src.plot_utils import plot_prediction

st.set_page_config(layout="wide")

# Current timestamp in EST
current_date = pd.Timestamp.now(tz="US/Eastern")
st.title("Citi Bike Demand for the Next Hour")
st.header(f"{current_date.strftime('%Y-%m-%d %H:%M:%S')} (EST)")

# Sidebar progress
progress_bar = st.sidebar.progress(0)
N_STEPS = 5

# Step 1: Fetch Features
with st.spinner(text="Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write("Inference features fetched from the store")
    progress_bar.progress(1 / N_STEPS)

# Step 2: Fetch Predictions
with st.spinner("Fetching Predictions"):
    predictions = fetch_next_hour_predictions()
    st.sidebar.write("Predictions fetched")
    progress_bar.progress(2 / N_STEPS)

# Step 3: Add Station Metadata

station_dict = {
    int(5905.0): {"name": "Broadway & E 14 St",
           "longitude": -73.99074142,
           "latitude": 40.73454567
           },
    int(6140.0): {"name": "W 21 St & 6 Ave",
           "longitude": -73.99415556,
           "latitude": 40.74173969},
    int(6822.0): {"name": "1 Ave & E 68 St",
           "longitude": -73.958115339,
           "latitude": 40.765112281}
}

with st.spinner("Merging Predictions with Station Information"):
    station_df = pd.DataFrame.from_dict(station_dict, orient='index').reset_index()
    station_df.rename(columns={'index': 'start_station_id'}, inplace=True)
    predictions["start_station_id"] = predictions["start_station_id"].round(0).astype(int)
    predictions = pd.merge(predictions, station_df, on='start_station_id', how='left')
    features["start_station_id"] = features["start_station_id"].round(0).astype(int)
    features = pd.merge(features, station_df, on='start_station_id', how='left')
    st.sidebar.write("Information merged")
    progress_bar.progress(3 / N_STEPS)

# Map visualization
with st.spinner("Making Map of NYC"):
    st.subheader("Predicted Demand Map")
    map_center = [40.76, -73.98]
    map_obj = folium.Map(location=map_center, zoom_start=12)

    for _, row in predictions.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            popup=f"{row['name']}: {row['predicted_demand']:.0f} rides",
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
        ).add_to(map_obj)

    st_folium(map_obj, width=800, height=600)
    st.sidebar.write("Map Plotted")
    progress_bar.progress(4 / N_STEPS)

# Data table
st.subheader("Predictions")
st.dataframe(
    predictions[["start_station_id", "name", "start_hour", "predicted_demand"]]
    .sort_values("predicted_demand", ascending=False)
    .reset_index(drop=True)
)

rides = fetch_hourly_rides(672)
preds = fetch_predictions(672)

preds["start_station_id"] = preds["start_station_id"].round(0).astype(int)
preds = pd.merge(preds, station_df, on='start_station_id', how='left')
rides["start_station_id"] = rides["start_station_id"].round(0).astype(int)
rides = pd.merge(rides, station_df, on='start_station_id', how='left')

# Station selector
st.subheader("Station-Level Trend")
station_options = preds["name"].dropna().unique()
selected_station = st.selectbox("Select a station to view predictions and actual rides", station_options)

# Filter data for the selected station
preds_station_data = preds[preds["name"] == selected_station].sort_values("start_hour")
rides_station_data = rides[rides["name"] == selected_station].sort_values("start_hour")


fig = go.Figure()
# Base plot
fig.add_trace(go.Scatter(
    x=rides_station_data["start_hour"],
    y=rides_station_data["rides"],
    mode="lines+markers",
    name="Actual Rides"
))

fig.add_trace(go.Scatter(
    x=preds_station_data["start_hour"],
    y=preds_station_data["predicted_demand"],
    mode="lines+markers",
    name="Predicted Demand",
    line=dict(dash='dash', color="orange")
))

fig.update_layout(
    title=f"Ride Counts and Predicted Demand for Station @ {selected_station}",
    xaxis_title="Time",
    yaxis_title="Rides",
    template="plotly_white"
)

st.plotly_chart(fig, theme="streamlit", use_container_width=True)