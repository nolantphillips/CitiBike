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

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

st.set_page_config(layout="wide")

# Current timestamp in EST
current_date = pd.Timestamp.now(tz="US/Eastern")
st.title("Citi Bike Demand for the Next Hour")
st.header(f"{current_date.strftime('%Y-%m-%d %H:%M:%S')} (EST)")

# Sidebar progress
progress_bar = st.sidebar.progress(0)
N_STEPS = 2

# Step 1: Fetch Predictions
with st.spinner("Fetching Predictions"):
    predictions = fetch_next_hour_predictions()
    st.sidebar.write("Predictions fetched")
    progress_bar.progress(1 / N_STEPS)

# Step 2: Add Station Metadata

station_dict = {
    np.float32(5905.140137): {"name": "Broadway & E 14 St",
           "longitude": -73.99074142,
           "latitude": 40.73454567
           },
    np.float32(6140.049805): {"name": "W 21 St & 6 Ave",
           "longitude": -73.99415556,
           "latitude": 40.74173969},
    np.float32(6822.089844): {"name": "1 Ave & E 68 St",
           "longitude": -73.958115339,
           "latitude": 40.765112281}
}

station_df = pd.DataFrame.from_dict(station_dict, orient='index').reset_index()
station_df.rename(columns={'index': 'start_station_id'}, inplace=True)
predictions = pd.merge(predictions, station_df, on='start_station_id', how='left')
progress_bar.progress(2 / N_STEPS)

# Map visualization
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

# Data table
st.subheader("Top Predictions")
st.dataframe(
    predictions[["start_station_id", "name", "start_hour", "predicted_demand"]]
    .sort_values("predicted_demand", ascending=False)
    .reset_index(drop=True)
)