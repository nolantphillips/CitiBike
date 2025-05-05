import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)


import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import fetch_hourly_rides, fetch_predictions

st.title("Mean Absolute Error (MAE) by Start Hour")

# Sidebar for user input
st.sidebar.header("Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,  # Minimum allowed value
    max_value=24 * 28,  # (Optional) Maximum allowed value
    value=12,  # Initial/default value
    step=1,  # Step size for increment/decrement
)

station_dict = {
    int(5905): {"name": "Broadway & E 14 St",
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

# Fetch data
st.write("Fetching data for the past", past_hours, "hours...")
df1 = fetch_hourly_rides(past_hours)
df2 = fetch_predictions(past_hours)

# Merge the DataFrames on 'start_station_id' and 'start_hour'
merged_df = pd.merge(df1, df2, on=["start_station_id", "start_hour"])

# Calculate the absolute error
merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])

# Group by 'pickup_hour' and calculate the mean absolute error (MAE)
mae_by_hour = merged_df.groupby("start_hour")["absolute_error"].mean().reset_index()
mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)

# Create a Plotly plot
fig = px.line(
    mae_by_hour,
    x="start_hour",
    y="MAE",
    title=f"Mean Absolute Error (MAE) for the Past {past_hours} Hours",
    labels={"start_hour": "Start Hour", "MAE": "Mean Absolute Error"},
    markers=True,
)

# Display the plot
st.plotly_chart(fig)
st.write(f'Average MAE: {mae_by_hour["MAE"].mean()}')

mae_by_station = (
    merged_df.groupby(["start_station_id", "start_hour"])["absolute_error"]
    .mean()
    .reset_index()
    .rename(columns={"absolute_error": "MAE"})
)
mae_by_station["start_station_id"] = mae_by_station["start_station_id"].round(0).astype(int)

station_ids = mae_by_station["start_station_id"].unique().round(0).astype(int)
selected_station = st.sidebar.selectbox("Select Station ID", sorted(station_ids))
station_name = station_dict[selected_station]["name"]

station_mae = mae_by_station[mae_by_station["start_station_id"] == selected_station]

fig = px.line(
    station_mae,
    x="start_hour",
    y="MAE",
    title=f"MAE for Station {station_name} (Past {past_hours} Hours)",
    labels={"start_hour": "Start Hour", "MAE": "Mean Absolute Error"},
    markers=True,
)

# Display
st.plotly_chart(fig)
st.write(f'Average MAE for Station {selected_station}: {station_mae["MAE"].mean():.2f}')