from datetime import timedelta
from typing import Optional

import pandas as pd
import plotly.express as px


import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from typing import Optional
from datetime import timedelta

def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: int,
    predictions: Optional[pd.Series] = None,
) -> Figure:
    """
    Plots the time series data for a specific row of NYC taxi data.
    """
    # Get the row by index
    location_features = features.iloc[row_id]
    actual_target = targets.iloc[row_id]

    # Identify time series columns
    time_series_columns = [col for col in features.columns if col.startswith("rides_t-")]
    time_series_values = [location_features[col] for col in time_series_columns] + [actual_target]

    # Generate timestamps
    start_time = pd.to_datetime(location_features["start_hour"])
    time_series_dates = pd.date_range(
        start=start_time - timedelta(hours=len(time_series_columns)),
        end=start_time,
        freq="h",
    )

    # Title
    title = f"Start Hour: {location_features['start_hour']}, Start Station ID: {location_features['start_station_id']}"

    # Base plot
    fig = px.line(
        x=time_series_dates,
        y=time_series_values,
        template="plotly_white",
        markers=True,
        title=title,
        labels={"x": "Time", "y": "Ride Counts"},
    )

    # Actual target marker
    fig.add_scatter(
        x=[time_series_dates[-1]],
        y=[actual_target],
        mode="markers",
        marker=dict(size=10, color="green"),
        name="Actual Value",
    )

    # Prediction marker if available
    if predictions is not None:
        predicted_value = predictions.iloc[row_id]
        fig.add_scatter(
            x=[time_series_dates[-1]],
            y=[predicted_value],
            mode="markers",
            marker=dict(size=15, color="red", symbol="x"),
            name="Prediction",
        )

    return fig

def plot_prediction(features: pd.DataFrame, prediction: int):
    # Identify time series columns (e.g., historical ride counts)
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]
    time_series_values = [
        features[col].iloc[0] for col in time_series_columns
    ] + prediction["predicted_demand"].to_list()

    # Convert pickup_hour Series to single timestamp
    pickup_hour = pd.Timestamp(features["start_hour"].iloc[0])

    # Generate corresponding timestamps for the time series
    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(time_series_columns)),
        end=pickup_hour,
        freq="h",
    )

    # Create a DataFrame for the historical data
    historical_df = pd.DataFrame(
        {"datetime": time_series_dates, "rides": time_series_values}
    )

    # Create the plot title with relevant metadata
    title = f"Start Hour: {pickup_hour}, Start Station ID: {features['start_station_id'].iloc[0]}"

    # Create the base line plot
    fig = px.line(
        historical_df,
        x="datetime",
        y="rides",
        template="plotly_white",
        markers=True,
        title=title,
        labels={"datetime": "Time", "rides": "Ride Counts"},
    )

    # Add prediction point
    fig.add_scatter(
        x=[pickup_hour],  # Last timestamp
        y=prediction["predicted_demand"].to_list(),
        line_color="red",
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        name="Prediction",
    )

    return fig