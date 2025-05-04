from datetime import datetime, timedelta

import pandas as pd

import src.config as config
from src.inference import (
    get_feature_store,
    get_model_predictions,
    load_model_from_registry,
)

# Get the current datetime64[us, Etc/UTC]
# for number in range(22, 24 * 29):
# current_date = pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=number)
current_date = pd.Timestamp.now(tz="US/Eastern")
feature_store = get_feature_store()

# read time-series data from the feature store
fetch_data_to = current_date - timedelta(hours=1)
fetch_data_from = current_date - timedelta(days=1 * 29)
print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
)

ts_data = feature_view.get_batch_data(
    start_time=(fetch_data_from - timedelta(days=1)),
    end_time=(fetch_data_to + timedelta(days=1)),
)
ts_data = ts_data[ts_data.start_hour.between(fetch_data_from, fetch_data_to)]
ts_data.sort_values(["start_station_id", "start_hour"]).reset_index(drop=True)
ts_data["start_hour"] = ts_data["start_hour"].dt.tz_convert("US/Eastern")

from src.data_utils import transform_ts_data_into_features_and_target_loop

features, _ = transform_ts_data_into_features_and_target_loop(
    ts_data, window_size=24 * 28, step_size=23
)

model_5905 = load_model_from_registry(station_id=5905)
model_6140 = load_model_from_registry(station_id=6140)
model_6822 = load_model_from_registry(station_id=6822)

predictions = get_model_predictions(model_5905=model_5905, model_6140=model_6140, model_6822=model_6822, features)
predictions["start_hour"] = current_date.ceil("h")
print(predictions)

feature_group = get_feature_store().get_or_create_feature_group(
    name=config.FEATURE_GROUP_MODEL_PREDICTION,
    version=1,
    description="Predictions from LGBM Model",
    primary_key=["start_station_id", "start_hour"],
    event_time="start_hour",
)

feature_group.insert(predictions, write_options={"wait_for_job": False})