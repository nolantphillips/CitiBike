from datetime import datetime, timedelta, timezone

import hopsworks
import numpy as np
import pandas as pd
from hsfs.feature_store import FeatureStore

import src.config as config
from src.data_utils import transform_ts_data_into_features_and_target_loop

import pytz


def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
    )


def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()


def get_model_predictions(model_5905, model_6140, model_6822, features: pd.DataFrame) -> pd.DataFrame:
    # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
    features_5905 = features[features["start_station_id"] == 5905.140137]
    features_6140 = features[features["start_station_id"] == 6140.049805]
    features_6822 = features[features["start_station_id"] == 6822.089844]

    predictions_5905 = model_5905.predict(features_5905)
    predictions_6140 = model_6140.predict(features_6140)
    predictions_6822 = model_6822.predict(features_6822)

    predictions = np.concatenate([predictions_5905, predictions_6140, predictions_6822])

    results = pd.DataFrame()
    results["start_station_id"] = features.sort_values(by = "start_station_id", ascending=True)["start_station_id"].values
    results["predicted_demand"] = predictions.round(0)

    return results


def load_batch_of_features_from_store(
    current_date: datetime,
) -> pd.DataFrame:
    feature_store = get_feature_store()

    # read time-series data from the feature store
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
    )

    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
    )
    ts_data = ts_data[ts_data.start_hour.between(fetch_data_from, fetch_data_to)]

    # Sort data by location and time
    ts_data.sort_values(by=["start_station_id", "start_hour"], inplace=True)

    features, _ = transform_ts_data_into_features_and_target_loop(
        ts_data, window_size=24 * 28, step_size=23
    )

    return features


def load_model_from_registry(station_id, version=None):
    from pathlib import Path

    import joblib

    from src.pipeline_utils import (  # Import custom classes/functions
        TemporalFeatureEngineer,
        average_rides_last_4_weeks,
    )

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    if station_id == 5905:
        models = model_registry.get_models(name=config.MODEL_NAME_5905)
        model = max(models, key=lambda model: model.version)
        model_dir = model.download()
        model = joblib.load(Path(model_dir) / f"lgb_model_{station_id}.pkl")

        return model
    
    elif station_id == 6140:
        models = model_registry.get_models(name=config.MODEL_NAME_6140)
        model = max(models, key=lambda model: model.version)
        model_dir = model.download()
        model = joblib.load(Path(model_dir) / f"lgb_model_{station_id}.pkl")

        return model

    elif station_id == 6822:
        models = model_registry.get_models(name=config.MODEL_NAME_6822)
        model = max(models, key=lambda model: model.version)
        model_dir = model.download()
        model = joblib.load(Path(model_dir) / f"lgb_model_{station_id}.pkl")

        return model
    
def load_metrics_from_registry(station_id, version=None):

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    if station_id == 5905:
        models = model_registry.get_models(name=config.MODEL_NAME_5905)
        model = max(models, key=lambda model: model.version)

        return model.training_metrics
    
    elif station_id == 6140:
        models = model_registry.get_models(name=config.MODEL_NAME_6140)
        model = max(models, key=lambda model: model.version)

        return model.training_metrics

    elif station_id == 6822:
        models = model_registry.get_models(name=config.MODEL_NAME_6822)
        model = max(models, key=lambda model: model.version)

        return model.training_metrics

def fetch_next_hour_predictions():
    # Get current time and round up to next hour
    now = datetime.now() - timedelta(hours=5)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
    df = fg.read()

    df["start_hour"] = df["start_hour"].dt.tz_convert("US/Eastern")
    df["start_hour"] = df["start_hour"].dt.tz_localize(None)
    # Then filter for next hour in the DataFrame
    df = df[df["start_hour"] == next_hour]

    print(f"Current EST time: {now}")
    print(f"Next hour: {next_hour}")
    print(f"Found {len(df)} records")
    return df


def fetch_predictions(hours):
    current_hour = (pd.Timestamp.now() - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)

    df = fg.filter((fg.start_hour >= current_hour)).read()
    df["start_hour"] = df["start_hour"].dt.tz_convert("US/Eastern")
    return df


def fetch_hourly_rides(hours):
    current_hour = (pd.Timestamp.now() - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    query = fg.select_all()
    query = query.filter(fg.start_hour >= current_hour)
    df = query.read()
    df["start_hour"] = df["start_hour"].dt.tz_convert("US/Eastern")

    return df


def fetch_days_data(days):
    current_date = pd.to_datetime(datetime.now()).tz_localize("US/Eastern")
    fetch_data_from = current_date - timedelta(days=(365 + days))
    fetch_data_to = current_date - timedelta(days=365)
    print(fetch_data_from, fetch_data_to)
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    query = fg.select_all()
    # query = query.filter((fg.start_hour >= fetch_data_from))
    df = query.read()
    df["start_hour"] = df["start_hour"].dt.tz_convert("US/Eastern")
    cond = (df["start_hour"] >= fetch_data_from) & (df["start_hour"] <= fetch_data_to)
    return df[cond]