import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Define directories
PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
MODELS_DIR = PARENT_DIR / "models"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

FEATURE_GROUP_NAME = "bike_time_series_hourly_feature_group"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "bike_time_series_hourly_feature_view"
FEATURE_VIEW_VERSION = 1

FEATURE_GROUP_MODEL_PREDICTION = "bike_hourly_model_prediction"

MODEL_NAME_5905 = "bike_demand_predictor_next_hour_5905"
MODEL_VERSION_5905 = 1
MODEL_NAME_6140 = "bike_demand_predictor_next_hour_6140"
MODEL_VERSION_6140 = 1
MODEL_NAME_6822 = "bike_demand_predictor_next_hour_6822"
MODEL_VERSION_6822 = 1