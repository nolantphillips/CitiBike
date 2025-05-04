import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import calendar

# Add the parent directory to the Python path
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytz
import requests
from io import BytesIO
import zipfile

from src.config import RAW_DATA_DIR

def fetch_raw_data(year:int, month:int) -> str:
    if (year == 2024 and month in range(1, 5)) or (year == 2025 and month == 3):
        url = f"https://s3.amazonaws.com/tripdata/{year}{month:02}-citibike-tripdata.csv.zip"

        response = requests.get(url)
        if response.status_code == 200:
            path = Path('..') / 'data' / 'raw' / f"citi_rides_{year}_{month:02}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(BytesIO(response.content)) as z:
                csv_files = [f for f in z.namelist() if f.lower().endswith('.csv') and 'macosx' not in f.lower()]

                if not csv_files:
                    print("No CSV files found in zip file")
                else:
                    dfs = []
                    for csv_file in csv_files:
                        with z.open(csv_file) as f:
                            df = pd.read_csv(f)
                            for col in ['start_station_id', 'end_station_id']:
                                if col in df.columns:
                                    df[col] = df[col].astype(str)

                            dfs.append(df)

                    if dfs:
                        combined_df = pd.concat(dfs, ignore_index=True)
                        combined_df.to_parquet(path, index=False)
                        print(f"Saved {len(combined_df)} rows to {path}")
                        return str(path)
        else:
            print("Incorrect URL")

    elif (year >= 2024 and month in range (5, 13)) or (year == 2025 and month in (1, 2)):
        url = f"https://s3.amazonaws.com/tripdata/{year}{month:02}-citibike-tripdata.zip"

        response = requests.get(url)
        if response.status_code == 200:
            path = RAW_DATA_DIR / f"citi_rides_{year}_{month:02}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(BytesIO(response.content)) as z:
                csv_files = [f for f in z.namelist() if f.lower().endswith('.csv') and 'macosx' not in f.lower()]

                if not csv_files:
                    print("No CSV files found in zip file")
                else:
                    dfs = []
                    for csv_file in csv_files:
                        with z.open(csv_file) as f:
                            df = pd.read_csv(f)
                            for col in ['start_station_id', 'end_station_id']:
                                if col in df.columns:
                                    df[col] = df[col].astype(str)

                            dfs.append(df)

                    if dfs:
                        combined_df = pd.concat(dfs, ignore_index=True)
                        combined_df.to_parquet(path, index=False)
                        print(f"Saved {len(combined_df)} rows to {path}")
                        return str(path)
        else:
            print("Incorrect URL")

    elif year < 2024:
        url = f"https://s3.amazonaws.com/tripdata/{year}-citibike-tripdata.zip"

        response = requests.get(url)
        if response.status_code == 200:
            path = RAW_DATA_DIR / f"citi_rides_{year}_{month:02}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)

            outer_zip = zipfile.ZipFile(BytesIO(response.content))
            target_month_prefix = f"{year}{month:02}"
            monthly_zip_name = next(
                (f for f in outer_zip.namelist()
                 if f.endswith('.zip') and target_month_prefix in f and 'macosx' not in f.lower()),
                 None
            )

            if not monthly_zip_name:
                print(f"No zip found for {target_month_prefix}")
        
            with outer_zip.open(monthly_zip_name) as nested_zip_file:
                with zipfile.ZipFile(BytesIO(nested_zip_file.read())) as inner_zip:
                    csv_files = [f for f in inner_zip.namelist()
                                if f.endswith('.csv') and 'macosx' not in f.lower()]
                    dfs = []

                    for csv_file in csv_files:
                        with inner_zip.open(csv_file) as f:
                            df = pd.read_csv(f)

                            for col in ['start_station_id', 'end_station_id']:
                                if col in df.columns:
                                    df[col] = df[col].astype(str)

                            dfs.append(df)

                    if dfs:
                        combined_df = pd.concat(dfs, ignore_index=True)
                        combined_df.to_parquet(path, index=False)
                        print(f"Saved {len(combined_df)} rows to {path}")
                        return str(path)
                    else:
                        print("No CSV files found in the inner ZIP.")
        else:
            print("File not found")

def filter_bike_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Filters CitiBike ride data for a specific year and month, removing outliers and invalid records.
    Args:
        rides (pd.DataFrame): DataFrame containing CitiBike ride data.
        year (int): Year to filter for.
        month (int): Month to filter for (1-12).

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid rides for the specified year and month.

    Raises:
        ValueError: If no valid rides are found or if input parameters are invalid.
    """

    # Validate inputs
    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12")
    if not isinstance(year, int) or not isinstance(month, int):
        raise ValueError("Year and month must be integers.")
    
    # Calculate start and end dates for the specified month
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = pd.Timestamp(year=year + (month // 12), month=(month % 12) + 1, day=1)

    # Add a duration column for filtering
    time_cols = ['started_at', 'ended_at']
    for col in time_cols:
        rides[col] = pd.to_datetime(rides[col], errors='coerce')
    rides['duration'] = rides['ended_at'] - rides['started_at']

    # Define filters
    duration_filter = (rides["duration"] > pd.Timedelta(0)) & (rides["duration"] <= pd.Timedelta(hours=1))
    location_filter = rides["start_station_id"].isin(('5905.14', '6140.05', '6822.09'))
    date_range_filter = (rides["started_at"] >= start_date) & (rides["ended_at"] < end_date)

    # Combine all filters
    final_filter = (
        duration_filter & location_filter & date_range_filter
    )

    # Calculate dropped records
    total_records = len(rides)
    valid_records = final_filter.sum()
    records_dropped = total_records - valid_records
    percent_dropped = (records_dropped / total_records) * 100

    print(f"Total records: {total_records:,}")
    print(f"Valid records: {valid_records:,}")
    print(f"Records dropped: {records_dropped:,} ({percent_dropped:.2f}%)")

    # Filter the DataFrame
    validated_rides = rides[final_filter]
    validated_rides = validated_rides[["started_at", "start_station_id"]]
    validated_rides["start_station_id"] = validated_rides["start_station_id"].astype(float)

    # Verify we have data in the correct time range
    if validated_rides.empty:
        raise ValueError(f"No valid rides found for {year}-{month:02} after filtering.")

    return validated_rides

def load_and_process_bike_data(
        years: Optional[List[int]], months: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load and process CitiBike ride data for a specified year and list of months.

    Args:
        year (Optional[List[int]]): List of years to load.
        months (Optional[List[int]]): List of months to load. If None, loads all months (1-12).

    Returns:
        pd.DataFrame: Combined and processed ride data for the specified year and months.

    Raises:
        Exception: If no data could be loaded for the specified year and months.
    """

    # Use all months if none are specified
    if months is None:
        months = list(range(1, 13))

    # List to store DataFrames for each month
    yearly_rides = []
    for year in years:
        monthly_rides = []
        if year == 2025:
            months = list(range(1,4))

        for month in months:
            # Construct the file path
            file_path = RAW_DATA_DIR / f"citi_rides_{year}_{month:02}.parquet"

            try:
                # Download the file if it doesn't exist
                if not file_path.exists():
                    print(f"Downloading data for {year}-{month:02}...")
                    fetch_raw_data(year, month)
                    print(f"Successfully downloaded data for {year}-{month:02}.")
                else:
                    print(f"File already exists for {year}-{month:02}.")

                # Load the data
                print(f"Loading data for {year}-{month:02}...")
                rides = pd.read_parquet(file_path, engine="pyarrow")

                # Filter and process the data
                rides = filter_bike_data(rides, year, month)
                print(f"Successfully processed data for {year}-{month:02}.")

                # Append the processed DataFrame to the list
                monthly_rides.append(rides)

            except FileNotFoundError:
                print(f"File not found for {year}-{month:02}. Skipping...")
            except Exception as e:
                print(f"Error processing data for {year}-{month:02}: {str(e)}")
                continue

        # Combine all monthly data
        if not monthly_rides:
            raise Exception(
                f"No data could be loaded for the year {year} and specified months: {months}"
            )

        print("Combining all monthly data...")
        month_combined_rides = pd.concat(monthly_rides, ignore_index=True)
        print(f"Data loading and processing for {year} complete!")
        yearly_rides.append(month_combined_rides)

    print("Combining all data...")
    combined_rides = pd.concat(yearly_rides, ignore_index=True)
    print("Data loading and processing complete!")

    return combined_rides.sort_values(by="started_at", ascending=True)

def fill_missing_rides_full_range(df, hour_col, location_col, rides_col):
    """
    Fills in missing rides for all hours in the range and all unique locations.

    Parameters:
    - df: DataFrame with columns [hour_col, location_col, rides_col]
    - hour_col: Name of the column containing hourly timestamps
    - location_col: Name of the column containing location IDs
    - rides_col: Name of the column containing ride counts

    Returns:
    - DataFrame with missing hours and locations filled in with 0 rides
    """
    # Ensure the hour column is in datetime format
    df[hour_col] = pd.to_datetime(df[hour_col])

    # Get the full range of hours (from min to max) with hourly frequency
    full_hours = pd.date_range(
        start=df[hour_col].min(),
        end=df[hour_col].max(),
        freq="h"
    )

    # Get all unique location IDs
    all_locations = df[location_col].unique()

    # Create a DataFrame with all combinations of hours and locations
    full_combinations = pd.DataFrame(
        [(hour, location) for hour in full_hours for location in all_locations],
        columns=[hour_col, location_col]
    )

    # Merge the original DataFrame with the full combinations DataFrame
    merged_df = pd.merge(full_combinations, df, on=[hour_col, location_col], how='left')

    # Fill missing rides with 0
    merged_df[rides_col] = merged_df[rides_col].fillna(0).astype(int)

    return merged_df

def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw ride data into time series format.

    Args:
        rides: DataFrame with pickup_datetime and location columns

    Returns:
        pd.DataFrame: Time series data with filled gaps
    """
    # Floor datetime to hour efficiently
    rides["start_hour"] = rides["started_at"].dt.floor("h")

    # Aggregate and fill gaps
    agg_rides = (
        rides.groupby(["start_hour", "start_station_id"])
        .size()
        .reset_index(name="rides")
    )

    agg_rides_all_slots = (
        fill_missing_rides_full_range(
            agg_rides, "start_hour", "start_station_id", "rides"
        )
        .sort_values(["start_station_id", "start_hour"])
        .reset_index(drop=True)
    )

    # important
    agg_rides_all_slots = agg_rides_all_slots.astype(
        {"start_station_id": "float32", "rides": "int16"}
    )
    return agg_rides_all_slots

def transform_ts_data_into_features_and_target_loop(
    df, feature_col="rides", window_size=12, step_size=1
):
    """
    Transforms time series data for all unique location IDs into a tabular format.
    The first `window_size` rows are used as features, and the next row is the target.
    The process slides down by `step_size` rows at a time to create the next set of features and target.
    Feature columns are named based on their hour offsets relative to the target.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing time series data with 'start_hour' column.
        feature_col (str): The column name containing the values to use as features and target (default is "rides").
        window_size (int): The number of rows to use as features (default is 12).
        step_size (int): The number of rows to slide the window by (default is 1).

    Returns:
        tuple: (features DataFrame with pickup_hour, targets Series, complete DataFrame)
    """
    # Get all unique location IDs
    location_ids = df["start_station_id"].unique()
    # List to store transformed data for each location
    transformed_data = []

    # Loop through each location ID and transform the data
    for location_id in location_ids:
        try:
            # Filter the data for the given location ID
            location_data = df[df["start_station_id"] == location_id].reset_index(
                drop=True
            )

            # Extract the feature column and pickup_hour as NumPy arrays
            values = location_data[feature_col].values
            times = location_data["start_hour"].values

            # Ensure there are enough rows to create at least one window
            if len(values) <= window_size:
                raise ValueError("Not enough data to create even one window.")

            # Create the tabular data using a sliding window approach
            rows = []
            for i in range(0, len(values) - window_size, step_size):
                # The first `window_size` values are features, and the next value is the target
                features = values[i : i + window_size]
                target = values[i + window_size]
                # Get the corresponding target timestamp
                target_time = times[i + window_size]
                # Combine features, target, location_id, and timestamp
                row = np.append(features, [target, location_id, target_time])
                rows.append(row)

            # Convert the list of rows into a DataFrame
            feature_columns = [
                f"{feature_col}_t-{window_size - i}" for i in range(window_size)
            ]
            all_columns = feature_columns + ["target", "start_station_id", "start_hour"]
            transformed_df = pd.DataFrame(rows, columns=all_columns)

            # Append the transformed data to the list
            transformed_data.append(transformed_df)

        except ValueError as e:
            print(f"Skipping start_station_id {location_id}: {str(e)}")

    # Combine all transformed data into a single DataFrame
    if not transformed_data:
        raise ValueError(
            "No data could be transformed. Check if input DataFrame is empty or window size is too large."
        )

    final_df = pd.concat(transformed_data, ignore_index=True)

    # Extract features (including pickup_hour), targets, and keep the complete DataFrame
    features = final_df[feature_columns + ["start_hour", "start_station_id"]]
    targets = final_df["target"]

    return features, targets

def split_time_series_data(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits a time series DataFrame into training and testing sets based on a cutoff date.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        cutoff_date (datetime): The date used to split the data into training and testing sets.
        target_column (str): The name of the target column to separate from the features.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            - X_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training target values.
            - X_test (pd.DataFrame): Testing features.
            - y_test (pd.Series): Testing target values.
    """
    # Split the data into training and testing sets based on the cutoff date
    train_data = df[df["start_hour"] < cutoff_date].reset_index(drop=True)
    test_data = df[df["start_hour"] >= cutoff_date].reset_index(drop=True)

    # Separate features (X) and target (y) for both training and testing sets
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    return X_train, y_train, X_test, y_test

def fetch_batch_raw_data(from_date: Union[datetime, str], to_date: Union[datetime, str]) -> pd.DataFrame:
    """
    Simulate production data by sampling historical data from 52 weeks ago (i.e., 1 year).

    Args:
        from_date (datetime or str): The start date for the data batch.
        to_date (datetime or str): The end date for the data batch.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated production data.
    """
    # Convert string inputs to datetime if necessary
    if isinstance(from_date, str):
        from_date = datetime.fromisoformat(from_date)
    if isinstance(to_date, str):
        to_date = datetime.fromisoformat(to_date)

    # Validate input dates
    if not isinstance(from_date, datetime) or not isinstance(to_date, datetime):
        raise ValueError("Both 'from_date' and 'to_date' must be datetime objects or valid ISO format strings.")
    if from_date >= to_date:
        raise ValueError("'from_date' must be earlier than 'to_date'.")

    # Shift dates back by 52 weeks (1 year)
    historical_from_date = from_date - timedelta(weeks=52)
    historical_to_date = to_date - timedelta(weeks=52)

    # Load and filter data for the historical period
    rides_from = load_and_process_bike_data(years=[historical_from_date.year], months=[historical_from_date.month])
    rides_from = rides_from[rides_from.started_at >= historical_from_date]

    if historical_to_date.month != historical_from_date.month:
        rides_to = load_and_process_bike_data(years=[historical_to_date.year], months=[historical_to_date.month])
        rides_to = rides_to[rides_to.started_at < historical_to_date]
        # Combine the filtered data
        rides = pd.concat([rides_from, rides_to], ignore_index=True)
    else:
        rides_from = rides_from[
            (rides_from.started_at >= historical_from_date) &
            (rides_from.started_at < historical_to_date)
        ]
        rides = rides_from
    # Shift the data forward by 52 weeks to simulate recent data
    rides['started_at'] += timedelta(weeks=52)

    # Sort the data for consistency
    rides.sort_values(by=['start_station_id', 'started_at'], inplace=True)

    return rides