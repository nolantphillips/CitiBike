import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error
import pandas as pd

import src.config as config
from src.data_utils import transform_ts_data_into_features_and_target_loop
from src.inference import (
    fetch_days_data,
    get_hopsworks_project,
    load_metrics_from_registry,
    load_model_from_registry,
)
from src.pipeline_utils import get_pipeline

print(f"Fetching data from group store ...")
ts_data = fetch_days_data(180)

print(f"Transforming to ts_data ...")

features, targets = transform_ts_data_into_features_and_target_loop(
    ts_data, window_size=24 * 28, step_size=23
)

features_targets = features.copy()
features_targets["target"] = targets

cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=28))

df_5905 = features_targets[features_targets["start_station_id"] == 5905.140137]
df_6140 = features_targets[features_targets["start_station_id"] == 6140.049805]
df_6822 = features_targets[features_targets["start_station_id"] == 6822.089844]

X_train_5905, y_train_5905, X_test_5905, y_test_5905 = split_time_series_data(
    df_5905,
    cutoff_date=cutoff_date,
    target_column="target"
)

X_train_6140, y_train_6140, X_test_6140, y_test_6140 = split_time_series_data(
    df_6140,
    cutoff_date=cutoff_date,
    target_column="target"
)

X_train_6822, y_train_6822, X_test_6822, y_test_6822 = split_time_series_data(
    df_6822,
    cutoff_date=cutoff_date,
    target_column="target"
)

models_dict = {}
preds_dict = {}
mae = {}

for i in range(0,3):
    if i == 0:
        pipeline = get_pipeline()
        print(f"Training model 5905 ...")
        pipeline.fit(X_train_5905, y_train_5905)
        preds = pipeline.predict(X_test_5905)
        preds_dict[5905] = preds
        test_mae = mean_absolute_error(y_test_5905, preds)
        mae[5905] = test_mae
        models_dict[5905] = pipeline

        metric = load_metrics_from_registry(station_id=5905)

        print(f"The new MAE is {test_mae:.4f}")
        print(f"The previous MAE is {metric['test_mae']:.4f}")

        if test_mae < metric.get("test_mae"):
            print(f"Registering new model")
            model_path = config.MODELS_DIR / "lgb_model_5905.pkl"
            joblib.dump(pipeline, model_path)

            input_schema = Schema(X_train_5905)
            output_schema = Schema(y_train_5905)
            model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
            project = get_hopsworks_project()
            model_registry = project.get_model_registry()

            model = model_registry.sklearn.create_model(
                name="bike_demand_predictor_next_hour_5905",
                metrics={"test_mae": test_mae},
                description="LightGBM regressor",
                input_example=X_train_5905.sample(),
                model_schema=model_schema,
            )
            model.save(model_path)
        else:
            print(f"Skipping model registration because new model is not better!")

    if i == 1:
        pipeline = get_pipeline()
        print(f"Training model 6140 ...")
        pipeline.fit(X_train_6140, y_train_6140)
        preds = pipeline.predict(X_test_6140)
        preds_dict[6140] = preds
        test_mae = mean_absolute_error(y_test_6140, preds)
        mae[6140] = test_mae
        models_dict[6140] = pipeline

        metric = load_metrics_from_registry(station_id=6140)

        print(f"The new MAE is {test_mae:.4f}")
        print(f"The previous MAE is {metric['test_mae']:.4f}")

        if test_mae < metric.get("test_mae"):
            print(f"Registering new model")
            model_path = config.MODELS_DIR / "lgb_model_6140.pkl"
            joblib.dump(pipeline, model_path)

            input_schema = Schema(X_train_6140)
            output_schema = Schema(y_train_6140)
            model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
            project = get_hopsworks_project()
            model_registry = project.get_model_registry()

            model = model_registry.sklearn.create_model(
                name="bike_demand_predictor_next_hour_6140",
                metrics={"test_mae": test_mae},
                description="LightGBM regressor",
                input_example=X_train_6140.sample(),
                model_schema=model_schema,
            )
            model.save(model_path)
        else:
            print(f"Skipping model registration because new model is not better!")

    if i == 2:
        pipeline = get_pipeline()
        print(f"Training model 6822 ...")
        pipeline.fit(X_train_6822, y_train_6822)
        preds = pipeline.predict(X_test_6822)
        preds_dict[6822] = preds
        test_mae = mean_absolute_error(y_test_6822, preds)
        mae[6822] = test_mae
        models_dict[6822] = pipeline

        metric = load_metrics_from_registry(station_id=6822)

        print(f"The new MAE is {test_mae:.4f}")
        print(f"The previous MAE is {metric['test_mae']:.4f}")

        if test_mae < metric.get("test_mae"):
            print(f"Registering new model")
            model_path = config.MODELS_DIR / "lgb_model_6822.pkl"
            joblib.dump(pipeline, model_path)

            input_schema = Schema(X_train_6822)
            output_schema = Schema(y_test_6822)
            model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
            project = get_hopsworks_project()
            model_registry = project.get_model_registry()

            model = model_registry.sklearn.create_model(
                name="bike_demand_predictor_next_hour_6822",
                metrics={"test_mae": test_mae},
                description="LightGBM regressor",
                input_example=X_train_6822.sample(),
                model_schema=model_schema,
            )
            model.save(model_path)
        else:
            print(f"Skipping model registration because new model is not better!")