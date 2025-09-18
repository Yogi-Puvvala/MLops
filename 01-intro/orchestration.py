import fastparquet
import pyarrow
import pandas as pd
import numpy as np
import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

from prefect import task, flow


@task
def get_df_task():
    df = pd.read_parquet("green_trip_data.parquet", engine="pyarrow")
    df.head()

    df["duration"] = df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    df["duration"] = df["duration"].astype("str")

    def getMins(x):
        h, m, s = list(map(int, x.split()[2].split(":")))
        return h * 60 + m + s // 60

    df["duration"] = df["duration"].apply(lambda x: getMins(x))
    df["hour"] = df["lpep_pickup_datetime"].dt.hour
    df["minute"] = df["lpep_pickup_datetime"].dt.minute
    df["time_numeric"] = df["hour"] + df["minute"] / 60

    df["time_sin"] = np.sin(2 * np.pi * df["time_numeric"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["time_numeric"] / 24)

    nominal_cols = ["PULocationID", "DOLocationID"]
    numerical_cols = ["time_sin", "time_cos", "trip_distance", "total_amount"]

    df = df[numerical_cols + nominal_cols + ["duration"]]

    return df, numerical_cols, nominal_cols


@task
def evaluate_model_task(model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)  
    return {"r2_score": r2, "MSE": mse, "RMSE": rmse}

@task
def train_model_task(df, numerical_cols, nominal_cols, models):
    X = df.drop("duration", axis=1)
    y = df["duration"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(sparse_output=True, handle_unknown="ignore"), nominal_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )

    LR_model = Pipeline([("preprocess", preprocess), ("model", LinearRegression())])
    LR_model.fit(X_train, y_train)
    models["LR_model"] = LR_model

    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(sparse_output=True, handle_unknown="ignore"), nominal_cols),
        ]
    )

    RFR_model = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                max_features="sqrt",
                min_samples_split=5,
                min_samples_leaf=2,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )
    RFR_model.fit(X_train, y_train)
    models["RFR_model"] = RFR_model

    XGBR_model = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=3,
                gamma=0,
                reg_alpha=0.1,
                reg_lambda=1,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=42,
            )),
        ]
    )
    XGBR_model.fit(X_train, y_train)
    models["XGBR_model"] = XGBR_model

    return models


@task
def exp_tracking_task(exp_name, models, df):
    X = df.drop("duration", axis=1)
    y = df["duration"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment(exp_name)

    for mod in models:
        with mlflow.start_run(run_name=mod):
            mlflow.log_params(models[mod].get_params())
            mlflow.log_metrics(evaluate_model_task(models[mod], X_test, y_test))

            pred_vals = models[mod].predict(X_test)
            signature = mlflow.models.infer_signature(X_test, pred_vals)

            mlflow.sklearn.log_model(
                models[mod],
                artifact_path=mod,  
                signature=signature,
                input_example=pd.DataFrame(
                    {
                        "trip_distance": [180],
                        "total_amount": [250],
                        "PULocationID": [70],
                        "DOLocationID": [160],
                        "time_sin": [np.sin(2 * np.pi * 4 / 24)],
                        "time_cos": [np.cos(2 * np.pi * 4 / 24)],
                    }
                ),
            )
            mlflow.log_artifact("green_trip_data.parquet", artifact_path="dataset")


@task
def model_registry_task(exp_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(exp_name)
    exp_id = experiment.experiment_id
    best_run = client.search_runs(
        experiment_ids=[exp_id], order_by=["metrics.RMSE ASC"], max_results=1
    )[0]
    
    best_run_id = best_run.info.run_id
    best_run_mod = best_run.data.tags["mlflow.runName"]
    model_uri = f"runs:/{best_run_id}/{best_run_mod}"

    try:
        champ_model = client.get_model_version_by_alias(exp_name, "champion")
        champ_run_id = champ_model.run_id
        champ_metric = client.get_run(champ_run_id).data.metrics["RMSE"]
        new_metric = client.get_run(best_run_id).data.metrics["RMSE"]

        new_reg_mod = mlflow.register_model(model_uri, name=exp_name)

        if champ_metric > new_metric:
            client.set_registered_model_alias(
                name=exp_name, alias="challenger", version=new_reg_mod.version
            )
        else:
            client.set_registered_model_alias(
                name=exp_name, alias="challenger", version=champ_model.version
            )
            client.set_registered_model_alias(
                name=exp_name, alias="champion", version=new_reg_mod.version
            )

    except Exception:
        registered_model = mlflow.register_model(model_uri, name=exp_name)
        client.set_registered_model_alias(
            name=exp_name, alias="champion", version=registered_model.version
        )


@task
def predict_vals_task(df, exp_name):
    X = df.drop("duration", axis=1)
    y = df["duration"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_uri = f"models:/{exp_name}@champion"  # FIXED
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    print("Predicted Values:", loaded_model.predict(X_test))


@flow(name = "ML pipeline with mlflow + prefect")
def ml_pipeline_prefect(exp_name, models):
    df, numerical_cols, nominal_cols = get_df_task()
    models = train_model_task(df, numerical_cols, nominal_cols, models)
    exp_tracking_task(exp_name, models, df)
    model_registry_task(exp_name)
    predict_vals_task(df, exp_name)


if __name__ == "__main__":
    exp_name = "Duration Prediction Model v4"
    models = {}
    ml_pipeline_prefect(exp_name, models)
