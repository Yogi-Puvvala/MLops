#!/usr/bin/env python
# coding: utf-8

# # Experiment Name:

# In[64]:


exp_name = "Duration Prediction Model v4"
models = {}


# In[65]:


import fastparquet
import pyarrow
import pandas as pd
import numpy as np


# In[66]:


df = pd.read_parquet("green_trip_data.parquet", engine = "pyarrow")


# In[67]:


df.head()


# In[68]:


df.isna().sum()


# In[69]:


df.info()


# In[70]:


# Important Feature's Imformation

# Target: Duration
# Model: Regression Model
# Independent features: trip_distance, total_amount, PULocationID, DOLocationID, pickup_time -> (which have to converted into time_sin and time_cos)  


# In[71]:


df["duration"] = df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
df["duration"] = df["duration"].astype("str")
df.loc[0, "duration"]


# In[72]:


def getMins(x):
    h, m, s = list(map(int, x.split()[2].split(":")))
    return h*60 + m + s // 60


# In[73]:


df["duration"] = df["duration"].apply(lambda x: getMins(x))


# In[74]:


df["hour"] = df["lpep_pickup_datetime"].dt.hour
df["minute"] = df["lpep_pickup_datetime"].dt.minute
df["time_numeric"] = df["hour"] + df["minute"] / 60


# In[75]:


df["time_sin"] = np.sin(2 * np.pi * df["time_numeric"] / 24)
df["time_cos"] = np.cos(2 * np.pi * df["time_numeric"] / 24)


# In[76]:


nominal_cols = ["PULocationID", "DOLocationID"]
numerical_cols = ["time_sin", "time_cos", "trip_distance", "total_amount"]

df = df[numerical_cols + nominal_cols + ["duration"]]


# In[77]:


df.head()


# In[78]:


X = df.drop("duration", axis = 1)
y = df["duration"]


# In[79]:


# models -> Xgboost, random_forest, linear_regression, knn


# In[80]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error


# ## LinearRegression Model

# In[81]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[82]:


preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(sparse_output = True, handle_unknown = "ignore"), nominal_cols),
    ("num", StandardScaler(), numerical_cols)
])


# In[83]:


LR_model = Pipeline([
    ("preprocess", preprocess),
    ("model", LinearRegression())
])


# In[84]:


LR_model.fit(X_train, y_train)


# In[85]:


print("Training Score:", LR_model.score(X_train, y_train))
print("Testing Score:", LR_model.score(X_test, y_test))


# In[86]:


models["LR_model"] = LR_model


# ## RandomForestRegressor

# In[87]:


preprocess = ColumnTransformer([
    ("num", "passthrough", numerical_cols),
    ("cat", OneHotEncoder(sparse_output = True, handle_unknown = "ignore"), nominal_cols)
])


# In[88]:


RFR_model = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(
            n_estimators=200,           # Number of trees in the forest
            max_depth=15,               # Controls tree depth to prevent overfitting
            max_features='sqrt',        # Good balance between randomness and performance
            min_samples_split=5,        # Minimum samples to split a node
            min_samples_leaf=2,         # Minimum samples at a leaf node
            bootstrap=True,             # Enables sampling with replacement
            random_state=42,            # Ensures reproducibility
            n_jobs=-1 
    ))
])


# In[89]:


RFR_model.fit(X_train, y_train)


# In[90]:


print("Training Score:", RFR_model.score(X_train, y_train))
print("Testing Score:", RFR_model.score(X_test, y_test))


# In[91]:


models["RFR_model"] = RFR_model


# ## XGBoostRegressor

# In[92]:


XGBR_model = Pipeline([
    ("preprocess", preprocess),
    ("model", XGBRegressor(
            n_estimators=300,           # Number of boosting rounds
            learning_rate=0.05,         # Step size shrinkage
            max_depth=6,                # Controls tree complexity
            subsample=0.8,              # Fraction of samples used per tree
            colsample_bytree=0.7,       # Fraction of features used per tree
            min_child_weight=3,         # Minimum sum of instance weight in a child
            gamma=0,                    # Minimum loss reduction to make a split
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=1,               # L2 regularization
            objective='reg:squarederror', # Standard regression loss
            n_jobs=-1,                  # Parallelize across all cores
            random_state=42 
    ))
])


# In[93]:


XGBR_model.fit(X_train, y_train)


# In[94]:


print("Training Score:", XGBR_model.score(X_train, y_train))
print("Testing Score:", XGBR_model.score(X_test, y_test))


# In[95]:


models["XGBR_model"] = XGBR_model


# ## Exp-Tracking and Adding Models to registry

# In[96]:


import mlflow


# In[97]:


import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as r2_fn, mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(exp_name)

for mod in models:
    with mlflow.start_run(run_name=mod):

        # Logging Params
        mlflow.log_params(models[mod].get_params())

        # Logging Metrics
        pred_vals = models[mod].predict(X_test)
        r2 = r2_fn(y_test, pred_vals)
        mse = mean_squared_error(y_test, pred_vals)
        rmse = np.sqrt(mse)

        mlflow.log_metrics({
            "r2_score": r2,
            "MSE": mse,
            "RMSE": rmse
        })

        # Infer model signature
        signature = mlflow.models.infer_signature(X_test, pred_vals)

        # Logging Model
        mlflow.sklearn.log_model(
            models[mod],
            name=mod, 
            # registered_model_name=exp_name,
            signature=signature,
            input_example=pd.DataFrame({
                "trip_distance": [180],
                "total_amount": [250],
                "PULocationID": [70],
                "DOLocationID": [160],
                "time_sin": [np.sin(2 * np.pi * 4/24)],
                "time_cos": [np.cos(2 * np.pi * 4/24)]    
            })
        )

        # Logging Artifact (single file)
        mlflow.log_artifact("green_trip_data.parquet", artifact_path="dataset")


# ## Setting alias according to the metrics

# In[98]:


from mlflow import MlflowClient


# In[102]:


from mlflow import MlflowClient
import mlflow

client = MlflowClient()

experiment = client.get_experiment_by_name(exp_name)
exp_id = experiment.experiment_id

best_run = client.search_runs(
    experiment_ids=[exp_id],
    order_by=["metrics.RMSE ASC"],
    max_results=1
)
best_run_id = best_run[0].info.run_id

# Use correct artifact path from logging
mod_name = best_run[0].data.tags.get("mlflow.runName")  # Assuming run_name=mod was set
model_uri = f"runs:/{best_run_id}/{mod_name}"

try:
    champ_model = client.get_model_version_by_alias(exp_name, "champion")

    champ_run_id = champ_model.run_id
    champ_metric = client.get_run(champ_run_id).data.metrics["RMSE"]
    new_metric = client.get_run(best_run_id).data.metrics["RMSE"]

    # Register new model version
    new_reg_mod = mlflow.register_model(model_uri, name=exp_name)

    if champ_metric > new_metric:
        client.set_registered_model_alias(name=exp_name, alias="challenger", version=new_reg_mod.version)
    else:
        client.set_registered_model_alias(name=exp_name, alias="challenger", version=champ_model.version)
        client.set_registered_model_alias(name=exp_name, alias="champion", version=new_reg_mod.version)

except Exception:
    # No champion exists → register and promote
    registered_model = mlflow.register_model(model_uri, name=exp_name)
    client.set_registered_model_alias(name=exp_name, alias="champion", version=registered_model.version)


# ## Loading the "champion" version

# In[100]:


model_uri = f"models:/{exp_name}@champion"

mod = mlflow.pyfunc.load_model(model_uri)


# In[101]:


mod.predict(X_test)


# In[ ]:




