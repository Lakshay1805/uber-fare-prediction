import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from feature_engineering import full_feature_pipeline
from preprocessing import full_process_pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  cross_val_score

df = pd.read_csv("../data/uber.csv")
df = full_feature_pipeline(df)

x = df.drop(columns="fare_amount")
y = df['fare_amount']

num_cols = x.select_dtypes(include="number").columns.tolist()
cat_cols = x.select_dtypes(include=["object" , "string" , "category"]).columns.tolist()

preprocessor = full_process_pipeline(num_cols , cat_cols)

lr = Pipeline([
    ("processing" , preprocessor),
    ("model" , LinearRegression())
])
lr_mae_score = cross_val_score(lr , x , y , cv=5, scoring="neg_mean_absolute_error" , n_jobs=-1)
lr_mse_score = cross_val_score(lr , x , y , cv=5, scoring = "neg_mean_squared_error" , n_jobs=-1)
lr_rmse_score = np.sqrt(- lr_mse_score)
lr_r2_score = cross_val_score(lr , x , y , cv=5, scoring="r2" , n_jobs=-1)

print("MAE:", -lr_mae_score.mean())
print("MAE Std:", lr_mae_score.std())

print("RMSE:", lr_rmse_score.mean())
print("RMSE Std:", lr_rmse_score.std())

print("R2:", lr_r2_score.mean())
print("R2 Std:", lr_r2_score.std())

print(
    "-------------------------------------------------------------------------------------"
)

rf = Pipeline([
    ("processing" , preprocessor),
    ("model" , RandomForestRegressor())
])
rf_mae_score = cross_val_score(rf , x , y , cv=5 , scoring="neg_mean_absolute_error" , n_jobs=-1)
rf_mse_score = cross_val_score(rf , x , y , cv=5 , scoring = "neg_mean_squared_error" , n_jobs=-1)
rf_rmse_score = np.sqrt(- rf_mse_score)
rf_r2_score = cross_val_score(rf , x , y , cv=5 , scoring="r2" , n_jobs=-1)

print("MAE:", -rf_mae_score.mean())
print("MAE Std:", rf_mae_score.std())

print("RMSE:", rf_rmse_score.mean())
print("RMSE Std:", rf_rmse_score.std())

print("R2:", rf_r2_score.mean())
print("R2 Std:", rf_r2_score.std())

print(
    "-------------------------------------------------------------------------------------"
)

dt = Pipeline([
    ("processing" , preprocessor),
    ("model" , DecisionTreeRegressor())
])
dt_mae_score = cross_val_score(dt , x , y , cv=5 , scoring="neg_mean_absolute_error" , n_jobs=-1)
dt_mse_score = cross_val_score(dt , x , y , cv=5 , scoring = "neg_mean_squared_error" , n_jobs=-1)
dt_rmse_score = np.sqrt(- dt_mse_score)
dt_r2_score = cross_val_score(dt , x , y , cv=5 , scoring="r2" , n_jobs=-1)

print("MAE:", -dt_mae_score.mean())
print("MAE Std:", dt_mae_score.std())

print("RMSE:", dt_rmse_score.mean())
print("RMSE Std:", dt_rmse_score.std())

print("R2:", dt_r2_score.mean())
print("R2 Std:", dt_r2_score.std())

print(
    "-------------------------------------------------------------------------------------"
)

xg = Pipeline([
    ("processing" , preprocessor),
    ("model" , XGBRegressor())
])
xg_mae_score = cross_val_score(xg , x , y , cv=5 , scoring="neg_mean_absolute_error" , n_jobs=-1)
xg_mse_score = cross_val_score(xg , x , y , cv=5 , scoring = "neg_mean_squared_error" , n_jobs=-1)
xg_rmse_score = np.sqrt(- xg_mse_score)
xg_r2_score = cross_val_score(xg , x , y , cv=5 , scoring="r2" , n_jobs=-1)

print("MAE:", -xg_mae_score.mean())
print("MAE Std:", xg_mae_score.std())

print("RMSE:", xg_rmse_score.mean())
print("RMSE Std:", xg_rmse_score.std())

print("R2:", xg_r2_score.mean())
print("R2 Std:", xg_r2_score.std())



# We will choose XGboost  as it performs best across all models .