import os
import pandas as pd
import numpy as np
import joblib

from feature_engineering import full_feature_pipeline
from preprocessing import full_process_pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

MODEL_FILE = "model.pkl"

df = pd.read_csv("../data/uber.csv")

df = full_feature_pipeline(df)

x = df.drop(columns="fare_amount")
y = df['fare_amount']

num_cols = x.select_dtypes(include="number").columns.tolist()
cat_cols = x.select_dtypes(include=["object" , "string" , "category"]).columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2, random_state=42)


preprocessor = full_process_pipeline(num_cols , cat_cols)

model_pipeline = Pipeline([
    ("processor" , preprocessor),
    ("model" , XGBRegressor())
])

param_grid_xg = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [4, 6, 8],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid_xg,
    cv=cv,
    scoring="neg_mean_absolute_error",
    verbose=2,
    n_jobs=-1
)

if not os.path.exists(MODEL_FILE):
    grid.fit(x_train , y_train)
    model = grid.best_estimator_
    print(f"Best CV MAE: {grid.best_score_:.4f}")
    print(grid.best_params_)

    joblib.dump(model , MODEL_FILE)

else:
    model = joblib.load(MODEL_FILE)

    y_pred = model.predict(x_test)

    print(f"r2 score is : {r2_score(y_test , y_pred)}")
    print(f"MAE : {mean_absolute_error(y_test , y_pred)}")
    print(f"MSE : {mean_squared_error(y_test , y_pred)}")
    print(f" RMSE : {np.sqrt(mean_squared_error(y_test, y_pred))}")