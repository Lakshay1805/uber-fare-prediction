import joblib
from ml.feature_engineering import predict_feature_pipeline

model = joblib.load("./ml/model.pkl")

def prediction(data):
    data = predict_feature_pipeline(data)
    prediction = model.predict(data)
    return prediction