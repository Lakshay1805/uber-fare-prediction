from flask import Blueprint, request, jsonify
import pandas as pd

from ml.predict import prediction

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    print(data)

    df = pd.DataFrame([data])

    pred = prediction(df)

    return jsonify({
        "predicted_fare": round(float(pred[0]), 2)
    })