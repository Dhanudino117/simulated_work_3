from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_FILE = "model.pkl"

# TRAIN MODEL FIRST TIME
if not os.path.exists(MODEL_FILE):

    data = {
        "hours": [1, 2, 3, 4, 5, 6, 7, 8],
        "score": [10, 20, 30, 40, 50, 60, 70, 80]
    }

    df = pd.DataFrame(data)

    X = df[["hours"]]
    y = df["score"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)

# LOAD MODEL
model = joblib.load(MODEL_FILE)

# HOME ROUTE
@app.route("/")
def home():
    return "ML Server Running"

# PREDICTION ROUTE
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    hours = data["hours"]

    prediction = model.predict([[hours]])

    return jsonify({
        "predicted_score": round(float(prediction[0]), 2)
    })

# START SERVER
if __name__ == "__main__":
    app.run(debug=True)