from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

app = Flask(__name__)
CORS(app)

# -----------------------------------
# MUNICIPAL WATER DATASET
# -----------------------------------

data = {

    "household_size": [
        2, 3, 4, 5, 6,
        2, 3, 4, 5, 6,
        3, 4, 5, 6, 7
    ],

    "temperature": [
        22, 24, 26, 30, 34,
        21, 23, 27, 31, 35,
        25, 28, 32, 36, 38
    ],

    "month": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 1, 2, 3
    ],

    "water_usage": [
        120, 150, 180, 240, 300,
        130, 160, 210, 260, 320,
        170, 220, 280, 340, 380
    ]
}

df = pd.DataFrame(data)

# -----------------------------------
# FEATURES AND TARGET
# -----------------------------------

X = df[[
    "household_size",
    "temperature",
    "month"
]]

y = df["water_usage"]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# LINEAR REGRESSION
# -----------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

# -----------------------------------
# RANDOM FOREST
# -----------------------------------

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)

# -----------------------------------
# EVALUATION
# -----------------------------------

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

# -----------------------------------
# PRINT RESULTS
# -----------------------------------

print("\n========== MODEL EVALUATION ==========\n")

print("Linear Regression")
print("MAE :", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))

print("\nRandom Forest")
print("MAE :", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))

# -----------------------------------
# ACTUAL VS PREDICTED GRAPH
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    linear_predictions,
    label="Linear Regression"
)

plt.scatter(
    y_test,
    rf_predictions,
    label="Random Forest"
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()]
)

plt.xlabel("Actual Water Usage")
plt.ylabel("Predicted Water Usage")

plt.title("Actual vs Predicted Water Usage")

plt.legend()

plt.savefig("actual_vs_predicted.png")

print("\nGraph saved as:")
print("actual_vs_predicted.png")

# -----------------------------------
# HOME ROUTE
# -----------------------------------

@app.route("/")
def home():

    return "Municipal Water ML Server Running"

# -----------------------------------
# PREDICTION API
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        household_size = float(
            data["household_size"]
        )

        temperature = float(
            data["temperature"]
        )

        month = float(
            data["month"]
        )

        prediction = random_forest_model.predict([[
            household_size,
            temperature,
            month
        ]])

        result = round(float(prediction[0]), 2)

        return jsonify({

            "predicted_water_usage": result,

            "linear_regression_mae":
                round(linear_mae, 2),

            "linear_regression_rmse":
                round(linear_rmse, 2),

            "random_forest_mae":
                round(rf_mae, 2),

            "random_forest_rmse":
                round(rf_rmse, 2)

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# -----------------------------------
# RUN SERVER
# -----------------------------------

if __name__ == "__main__":

    app.run(debug=True)