import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ModuleNotFoundError:
    Flask = None
    jsonify = None
    request = None
    CORS = None
    FLASK_AVAILABLE = False


DATA_PATH = Path('processed_data/water_usage_clean.csv')
PLOT_PATH = Path('actual_vs_predicted.png')
MODEL_PATH = Path('model.pkl')
METRICS_PATH = Path('output/model_comparison.csv')
TARGET_COLUMN = 'usage_liters'
NUMERIC_FEATURES = ['household_size', 'month_num', 'is_summer_peak']
CATEGORICAL_FEATURES = ['neighborhood', 'season']
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DEFAULT_NEIGHBORHOOD = 'Green Park'


app = Flask(__name__) if FLASK_AVAILABLE else None
if app is not None:
    CORS(app)


def month_to_season(month_num):
    if month_num in [12, 1, 2]:
        return 'Winter'
    if month_num in [3, 4, 5]:
        return 'Spring'
    if month_num in [6, 7, 8]:
        return 'Summer'
    return 'Autumn'


def normalize_neighborhood(value):
    if value is None or str(value).strip() == '':
        return DEFAULT_NEIGHBORHOOD
    return str(value).strip().title()


def build_prediction_frame(data):
    errors = []

    try:
        household_size = float(data.get('household_size'))
    except (TypeError, ValueError):
        household_size = None
        errors.append("household_size must be a number.")

    try:
        month_num = int(data.get('month'))
    except (TypeError, ValueError):
        month_num = None
        errors.append("month must be an integer from 1 to 12.")

    if household_size is not None and household_size < 1:
        errors.append("household_size must be at least 1.")

    if month_num is not None and month_num not in range(1, 13):
        errors.append("month must be between 1 and 12.")

    if errors:
        return None, errors

    neighborhood = normalize_neighborhood(data.get('neighborhood'))
    season = month_to_season(month_num)
    prediction_frame = pd.DataFrame([{
        'household_size': household_size,
        'month_num': month_num,
        'is_summer_peak': month_num in [6, 7, 8],
        'neighborhood': neighborhood,
        'season': season,
    }])

    return prediction_frame, []


def make_preprocessor():
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ('numeric', numeric_pipeline, NUMERIC_FEATURES),
        ('categorical', categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def load_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} is missing. Run scripts/01_load_clean.py first."
        )

    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]).difference(df.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")

    return df


def build_models():
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            min_samples_leaf=3,
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }


def evaluate_model(y_true, predictions):
    mse = mean_squared_error(y_true, predictions)
    return {
        'mae': round(mean_absolute_error(y_true, predictions), 2),
        'rmse': round(mse ** 0.5, 2),
        'r2': round(r2_score(y_true, predictions), 3),
    }


def save_actual_vs_predicted_plot(y_test, predictions_by_model):
    plt.figure(figsize=(9, 6))

    for model_name, predictions in predictions_by_model.items():
        plt.scatter(
            y_test,
            predictions,
            alpha=0.65,
            s=36,
            label=model_name,
        )

    low = min(y_test.min(), *(pred.min() for pred in predictions_by_model.values()))
    high = max(y_test.max(), *(pred.max() for pred in predictions_by_model.values()))
    plt.plot([low, high], [low, high], color='black', linestyle='--', linewidth=1.3)
    plt.xlabel('Actual Water Usage (Liters)')
    plt.ylabel('Predicted Water Usage (Liters)')
    plt.title('Actual vs Predicted Water Usage')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


def train_all_models():
    df = load_training_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    trained_models = {}
    predictions_by_model = {}
    metric_rows = []

    for model_name, estimator in build_models().items():
        pipeline = Pipeline(steps=[
            ('preprocess', make_preprocessor()),
            ('model', estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, predictions)

        trained_models[model_name] = pipeline
        predictions_by_model[model_name] = predictions
        metric_rows.append({'model': model_name, **metrics})

    metrics_df = pd.DataFrame(metric_rows).sort_values('rmse').reset_index(drop=True)
    best_model_name = metrics_df.iloc[0]['model']

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    save_actual_vs_predicted_plot(y_test, predictions_by_model)

    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump({
            'model_name': best_model_name,
            'pipeline': trained_models[best_model_name],
            'features': FEATURE_COLUMNS,
            'metrics': metrics_df.to_dict(orient='records'),
        }, model_file)

    return trained_models, metrics_df, best_model_name


TRAINED_MODELS, MODEL_METRICS, BEST_MODEL_NAME = train_all_models()


print("\n========== MODEL EVALUATION ==========\n")
print(MODEL_METRICS.to_string(index=False))
print(f"\nBest model: {BEST_MODEL_NAME}")
print(f"Graph saved as: {PLOT_PATH}")
print(f"Metrics saved as: {METRICS_PATH}")


def predict_payload(data):
    prediction_frame, errors = build_prediction_frame(data)
    if errors:
        return None, errors

    predictions = {
        model_name: round(float(model.predict(prediction_frame)[0]), 2)
        for model_name, model in TRAINED_MODELS.items()
    }

    return {
        "input": {
            "household_size": float(prediction_frame.iloc[0]['household_size']),
            "month": int(prediction_frame.iloc[0]['month_num']),
            "neighborhood": prediction_frame.iloc[0]['neighborhood'],
            "season": prediction_frame.iloc[0]['season'],
            "is_summer_peak": bool(prediction_frame.iloc[0]['is_summer_peak']),
        },
        "best_model": BEST_MODEL_NAME,
        "predicted_water_usage": predictions[BEST_MODEL_NAME],
        "predictions_by_model": predictions,
        "metrics": MODEL_METRICS.to_dict(orient='records'),
    }, []


if app is not None:
    @app.route("/")
    def home():
        return jsonify({
            "status": "running",
            "best_model": BEST_MODEL_NAME,
            "predict_endpoint": "/predict",
        })

    @app.route("/model-metrics", methods=["GET"])
    def model_metrics():
        return jsonify({
            "best_model": BEST_MODEL_NAME,
            "metrics": MODEL_METRICS.to_dict(orient='records'),
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        payload, errors = predict_payload(request.get_json() or {})
        if errors:
            return jsonify({"errors": errors}), 400
        return jsonify(payload)


if __name__ == "__main__":
    if app is None:
        raise SystemExit("Install flask and flask-cors to run the prediction API.")
    app.run(debug=True)
