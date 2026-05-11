# Municipal Water Usage Analysis

**Team 06 | Applied Data Science Sprint #3**

## Problem Statement

Municipal corporations collect household water consumption data but often lack
visibility into usage inefficiencies. This project analyzes water usage to
identify abnormal consumption patterns, high-usage neighborhoods, seasonal demand
trends, and prediction opportunities.

## Project Structure

```text
water_analysis/
|-- raw_data/               # Original generated dataset
|-- processed_data/         # Cleaned and feature-engineered dataset
|-- scripts/
|   |-- 01_load_clean.py    # Data validation, cleaning, preprocessing
|   |-- 02_eda.py           # EDA and visualizations
|   `-- 03_insights.py      # Insights summary
|-- output/                 # Charts and model comparison output
|-- website/
|   `-- index.html          # Final presentation website
|-- generate_data.py        # Mock data generator
|-- ml.py                   # Model training and Flask prediction API
|-- model.pkl               # Saved best model pipeline
`-- requirements.txt
```

## How To Run

```bash
pip install -r requirements.txt

python generate_data.py
python scripts/01_load_clean.py
python scripts/02_eda.py
python scripts/03_insights.py

# Regenerate the ML graph, saved model, and model comparison CSV
python -c "import ml"

# Start the prediction API
python ml.py
```

Open `website/index.html` in a browser to view the presentation site.

## Key Findings

- Tech Hub has the highest average consumption at about 342 L/month.
- July is the peak month at about 359 L average usage.
- 18 readings are anomalous, representing 3.6% of records.
- Summer usage is about 49.1% higher than winter usage.
- The average usage per person is 124.5 L/month.

## Data Cleaning And Preprocessing

The cleaning script validates required columns, removes duplicates, standardizes
IDs and neighborhood names, parses dates, converts numeric fields safely, fills
missing usage values using neighborhood and household-size medians, and estimates
missing bills from the observed median cost-per-liter rate.

## Feature Engineering

The cleaned dataset includes extra modeling and analysis features:

- `season`
- `is_summer_peak`
- `usage_per_person`
- `billing_rate_usd_per_liter`
- `usage_gap_from_neighborhood_avg`
- `neighborhood_usage_zscore`
- `usage_segment`

## ML Models

The model pipeline trains and compares:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The latest comparison is saved to `output/model_comparison.csv`.

## Actual Vs Predicted Graph

`actual_vs_predicted.png` compares the real water usage values against model
predictions. The dashed diagonal line represents a perfect prediction. Points
close to the line mean the model predicted usage accurately, while points far
above or below the line show overprediction or underprediction.

In the latest run, Linear Regression has the lowest RMSE, so it is saved as the
best model in `model.pkl` and used by the prediction API.

## Prediction API

Start the server:

```bash
python ml.py
```

Send a prediction request:

```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"household_size\": 4, \"month\": 7, \"neighborhood\": \"Tech Hub\"}"
```

The API returns the best model prediction, all model predictions, and the model
comparison metrics.
