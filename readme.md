# 💧 Municipal Water Usage Analysis
**Team 06 | Applied Data Science Sprint #3**

## Problem Statement
Municipal corporations collect household water consumption data but lack visibility into usage inefficiencies. This project analyzes water usage to identify abnormal consumption patterns, high-usage neighborhoods, and seasonal demand trends.

## Project Structure
```
water_analysis/
├── raw_data/               # Original generated dataset
├── processed_data/         # Cleaned dataset
├── scripts/
│   ├── 01_load_clean.py    # Data loading & cleaning
│   ├── 02_eda.py           # EDA & visualizations
│   ├── 03_insights.py      # Insights summary
│   ├── 04_models.py        # Phase 2 — Isolation Forest anomaly detection
│   ├── 05_forecast.py      # Phase 2 — Random Forest demand forecasting
│   └── 06_cluster.py       # Phase 2 — KMeans neighborhood clustering
├── output/                 # All charts (PNG)
├── website/
│   └── index.html          # Final presentation website
├── generate_data.py        # Mock data generator
└── README.md
```

## How to Run (VS Code Terminal)

```bash
# 1. Install dependencies
pip install pandas matplotlib seaborn scikit-learn

# 2. Generate mock data
python generate_data.py

# 3. Clean the data
python scripts/01_load_clean.py

# 4. Generate EDA charts
python scripts/02_eda.py

# 5. Print insights
python scripts/03_insights.py

# 6. Phase 2 — ML scripts
python scripts/04_models.py
python scripts/05_forecast.py
python scripts/06_cluster.py

# 7. Open website — just open website/index.html in your browser
```

## Key Findings
- **Tech Hub** has the highest average water consumption (~300 L/month)
- **Summer** demand is ~40% higher than winter
- **~3% anomalies** detected — likely leaks or meter faults
- Per-person normalization reveals hidden inefficiencies in smaller households

## Tools Used
- Python 3.x, Pandas, Matplotlib, Seaborn, scikit-learn
- VS Code
- HTML/CSS for website