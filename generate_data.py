import pandas as pd
import numpy as np
import os

np.random.seed(42)

neighborhoods = ['Green Park', 'River Side', 'Old Town', 'Tech Hub', 'Market Square']
n_records = 500

months = pd.date_range(start='2023-01-01', periods=12, freq='MS')
rows = []

for _ in range(n_records):
    neighborhood = np.random.choice(neighborhoods)
    month = pd.Timestamp(np.random.choice(months))

    # Seasonal factor: summer (Jun-Aug) uses more water
    season_factor = 1.4 if month.month in [6, 7, 8] else (0.85 if month.month in [12, 1, 2] else 1.0)

    # Neighborhood base usage
    base = {'Green Park': 180, 'River Side': 220, 'Old Town': 160,
            'Tech Hub': 300, 'Market Square': 250}[neighborhood]

    usage = round(np.random.normal(base * season_factor, 30), 1)
    usage = max(50, usage)  # no negative usage

    # Inject some anomalies
    if np.random.rand() < 0.03:
        usage = round(usage * np.random.uniform(2.5, 4.0), 1)

    household_size = np.random.randint(1, 6)
    bill = round(usage * 0.05 + np.random.normal(0, 2), 2)

    rows.append({
        'household_id': f'HH{np.random.randint(1000, 9999)}',
        'neighborhood': neighborhood,
        'month': month.strftime('%Y-%m'),
        'water_usage_liters': usage,
        'household_size': household_size,
        'monthly_bill_usd': max(0, bill),
        'meter_reading': np.random.randint(1000, 9999)
    })

df = pd.DataFrame(rows)

# Introduce missing values
df.loc[df.sample(frac=0.02).index, 'water_usage_liters'] = np.nan
df.loc[df.sample(frac=0.01).index, 'monthly_bill_usd'] = np.nan

os.makedirs('raw_data', exist_ok=True)
df.to_csv('raw_data/water_usage_raw.csv', index=False)
print(f"Generated {len(df)} records -> raw_data/water_usage_raw.csv")
print(df.head())