import os

import pandas as pd


REQUIRED_COLUMNS = {
    'household_id',
    'neighborhood',
    'month',
    'water_usage_liters',
    'household_size',
    'monthly_bill_usd',
    'meter_reading',
}


def flag_outliers_iqr(group):
    q1 = group.quantile(0.25)
    q3 = group.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (group < lower_bound) | (group > upper_bound)


# Load
df = pd.read_csv('raw_data/water_usage_raw.csv')
missing_columns = REQUIRED_COLUMNS.difference(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

print("=== RAW DATA ===")
print(df.info())
print(df.describe())
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# Clean and preprocess
df_clean = df.copy().drop_duplicates()

df_clean['household_id'] = df_clean['household_id'].astype(str).str.strip().str.upper()
df_clean['neighborhood'] = df_clean['neighborhood'].astype(str).str.strip().str.title()
df_clean['month'] = pd.to_datetime(df_clean['month'], errors='coerce')

numeric_columns = [
    'water_usage_liters',
    'household_size',
    'monthly_bill_usd',
    'meter_reading',
]
for column in numeric_columns:
    df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')

df_clean = df_clean.dropna(
    subset=['household_id', 'neighborhood', 'month', 'household_size', 'meter_reading']
)

df_clean['household_size'] = df_clean['household_size'].round().clip(lower=1).astype('int8')
df_clean['meter_reading'] = df_clean['meter_reading'].round().astype('int32')
df_clean['water_usage_liters'] = df_clean['water_usage_liters'].clip(lower=0)
df_clean['monthly_bill_usd'] = df_clean['monthly_bill_usd'].clip(lower=0)

segment_usage_median = df_clean.groupby(
    ['neighborhood', 'household_size']
)['water_usage_liters'].transform('median')
neighborhood_usage_median = df_clean.groupby(
    'neighborhood'
)['water_usage_liters'].transform('median')
global_usage_median = df_clean['water_usage_liters'].median()

df_clean['water_usage_liters'] = (
    df_clean['water_usage_liters']
    .fillna(segment_usage_median)
    .fillna(neighborhood_usage_median)
    .fillna(global_usage_median)
)

valid_rates = df_clean.loc[
    (df_clean['water_usage_liters'] > 0) & df_clean['monthly_bill_usd'].notna(),
    'monthly_bill_usd',
] / df_clean.loc[
    (df_clean['water_usage_liters'] > 0) & df_clean['monthly_bill_usd'].notna(),
    'water_usage_liters',
]
median_rate = valid_rates.median()
df_clean['monthly_bill_usd'] = (
    df_clean['monthly_bill_usd']
    .fillna(df_clean['water_usage_liters'] * median_rate)
    .round(2)
)

df_clean.rename(columns={
    'water_usage_liters': 'usage_liters',
    'monthly_bill_usd': 'bill_usd',
}, inplace=True)

df_clean = df_clean.assign(
    usage_per_person=lambda x: (x['usage_liters'] / x['household_size']).round(2),
    month_name=lambda x: x['month'].dt.strftime('%b').astype('category'),
    month_num=lambda x: x['month'].dt.month.astype('int8'),
)

season_map = {
    12: 'Winter',
    1: 'Winter',
    2: 'Winter',
    3: 'Spring',
    4: 'Spring',
    5: 'Spring',
    6: 'Summer',
    7: 'Summer',
    8: 'Summer',
    9: 'Autumn',
    10: 'Autumn',
    11: 'Autumn',
}
neighborhood_avg_usage = df_clean.groupby('neighborhood')['usage_liters'].transform('mean')
neighborhood_std_usage = (
    df_clean.groupby('neighborhood')['usage_liters'].transform('std').replace(0, pd.NA)
)

df_clean = df_clean.assign(
    season=lambda x: x['month_num'].map(season_map).astype('category'),
    is_summer_peak=lambda x: x['month_num'].isin([6, 7, 8]),
    billing_rate_usd_per_liter=lambda x: (x['bill_usd'] / x['usage_liters']).round(4),
    usage_gap_from_neighborhood_avg=lambda x: (
        x['usage_liters'] - neighborhood_avg_usage
    ).round(2),
    neighborhood_usage_zscore=lambda x: (
        (x['usage_liters'] - neighborhood_avg_usage) / neighborhood_std_usage
    ).fillna(0).round(2),
)
df_clean['usage_segment'] = pd.cut(
    df_clean['usage_per_person'],
    bins=[float('-inf'), 75, 150, float('inf')],
    labels=['efficient', 'typical', 'high'],
).astype('category')

df_clean['is_anomaly'] = df_clean.groupby(
    'neighborhood'
)['usage_liters'].transform(flag_outliers_iqr)

df_clean['neighborhood'] = df_clean['neighborhood'].astype('category')
df_clean = df_clean.sort_values(['month', 'neighborhood', 'household_id']).reset_index(drop=True)

print("\n=== CLEANED DATA ===")
print(df_clean.info())
print(f"\nAnomalies detected (localized IQR): {df_clean['is_anomaly'].sum()}")

# Save
os.makedirs('processed_data', exist_ok=True)
df_clean.to_csv('processed_data/water_usage_clean.csv', index=False)
print("\nSaved -> processed_data/water_usage_clean.csv")
