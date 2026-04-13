import pandas as pd
import numpy as np
import os

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('raw_data/water_usage_raw.csv')
print("=== RAW DATA ===")
print(df.info())
print(df.describe())
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# ── Advanced Clean ────────────────────────────────────────────────────────────
df_clean = df.copy()

# 1. Drop duplicates
df_clean.drop_duplicates(inplace=True)

# 2. Advanced Imputation: Fill missing usage leveraging both neighborhood AND household size
df_clean['water_usage_liters'] = df_clean.groupby(['neighborhood', 'household_size'])['water_usage_liters'] \
    .transform(lambda x: x.fillna(x.median()))
# Fallback in case of remaining NaNs
df_clean['water_usage_liters'] = df_clean.groupby('neighborhood')['water_usage_liters'] \
    .transform(lambda x: x.fillna(x.median()))

# 3. Dynamic Bill Imputation: Base the missing bill on the derived cost-per-liter rate instead of just median
median_rate = (df_clean['monthly_bill_usd'] / df_clean['water_usage_liters']).median()
df_clean['monthly_bill_usd'] = df_clean['monthly_bill_usd'].fillna(df_clean['water_usage_liters'] * median_rate)

# 4. Memory Optimization and Typed Casting
df_clean['month'] = pd.to_datetime(df_clean['month'])
df_clean['household_size'] = pd.to_numeric(df_clean['household_size'], downcast='integer')
df_clean['neighborhood'] = df_clean['neighborhood'].astype('category')

# 5. Rename for clarity
df_clean.rename(columns={
    'water_usage_liters': 'usage_liters',
    'monthly_bill_usd':   'bill_usd'
}, inplace=True)

# 6. Add derived columns using chained .assign method for cleaner flow
df_clean = df_clean.assign(
    usage_per_person = lambda x: (x['usage_liters'] / x['household_size']).round(2),
    month_name = lambda x: x['month'].dt.strftime('%b').astype('category'),
    month_num = lambda x: x['month'].dt.month.astype('int8')
)

# 7. Robust Anomaly Detection: Localized IQR (Interquartile Range) Method per neighborhood
def flag_outliers_iqr(group):
    # Calculate IQR and dynamic boundaries per group to isolate extreme outliers
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (group < lower_bound) | (group > upper_bound)

df_clean['is_anomaly'] = df_clean.groupby('neighborhood')['usage_liters'].transform(flag_outliers_iqr)

print("\n=== ADVANCED CLEANED DATA ===")
print(df_clean.info())
print(f"\nAnomalies detected (IQR Localized): {df_clean['is_anomaly'].sum()}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('processed_data', exist_ok=True)
df_clean.to_csv('processed_data/water_usage_clean.csv', index=False)
print("\nSaved -> processed_data/water_usage_clean.csv")