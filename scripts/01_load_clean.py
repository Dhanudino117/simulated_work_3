import pandas as pd
import os

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('raw_data/water_usage_raw.csv')
print("=== RAW DATA ===")
print(df.info())
print(df.describe())
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# ── Clean ─────────────────────────────────────────────────────────────────────
df_clean = df.copy()

# 1. Drop duplicates
df_clean.drop_duplicates(inplace=True)

# 2. Fill missing usage with neighborhood median
df_clean['water_usage_liters'] = df_clean.groupby('neighborhood')['water_usage_liters'] \
    .transform(lambda x: x.fillna(x.median()))

# 3. Fill missing bill with median
df_clean['monthly_bill_usd'] = df_clean['monthly_bill_usd'].fillna(df_clean['monthly_bill_usd'].median())

# 4. Fix data types
df_clean['month'] = pd.to_datetime(df_clean['month'])
df_clean['household_size'] = df_clean['household_size'].astype(int)

# 5. Rename for clarity
df_clean.rename(columns={
    'water_usage_liters': 'usage_liters',
    'monthly_bill_usd':   'bill_usd'
}, inplace=True)

# 6. Add derived columns
df_clean['usage_per_person'] = (df_clean['usage_liters'] / df_clean['household_size']).round(1)
df_clean['month_name'] = df_clean['month'].dt.strftime('%b')
df_clean['month_num']  = df_clean['month'].dt.month

# 7. Flag anomalies (usage > mean + 2*std)
threshold = df_clean['usage_liters'].mean() + 2 * df_clean['usage_liters'].std()
df_clean['is_anomaly'] = df_clean['usage_liters'] > threshold

print("\n=== CLEANED DATA ===")
print(df_clean.info())
print(f"\nAnomalies detected: {df_clean['is_anomaly'].sum()}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('processed_data', exist_ok=True)
df_clean.to_csv('processed_data/water_usage_clean.csv', index=False)
print("\nSaved → processed_data/water_usage_clean.csv")