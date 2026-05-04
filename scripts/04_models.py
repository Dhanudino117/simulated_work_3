import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_score, recall_score

# ── Load cleaned data ────────────────────────────────────────────────────────
df = pd.read_csv('processed_data/water_usage_clean.csv', parse_dates=['month'])
os.makedirs('output', exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 150

# ── Feature Engineering ──────────────────────────────────────────────────────
# Sort for rolling calculations
df = df.sort_values(['household_id', 'month']).reset_index(drop=True)

# 1. Rolling 3-month average per household
df['rolling_3m_avg'] = (
    df.groupby('household_id')['usage_liters']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)

# 2. Deviation from neighborhood mean
neigh_mean = df.groupby('neighborhood')['usage_liters'].transform('mean')
df['deviation_from_neigh'] = df['usage_liters'] - neigh_mean

# 3. Usage per person ratio (already exists, but recalculate for consistency)
df['usage_per_person_ratio'] = df['usage_liters'] / df['household_size']

print("-- Feature Engineering Complete ------------------------------------")
print(f"   Features: rolling_3m_avg, deviation_from_neigh, usage_per_person_ratio")
print(f"   Total records: {len(df)}")

# -- Train Isolation Forest ------------------------------------------------
features = ['usage_liters', 'rolling_3m_avg', 'deviation_from_neigh', 'usage_per_person_ratio']
X = df[features].fillna(0)

iso_forest = IsolationForest(
    contamination=0.03,
    random_state=42,
    n_estimators=100
)
df['ml_anomaly'] = iso_forest.fit_predict(X)
# IsolationForest returns -1 for anomalies, 1 for normal -> convert to boolean
df['ml_anomaly'] = df['ml_anomaly'] == -1

print(f"\n-- Isolation Forest Results ----------------------------------------")
print(f"   ML anomalies detected: {df['ml_anomaly'].sum()}")
print(f"   Original anomalies:    {df['is_anomaly'].sum()}")

# -- Evaluate against existing anomaly flag --------------------------------
y_true = df['is_anomaly'].astype(int)
y_pred = df['ml_anomaly'].astype(int)

print(f"\n-- Classification Report -------------------------------------------")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print(f"   Precision: {precision:.3f}")
print(f"   Recall:    {recall:.3f}")

# ── Chart: Scatter plot of usage vs usage-per-person ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

normal_mask = ~df['ml_anomaly']
anomaly_mask = df['ml_anomaly']

ax.scatter(
    df.loc[normal_mask, 'usage_liters'],
    df.loc[normal_mask, 'usage_per_person_ratio'],
    alpha=0.4, s=25, color='#1a6fb5', label='Normal', zorder=2
)
ax.scatter(
    df.loc[anomaly_mask, 'usage_liters'],
    df.loc[anomaly_mask, 'usage_per_person_ratio'],
    color='red', s=70, marker='X', label='ML Anomaly',
    edgecolors='darkred', linewidths=0.5, zorder=5
)

ax.set_title('Isolation Forest — Anomaly Detection', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Usage (Liters)')
ax.set_ylabel('Usage per Person (Liters)')
ax.legend(framealpha=0.9, loc='upper right')

plt.tight_layout()
fig.savefig('output/06_anomaly_ml.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\n[OK] Saved output/06_anomaly_ml.png")

print("\n[DONE] Anomaly detection complete.")
