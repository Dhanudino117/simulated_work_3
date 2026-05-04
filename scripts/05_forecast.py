import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -- Load cleaned data --------------------------------------------------------
df = pd.read_csv('processed_data/water_usage_clean.csv', parse_dates=['month'])
os.makedirs('output', exist_ok=True)

# -- Style ---------------------------------------------------------------------
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 150

# -- Aggregate to neighborhood-month level ------------------------------------
neigh_monthly = (
    df.groupby(['neighborhood', 'month'])
    .agg(
        usage_liters=('usage_liters', 'mean'),
        household_size=('household_size', 'mean'),
    )
    .reset_index()
    .sort_values(['neighborhood', 'month'])
)

# -- Season encoding ----------------------------------------------------------
season_map = {12: 0, 1: 0, 2: 0,    # Winter = 0
              3: 1, 4: 1, 5: 1,      # Spring = 1
              6: 2, 7: 2, 8: 2,      # Summer = 2
              9: 3, 10: 3, 11: 3}    # Autumn = 3
neigh_monthly['season_int'] = neigh_monthly['month'].dt.month.map(season_map)
neigh_monthly['month_num'] = neigh_monthly['month'].dt.month

# -- Lag Features -------------------------------------------------------------
neigh_monthly['prev_month_usage'] = (
    neigh_monthly.groupby('neighborhood')['usage_liters']
    .shift(1)
)
neigh_monthly['rolling_3m_mean'] = (
    neigh_monthly.groupby('neighborhood')['usage_liters']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)

# Drop rows with NaN from lag creation
neigh_monthly = neigh_monthly.dropna().reset_index(drop=True)

print("-- Feature Engineering Complete -------------------------------------")
print(f"   Records after lag creation: {len(neigh_monthly)}")
print(f"   Features: prev_month_usage, rolling_3m_mean, season_int, month_num, household_size")

# -- Train/Test Split: last 3 months as test ----------------------------------
all_months = sorted(neigh_monthly['month'].unique())
test_months = all_months[-3:]
train_months = all_months[:-3]

train = neigh_monthly[neigh_monthly['month'].isin(train_months)]
test = neigh_monthly[neigh_monthly['month'].isin(test_months)]

feature_cols = ['prev_month_usage', 'rolling_3m_mean', 'season_int', 'month_num', 'household_size']
X_train = train[feature_cols]
y_train = train['usage_liters']
X_test = test[feature_cols]
y_test = test['usage_liters']

print(f"   Train set: {len(X_train)} rows ({len(train_months)} months)")
print(f"   Test set:  {len(X_test)} rows ({len(test_months)} months)")

# -- Train Random Forest ------------------------------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# -- Evaluate -----------------------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n-- Forecast Evaluation --------------------------------------------")
print(f"   RMSE: {rmse:.2f} liters")
print(f"   R² Score: {r2:.4f}")

# -- Chart 1: Actual vs Predicted ---------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

# Prepare test data for plotting
test_plot = test.copy()
test_plot['predicted'] = y_pred
test_agg = test_plot.groupby('month').agg(
    actual=('usage_liters', 'mean'),
    predicted=('predicted', 'mean')
).sort_index()

# Also plot training data for context
train_agg = train.groupby('month')['usage_liters'].mean().sort_index()

ax.plot(train_agg.index, train_agg.values, marker='o', linewidth=2,
        color='#1a6fb5', label='Training (Actual)', markersize=5)
ax.plot(test_agg.index, test_agg['actual'], marker='o', linewidth=2,
        color='#27ae60', label='Test (Actual)', markersize=7)
ax.plot(test_agg.index, test_agg['predicted'], marker='s', linewidth=2,
        color='#e74c3c', linestyle='--', label='Test (Predicted)', markersize=7)

# Shade test region
ax.axvspan(test_agg.index.min(), test_agg.index.max(),
           alpha=0.08, color='#e74c3c', label='Test Period')

ax.set_title('Demand Forecast — Actual vs Predicted', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Avg Usage (Liters)')
ax.legend(framealpha=0.9, loc='best')
fig.autofmt_xdate()
plt.tight_layout()
fig.savefig('output/07_forecast.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\n[OK] Saved output/07_forecast.png")

# -- Chart 2: Feature Importance ----------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
colors = ['#1a6fb5' if v < importances.mean() else '#06d6a0' for v in importances.values]

importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.set_ylabel('')

# Add value labels
for i, (val, name) in enumerate(zip(importances.values, importances.index)):
    ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9, fontweight='500')

plt.tight_layout()
fig.savefig('output/08_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[OK] Saved output/08_feature_importance.png")

print(f"\n[DONE] Demand forecasting complete. RMSE={rmse:.2f}, R²={r2:.4f}")
