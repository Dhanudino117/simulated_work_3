import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Load cleaned data ────────────────────────────────────────────────────────
df = pd.read_csv('processed_data/water_usage_clean.csv', parse_dates=['month'])
os.makedirs('output', exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 150

# ── Chart 1: Monthly Water Usage Trend ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
monthly = df.groupby('month')['usage_liters'].mean().sort_index()
ax.plot(monthly.index, monthly.values, marker='o', linewidth=2, color='#1a6fb5')
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color='#1a6fb5')
ax.set_title('Average Monthly Water Usage Trend', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Avg Usage (Liters)')
fig.autofmt_xdate()
plt.tight_layout()
fig.savefig('output/01_monthly_trend.png')
plt.close(fig)
print("✓ Saved output/01_monthly_trend.png")

# ── Chart 2: Neighborhood Comparison (Bar Chart) ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
neigh = df.groupby('neighborhood')['usage_liters'].mean().sort_values(ascending=True)
colors = ['#27ae60' if v < neigh.mean() else '#e74c3c' for v in neigh.values]
neigh.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Average Water Usage by Neighborhood', fontsize=14, fontweight='bold')
ax.set_xlabel('Avg Usage (Liters)')
ax.set_ylabel('')
plt.tight_layout()
fig.savefig('output/02_neighborhood_usage.png')
plt.close(fig)
print("✓ Saved output/02_neighborhood_usage.png")

# ── Chart 3: Seasonal Usage Distribution (Boxplot) ───────────────────────────
season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
df['season'] = df['month'].dt.month.map(season_map)
season_order = ['Winter', 'Spring', 'Summer', 'Autumn']

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='season', y='usage_liters', hue='season',
            order=season_order, palette='coolwarm', legend=False, ax=ax)
ax.set_title('Water Usage Distribution by Season', fontsize=14, fontweight='bold')
ax.set_xlabel('Season')
ax.set_ylabel('Usage (Liters)')
plt.tight_layout()
fig.savefig('output/03_seasonal_boxplot.png')
plt.close(fig)
print("✓ Saved output/03_seasonal_boxplot.png")

# ── Chart 4: Anomaly Detection Scatter ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
normal = df[df['is_anomaly'] == False]
anomaly = df[df['is_anomaly'] == True]
ax.scatter(normal['month'], normal['usage_liters'], alpha=0.4, s=20,
           color='#1a6fb5', label='Normal')
ax.scatter(anomaly['month'], anomaly['usage_liters'], color='red', s=60,
           marker='X', label='Anomaly', zorder=5)
ax.axhline(y=df['usage_liters'].mean() + 2 * df['usage_liters'].std(),
           color='red', linestyle='--', alpha=0.5, label='Threshold')
ax.set_title('Anomaly Detection — Water Usage', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Usage (Liters)')
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
fig.savefig('output/04_anomalies.png')
plt.close(fig)
print("✓ Saved output/04_anomalies.png")

# ── Chart 5: Neighborhood × Month Heatmap ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
pivot = df.pivot_table(values='usage_per_person', index='neighborhood',
                       columns='month_name', aggfunc='mean')
# Reorder months
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, linecolor='white')
ax.set_title('Per-Person Usage: Neighborhood × Month', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('')
plt.tight_layout()
fig.savefig('output/05_heatmap.png')
plt.close(fig)
print("✓ Saved output/05_heatmap.png")

print("\n✅ All 5 charts saved to output/ folder.")