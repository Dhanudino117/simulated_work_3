import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -- Load cleaned data --------------------------------------------------------
df = pd.read_csv('processed_data/water_usage_clean.csv', parse_dates=['month'])
os.makedirs('output', exist_ok=True)

# -- Style ---------------------------------------------------------------------
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 150

# -- Aggregate per-neighborhood features --------------------------------------
# Season encoding for variance calculation
season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
df['season'] = df['month'].dt.month.map(season_map)

# Compute seasonal variance per neighborhood
seasonal_usage = df.groupby(['neighborhood', 'season'])['usage_liters'].mean().reset_index()
seasonal_var = seasonal_usage.groupby('neighborhood')['usage_liters'].var().reset_index()
seasonal_var.columns = ['neighborhood', 'seasonal_variance']

# Aggregate neighborhood-level features
neigh_agg = df.groupby('neighborhood').agg(
    mean_usage=('usage_liters', 'mean'),
    anomaly_rate=('is_anomaly', 'mean'),
    avg_household_size=('household_size', 'mean'),
).reset_index()

neigh_agg = neigh_agg.merge(seasonal_var, on='neighborhood')

print("-- Neighborhood Feature Summary -----------------------------------")
print(neigh_agg.to_string(index=False))

# -- Standardize features ----------------------------------------------------
feature_cols = ['mean_usage', 'seasonal_variance', 'anomaly_rate', 'avg_household_size']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(neigh_agg[feature_cols])

# -- Elbow Method + Silhouette Score ------------------------------------------
print(f"\n-- Elbow / Silhouette Analysis ------------------------------------")
max_k = min(len(neigh_agg) - 1, 6)
for k in range(2, max_k + 1):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    sil = silhouette_score(X_scaled, labels)
    print(f"   k={k}  |  Inertia: {inertia:.2f}  |  Silhouette: {sil:.3f}")

# -- Run KMeans with k=3 -----------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
neigh_agg['cluster'] = kmeans.fit_predict(X_scaled)

# -- Label clusters semantically ----------------------------------------------
# Assign labels based on mean_usage ranking within clusters
cluster_means = neigh_agg.groupby('cluster')['mean_usage'].mean().sort_values()
label_map = {}
cluster_order = cluster_means.index.tolist()
label_map[cluster_order[0]] = 'Efficient'
label_map[cluster_order[1]] = 'Moderate usage'
label_map[cluster_order[2]] = 'High usage'

neigh_agg['cluster_label'] = neigh_agg['cluster'].map(label_map)

print(f"\n-- Cluster Assignments --------------------------------------------")
print(neigh_agg[['neighborhood', 'mean_usage', 'seasonal_variance',
                  'anomaly_rate', 'avg_household_size', 'cluster_label']].to_string(index=False))

# -- Cluster Summary Table ----------------------------------------------------
summary = neigh_agg.groupby('cluster_label').agg(
    neighborhoods=('neighborhood', lambda x: ', '.join(x)),
    avg_usage=('mean_usage', 'mean'),
    avg_seasonal_var=('seasonal_variance', 'mean'),
    avg_anomaly_rate=('anomaly_rate', 'mean'),
    avg_hh_size=('avg_household_size', 'mean'),
    count=('neighborhood', 'count')
).reset_index()

print(f"\n-- Cluster Summary ------------------------------------------------")
print(summary.to_string(index=False))

# -- PCA for 2D visualization ------------------------------------------------
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
neigh_agg['pca_1'] = coords[:, 0]
neigh_agg['pca_2'] = coords[:, 1]

explained = pca.explained_variance_ratio_
print(f"\n   PCA Explained Variance: PC1={explained[0]:.2%}, PC2={explained[1]:.2%}")

# -- Chart: 2D PCA Scatter colored by cluster ---------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

color_map = {'High usage': '#e74c3c', 'Moderate usage': '#f59e0b', 'Efficient': '#27ae60'}
marker_map = {'High usage': 'D', 'Moderate usage': 's', 'Efficient': 'o'}

for label in ['Efficient', 'Moderate usage', 'High usage']:
    mask = neigh_agg['cluster_label'] == label
    subset = neigh_agg[mask]
    ax.scatter(
        subset['pca_1'], subset['pca_2'],
        c=color_map[label], marker=marker_map[label],
        s=200, label=label, edgecolors='white', linewidths=1.5, zorder=5
    )
    # Add neighborhood name labels
    for _, row in subset.iterrows():
        ax.annotate(
            row['neighborhood'], (row['pca_1'], row['pca_2']),
            textcoords='offset points', xytext=(10, 8),
            fontsize=9, fontweight='600', color=color_map[label],
            arrowprops=dict(arrowstyle='-', color=color_map[label], alpha=0.4)
        )

ax.set_title('Neighborhood Clusters — PCA Projection', fontsize=14, fontweight='bold')
ax.set_xlabel(f'Principal Component 1 ({explained[0]:.1%} variance)')
ax.set_ylabel(f'Principal Component 2 ({explained[1]:.1%} variance)')
ax.legend(framealpha=0.9, loc='best', title='Cluster')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('output/09_clusters.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\n[OK] Saved output/09_clusters.png")

print("\n[DONE] Neighborhood clustering complete.")
