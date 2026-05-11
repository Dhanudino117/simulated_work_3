import pandas as pd

df = pd.read_csv('processed_data/water_usage_clean.csv', parse_dates=['month'])

print("=" * 55)
print("        WATER CONSUMPTION INSIGHTS SUMMARY")
print("=" * 55)

# 1. Highest usage neighborhood
top_neigh = df.groupby('neighborhood')['usage_liters'].mean().idxmax()
top_val   = df.groupby('neighborhood')['usage_liters'].mean().max()
print(f"\n1. Highest usage neighborhood : {top_neigh} ({top_val:.0f} L avg)")

# 2. Peak month
df['month_label'] = df['month'].dt.strftime('%B')
peak_month = df.groupby('month_label')['usage_liters'].mean().idxmax()
peak_val   = df.groupby('month_label')['usage_liters'].mean().max()
print(f"2. Peak consumption month     : {peak_month} ({peak_val:.0f} L avg)")

# 3. Anomalies
n_anomaly = df['is_anomaly'].sum()
pct = (n_anomaly / len(df)) * 100
print(f"3. Anomalous readings         : {n_anomaly} ({pct:.1f}% of records)")

# 4. Summer vs Winter
if 'season' not in df.columns:
    season_map = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
                  6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
    df['season'] = df['month'].dt.month.map(season_map)
summer = df[df['season']=='Summer']['usage_liters'].mean()
winter = df[df['season']=='Winter']['usage_liters'].mean()
print(f"4. Summer vs Winter avg usage : {summer:.0f} L vs {winter:.0f} L (+{((summer-winter)/winter*100):.1f}% in summer)")

# 5. Avg usage per person
avg_per_person = df['usage_per_person'].mean()
print(f"5. Avg usage per person/month : {avg_per_person:.1f} L")

# 6. Highest usage segment
if 'usage_segment' in df.columns:
    top_segment = df['usage_segment'].value_counts().idxmax()
    top_segment_count = df['usage_segment'].value_counts().max()
    print(f"6. Most common usage segment   : {top_segment} ({top_segment_count} records)")

print("\n" + "=" * 55)
print("CONSERVATION RECOMMENDATIONS")
print("=" * 55)
print(f"  - Focus leak audits on {top_neigh} (highest usage).")
print(f"  - Launch conservation campaigns before {peak_month}.")
print(f"  - Investigate {n_anomaly} flagged anomaly readings.")
print(f"  - Introduce tiered pricing for summer months.")
if 'usage_segment' in df.columns:
    print("  - Use usage segments to personalize conservation messaging.")
print("=" * 55)
