import pandas as pd
import json

df = pd.read_csv('processed_data/water_usage_clean.csv', parse_dates=['month'])

# 1. Monthly trend
monthly = df.groupby(df['month'].dt.strftime('%Y-%m'))['usage_liters'].mean().sort_index()
print("MONTHLY:" + json.dumps({k: round(v,1) for k,v in monthly.items()}))

# 2. Neighborhood averages
neigh = df.groupby('neighborhood')['usage_liters'].mean().sort_values()
print("NEIGH:" + json.dumps({k: round(v,1) for k,v in neigh.items()}))

# 3. Seasonal boxplot stats
season_map = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
              6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
df['season'] = df['month'].dt.month.map(season_map)

seasonal = {}
for s in ['Winter','Spring','Summer','Autumn']:
    d = df[df['season']==s]['usage_liters']
    seasonal[s] = {
        'min': round(d.min(),1), 'q1': round(d.quantile(0.25),1),
        'median': round(d.median(),1), 'q3': round(d.quantile(0.75),1),
        'max': round(d.max(),1), 'mean': round(d.mean(),1)
    }
print("SEASONAL:" + json.dumps(seasonal))

# 4. Anomalies
threshold = round(df['usage_liters'].mean() + 2 * df['usage_liters'].std(), 1)
print("THRESHOLD:" + str(threshold))

anomalies = df[df['is_anomaly']==True][['month','usage_liters','neighborhood']].copy()
anomalies['month'] = anomalies['month'].dt.strftime('%Y-%m')
print("ANOMALIES:" + anomalies.to_json(orient='records'))
print("ANOM_COUNT:" + str(int(df['is_anomaly'].sum())))

# 5. All data points for scatter
scatter = df[['month','usage_liters','is_anomaly']].copy()
scatter['month'] = scatter['month'].dt.strftime('%Y-%m')
# Too many points - sample normal ones
normal = scatter[scatter['is_anomaly']==False].to_dict('records')
anom = scatter[scatter['is_anomaly']==True].to_dict('records')
print("SCATTER_NORMAL:" + json.dumps([{'m':r['month'],'v':round(r['usage_liters'],1)} for r in normal]))
print("SCATTER_ANOM:" + json.dumps([{'m':r['month'],'v':round(r['usage_liters'],1)} for r in anom]))

# 6. Heatmap
pivot = df.pivot_table(values='usage_per_person', index='neighborhood', columns='month_name', aggfunc='mean')
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
print("HEATMAP:" + pivot.round(1).to_json())

# 7. Stats
print("TOTAL_RECORDS:" + str(len(df)))
print("NUM_NEIGHBORHOODS:" + str(df['neighborhood'].nunique()))

# Summer vs winter
summer_avg = round(df[df['season']=='Summer']['usage_liters'].mean(), 1)
winter_avg = round(df[df['season']=='Winter']['usage_liters'].mean(), 1)
pct_diff = round((summer_avg - winter_avg) / winter_avg * 100, 1)
print("SUMMER_AVG:" + str(summer_avg))
print("WINTER_AVG:" + str(winter_avg))
print("SUMMER_WINTER_PCT:" + str(pct_diff))

avg_per_person = round(df['usage_per_person'].mean(), 1)
print("AVG_PER_PERSON:" + str(avg_per_person))
