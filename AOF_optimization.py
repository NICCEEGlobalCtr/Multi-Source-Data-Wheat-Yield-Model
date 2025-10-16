import os
import pandas as pd
import numpy as np
import joblib

data_path = 'data/data_range.csv'
model_path = 'model/model_stacking.pkl'
out_dir = 'predictions'
out_path = os.path.join(out_dir, 'AOF_optimization.csv')

df = pd.read_csv(data_path)

if 'crop_type' in df.columns:
    df = pd.get_dummies(df, columns=['crop_type'])
crop_cols = [c for c in df.columns if c.startswith('crop_type_')]

static_feats = [
    'Altitude', 'SOM', 'TN', 'AN', 'AP', 'AK', 'pH',
    'irrigation', 'precipitation', 'Frost_free_day',
    'temperature', 'Nsurplus'
]
feature_cols = static_feats + ['N', 'P2O5', 'K20'] + crop_cols

df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in static_feats:
    if df[col].dtype.kind in 'biufc':
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)
for col in crop_cols:
    df[col].fillna(0, inplace=True)

model = joblib.load(model_path)
feat_order = model.get_booster().feature_names

def gen_range(vmin, vmax):
    vals = list(np.arange(vmin, vmax + 1, 5))
    if vals[-1] != vmax:
        vals.append(vmax)
    return vals

results = []
total = len(df)
for count, (_, row) in enumerate(df.iterrows(), start=1):
    print(f"Processing record {count}/{total}...")

    N_vals = gen_range(row['Nmin'], row['Nmax'])
    P_vals = gen_range(row['Pmin'], row['Pmax'])
    K_vals = gen_range(row['Kmin'], row['Kmax'])

    recs = []
    for n in N_vals:
        for p in P_vals:
            for k in K_vals:
                r = {f: row[f] for f in static_feats}
                for c in crop_cols:
                    r[c] = row[c]
                r['N'], r['P2O5'], r['K20'] = n, p, k
                recs.append(r)

    X_try = pd.DataFrame(recs, columns=feature_cols)
    X_try = X_try[feat_order]

    log_pred = model.predict(X_try)
    y_pred = np.expm1(log_pred)

    idx_max = np.argmax(y_pred)
    best = X_try.iloc[idx_max]

    results.append({
        'ID':                  row['ID'],
        'No.':                 row['No.'],
        'longitude':           row['longitude'],
        'latitude':            row['latitude'],
        'predicted_max_yield': y_pred[idx_max],
        'N_optimized':         best['N'],
        'P_optimized':         best['P2O5'],
        'K_optimized':         best['K20'],
    })

os.makedirs(out_dir, exist_ok=True)
out_df = pd.DataFrame(results)
out_df.to_csv(out_path, index=False)
print(f"AOF optimization results saved to: {out_path}")

disk_out_dir = r'D:\output'
disk_out_path = os.path.join(disk_out_dir, 'AOF_optimization.csv')
os.makedirs(disk_out_dir, exist_ok=True)
out_df.to_csv(disk_out_path, index=False)
print(f"Top-1 AOF optimization results also saved to: {disk_out_path}")
