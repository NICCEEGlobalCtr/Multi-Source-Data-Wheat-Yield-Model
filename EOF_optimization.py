import os
import pandas as pd
import numpy as np
import joblib

data_path = 'data/data_range.csv'
model_path = 'model/model_stacking.pkl'
out_dir = 'predictions'
out_filename = 'EOF_optimization.csv'
out_path = os.path.join(out_dir, out_filename)

disk_out_dir = r'D:\output'
disk_out_path = os.path.join(disk_out_dir, out_filename)

price_yield = 0.376
price_n = 0.78
price_p = 0.45
price_k = 0.91

df = pd.read_csv(data_path)

if 'No.' in df.columns:
    df.rename(columns={'No.': 'No'}, inplace=True)

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

total = len(df)
records = []
for idx, row in enumerate(df.itertuples(index=False), start=1):
    print(f"Processing record {idx}/{total}...")

    N_vals = gen_range(row.Nmin, row.Nmax)
    P_vals = gen_range(row.Pmin, row.Pmax)
    K_vals = gen_range(row.Kmin, row.Kmax)

    recs = []
    for n in N_vals:
        for p in P_vals:
            for k in K_vals:
                feat = {f: getattr(row, f) for f in static_feats}
                for c in crop_cols:
                    feat[c] = getattr(row, c)
                feat['N'], feat['P2O5'], feat['K20'] = n, p, k
                recs.append(feat)

    X_try = pd.DataFrame(recs, columns=feature_cols)[feat_order]
    log_pred = model.predict(X_try)
    y_pred = np.expm1(log_pred)

    costs_n = np.array([r['N'] for r in recs]) * price_n
    costs_p = np.array([r['P2O5'] for r in recs]) * price_p
    costs_k = np.array([r['K20'] for r in recs]) * price_k
    revenues = y_pred * price_yield - costs_n - costs_p - costs_k

    top_idx = np.argsort(revenues)[-10:][::-1]
    top_rev = revenues[top_idx]
    top_yield = y_pred[top_idx]
    top_combo = X_try.iloc[top_idx][['N', 'P2O5', 'K20']].reset_index(drop=True)

    out = {
        'ID': getattr(row, 'ID'),
        'No': getattr(row, 'No'),
        'longitude': getattr(row, 'longitude'),
        'latitude': getattr(row, 'latitude'),
    }
    for rnk, rev in enumerate(top_rev, start=1):
        out[f'revenue_{rnk}'] = rev
        out[f'yield_{rnk}'] = top_yield[rnk - 1]
        out[f'N_{rnk}'] = int(top_combo.loc[rnk - 1, 'N'])
        out[f'P_{rnk}'] = int(top_combo.loc[rnk - 1, 'P2O5'])
        out[f'K_{rnk}'] = int(top_combo.loc[rnk - 1, 'K20'])
    records.append(out)

os.makedirs(disk_out_dir, exist_ok=True)
out_df = pd.DataFrame(records)
out_df.to_csv(disk_out_path, index=False)
print(f"Top-10 EOF optimization results saved to: {disk_out_path}")

os.makedirs(out_dir, exist_ok=True)
out_df.to_csv(out_path, index=False)
print(f"Top-10 EOF optimization results saved to: {out_path}")
