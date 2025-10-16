import os
import pandas as pd
import numpy as np
import joblib

data_path = 'data/data_range.csv'
model_path = 'model/model_stacking.pkl'
out_dir = 'predictions'
out_filename = 'ESOF_optimization.csv'
out_path = os.path.join(out_dir, out_filename)

disk_out_dir = r'D:\output'
disk_out_path = os.path.join(disk_out_dir, out_filename)

price_grain = 0.376
price_n = 0.78
price_p = 0.45
price_k = 0.91

df = pd.read_csv(data_path)

if 'No.' in df.columns:
    df.rename(columns={'No.': 'No'}, inplace=True)

if 'crop_type' in df.columns:
    df = pd.get_dummies(df, columns=['crop_type'])
crop_cols = [col for col in df.columns if col.startswith('crop_type_')]

static_feats = [
    'Altitude', 'SOM', 'TN', 'AN', 'AP', 'AK', 'pH',
    'irrigation', 'precipitation', 'Frost_free_day',
    'temperature', 'Nsurplus', 'Ndep', 'Region'
]
feature_cols = static_feats + ['N', 'P2O5', 'K20'] + crop_cols

df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in static_feats:
    if col in df.columns and col != 'Region':
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
for col in crop_cols:
    df[col].fillna(0, inplace=True)

model = joblib.load(model_path)
try:
    feat_order = model.get_booster().feature_names
except Exception:
    feat_order = feature_cols

def gen_range(vmin, vmax):
    values = list(np.arange(int(vmin), int(vmax) + 1, 5))
    if values[-1] != int(vmax):
        values.append(int(vmax))
    return values

records = []
total_records = len(df)
for idx, row in enumerate(df.itertuples(index=False), start=1):
    print(f"Processing record {idx}/{total_records}...")

    N_vals = gen_range(row.Nmin, row.Nmax)
    P_vals = gen_range(row.Pmin, row.Pmax)
    K_vals = gen_range(row.Kmin, row.Kmax)

    combo_list = []
    for n in N_vals:
        for p in P_vals:
            for k in K_vals:
                feat = {feat: getattr(row, feat) for feat in static_feats}
                for c in crop_cols:
                    feat[c] = getattr(row, c)
                feat['N'] = n
                feat['P2O5'] = p
                feat['K20'] = k
                combo_list.append(feat)

    X_try = pd.DataFrame(combo_list, columns=feature_cols)[feat_order]
    log_pred = model.predict(X_try)
    y_pred = np.expm1(log_pred)

    N_arr = np.array([item['N'] for item in combo_list], dtype=float)
    P_arr = np.array([item['P2O5'] for item in combo_list], dtype=float)
    K_arr = np.array([item['K20'] for item in combo_list], dtype=float)

    region = getattr(row, 'Region')
    if region == 'North':
        N_env_cost = (2.99 * np.exp(0.0045 * N_arr)
                      + 6.9331 * np.exp(0.0057 * N_arr)
                      + 0.5985 * N_arr
                      + 17.73525)
    elif region == 'Center':
        N_env_cost = (5.75 * np.exp(0.0032 * N_arr)
                      + 5.1021 * np.exp(0.0080 * N_arr)
                      + 0.603025 * N_arr
                      + 14.85225)
    elif region in ['Yangtze River Plain', 'Yangtze_River_Plain', 'YangtzeRiverPlain']:
        N_env_cost = (6.785 * np.exp(0.0060 * N_arr)
                      + 2.30625 * np.exp(0.0078 * N_arr)
                      + 0.94105 * N_arr
                      + 3.37025)
    elif region == 'Southwest':
        N_env_cost = (6.785 * np.exp(0.0060 * N_arr)
                      + 2.30625 * np.exp(0.0078 * N_arr)
                      + 0.94105 * N_arr
                      + 3.37025)
    else:
        N_env_cost = np.zeros_like(N_arr)

    P_env_cost = 0.00247 * P_arr

    revenue = y_pred * price_grain
    cost_fert = N_arr * price_n + P_arr * price_p + K_arr * price_k
    neeb = revenue - cost_fert - N_env_cost - P_env_cost

    top_idx = np.argsort(neeb)[-10:][::-1]
    base_info = {
        'ID': getattr(row, 'ID', None),
        'No': getattr(row, 'No', None),
        'longitude': getattr(row, 'longitude', None),
        'latitude': getattr(row, 'latitude', None)
    }
    out = base_info.copy()
    for rank, pos in enumerate(top_idx, start=1):
        out[f'neeb_{rank}'] = float(neeb[pos])
        out[f'yield_{rank}'] = float(y_pred[pos])
        out[f'N_{rank}'] = int(N_arr[pos])
        out[f'P_{rank}'] = int(P_arr[pos])
        out[f'K_{rank}'] = int(K_arr[pos])
        out[f'N_env_cost_{rank}'] = float(N_env_cost[pos])
        out[f'P_env_cost_{rank}'] = float(P_env_cost[pos])
    records.append(out)

os.makedirs(disk_out_dir, exist_ok=True)
out_df = pd.DataFrame(records)
out_df.to_csv(disk_out_path, index=False)
print(f"Top-10 ESOF optimization results saved to: {disk_out_path}")

os.makedirs(out_dir, exist_ok=True)
out_df.to_csv(out_path, index=False)
print(f"Top-10 ESOF optimization results saved to: {out_path}")
