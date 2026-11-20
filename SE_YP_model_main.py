#!/usr/bin/env python3
import os, sys, time, json, platform, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

DATA_PATH = 'data/data.csv'
SITE_FILE = 'data/site_data.csv'
OUT_DIR = 'results/SE-YP'
USE_GPU = True

SEED = 42
SITE_KEY = 'No.'
N_SPLITS = 5
FEATURE_SETS = ['mundlak']
SCHEMES = ['random', 'site', 'spatial', 'province', 'region', 'year']
YEAR_SPLIT = 2010
SPATIAL_BLOCK_DEG = 2.0
REGION_MIN_SITES = 100
XGB_PARAMS = dict(n_estimators=411, max_depth=12, learning_rate=0.021464882631168122,
                  subsample=0.9479577158696486, colsample_bytree=0.7033846712737518)
XGB_SEARCH_SPACE = {'n_estimators': '100 to 500', 'max_depth': '4 to 12', 'learning_rate': '0.01 to 0.2 (log)',
                    'subsample': '0.6 to 1.0', 'colsample_bytree': '0.6 to 1.0'}
PRICE_GRAIN, PRICE_N, PRICE_P, PRICE_K = 0.376, 0.78, 0.45, 0.91
YIELD_THRESHOLDS = [0, 50, 100, 150, 200, 300, 400]
PROFIT_THRESHOLDS = [0, 10, 20, 30, 50, 75, 100]
STACK_N_JOBS = 1
CLIMATE_MISSING, YIELD_RANGE, DROP_FEATURES = 'drop', None, []

BASE_FEATURES = ['Altitude', 'SOM', 'TN', 'AN', 'AP', 'AK', 'pH', 'irrigation', 'precipitation',
                 'Frost_free_day', 'temperature', 'N', 'P2O5', 'K20', 'Nsurplus']
NUTRIENT_COLS = {'N': 'N', 'P': 'P2O5', 'K': 'K20'}
CLIMATE_COLS = ['precipitation', 'Frost_free_day', 'temperature']

POSTAL_PROVINCE = {
    '01': 'Inner Mongolia', '02': 'Inner Mongolia', '03': 'Shanxi', '04': 'Shanxi',
    '05': 'Hebei', '06': 'Hebei', '07': 'Hebei', '10': 'Beijing', '30': 'Tianjin',
    '11': 'Liaoning', '12': 'Liaoning', '13': 'Jilin', '15': 'Heilongjiang', '16': 'Heilongjiang',
    '20': 'Shanghai', '21': 'Jiangsu', '22': 'Jiangsu', '23': 'Anhui', '24': 'Anhui',
    '25': 'Shandong', '26': 'Shandong', '27': 'Shandong', '31': 'Zhejiang', '32': 'Zhejiang',
    '33': 'Jiangxi', '34': 'Jiangxi', '35': 'Fujian', '36': 'Fujian', '40': 'Chongqing',
    '41': 'Hunan', '42': 'Hunan', '43': 'Hubei', '44': 'Hubei', '45': 'Henan', '46': 'Henan', '47': 'Henan',
    '51': 'Guangdong', '52': 'Guangdong', '53': 'Guangxi', '54': 'Guangxi', '55': 'Guizhou', '56': 'Guizhou',
    '57': 'Hainan', '61': 'Sichuan', '62': 'Sichuan', '63': 'Sichuan', '64': 'Sichuan',
    '65': 'Yunnan', '66': 'Yunnan', '67': 'Yunnan', '71': 'Shaanxi', '72': 'Shaanxi',
    '73': 'Gansu', '74': 'Gansu', '75': 'Ningxia', '81': 'Qinghai', '83': 'Xinjiang', '84': 'Xinjiang',
    '85': 'Tibet', '86': 'Tibet',
}


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)

SITE_FILE_COVARIATES = ['precipitation', 'Frost_free_day', 'temperature', 'Nsurplus']
SITE_FILE_EXTRA = ['Ndep', 'Region', 'Nmin', 'Nmax', 'Pmin', 'Pmax', 'Kmin', 'Kmax']


def load_and_clean(path=DATA_PATH, site_file=SITE_FILE, climate_missing=CLIMATE_MISSING, yield_range=YIELD_RANGE,
                   drop_features=DROP_FEATURES, site_key=SITE_KEY, verbose=True):
    """Load the plot and site files, drop plots without a yield record, and build the covariates, treatment levels and feature sets used by the model."""
    raw = pd.read_csv(path)
    audit = dict(rows_total=int(len(raw)), sites_total=int(raw[site_key].nunique()))
    raw['site'] = raw[site_key]

    for nut, col in NUTRIENT_COLS.items():
        raw[f'{nut}_level'] = (raw.groupby('site')[col].rank(method='dense') - 1).astype(int)
        raw[f'{nut}_nlev'] = raw.groupby('site')[col].transform('nunique')
    raw['regular_3414'] = (raw['N_nlev'] == 4) & (raw['P_nlev'] == 4) & (raw['K_nlev'] == 4)
    audit['sites_regular_3414_design'] = int(raw.groupby('site')['regular_3414'].first().sum())

    raw['year'] = pd.to_numeric(raw['ID'].astype(str).str[7:11], errors='coerce')
    raw['province'] = raw['ID'].astype(str).str[:2].map(POSTAL_PROVINCE).fillna('Other')
    audit['sites_year_unparsable'] = int(raw.loc[raw['year'].isna(), 'site'].nunique())
    audit['years'] = {int(k): int(v) for k, v in raw.groupby('year')['site'].nunique().items()}

    key = raw['longitude'].round(5).astype(str) + '_' + raw['latitude'].round(5).astype(str)
    raw['loc_id'] = pd.factorize(key)[0]
    loc_sites = raw.groupby('loc_id')['site'].nunique()
    audit['distinct_coordinates'] = int(len(loc_sites))
    audit['sites_sharing_coordinates'] = int(loc_sites[loc_sites > 1].sum())

    my = raw['yield'].isna()
    audit['rows_missing_yield'] = int(my.sum())
    n_miss = raw.groupby('site')['yield'].apply(lambda s: int(s.isna().sum()))
    audit['sites_by_n_missing_yield_treatments'] = {int(k): int(v) for k, v in n_miss.value_counts().sort_index().items()}
    data = raw[~my].copy()

    audit['site_file'] = site_file
    if site_file:
        sf = pd.read_csv(site_file).set_index(site_key)
        missing_sites = set(data['site']) - set(sf.index)
        if missing_sites:
            raise KeyError(f'{len(missing_sites)} sites of {path} are not in {site_file}')
        for c in SITE_FILE_COVARIATES + SITE_FILE_EXTRA:
            if c in sf.columns:
                data[c] = data['site'].map(sf[c])
        audit['covariates_from_site_file'] = [c for c in SITE_FILE_COVARIATES + SITE_FILE_EXTRA if c in sf.columns]
        audit['sites_missing_climate'] = int(data.loc[data[CLIMATE_COLS].isna().any(axis=1), 'site'].nunique())
        audit['climate_missing_handling'] = 'site file'
    else:
        mc = data[CLIMATE_COLS].isna().any(axis=1)
        audit['sites_missing_climate'] = int(data.loc[mc, 'site'].nunique())
        audit['climate_missing_handling'] = climate_missing
        if climate_missing == 'drop':
            data = data[~mc].copy()
        elif climate_missing != 'impute':
            raise ValueError('CLIMATE_MISSING must be drop or impute')

    if yield_range is not None:
        lo, hi = yield_range
        bad = (data['yield'] < lo) | (data['yield'] > hi)
        audit['rows_outside_yield_range'] = int(bad.sum())
        data = data[~bad].copy()

    data = pd.get_dummies(data, columns=['crop_type'], dtype=int)
    data = data.replace([np.inf, -np.inf], np.nan)
    audit['sites_missing_Nsurplus'] = int(data.loc[data['Nsurplus'].isna(), 'site'].nunique()) if 'Nsurplus' in data else 0
    num_cols = [c for c in BASE_FEATURES if c in data.columns and pd.api.types.is_numeric_dtype(data[c])]
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    for c in data.select_dtypes(exclude='number').columns:
        if data[c].isna().any():
            data[c] = data[c].fillna(data[c].mode()[0])

    crop_cols = [c for c in data.columns if c.startswith('crop_type_')]
    feats = [f for f in BASE_FEATURES + crop_cols if f not in set(drop_features)]
    for nut, col in NUTRIENT_COLS.items():
        med = raw[raw[f'{nut}_level'] == 2].groupby('site')[col].first()
        data[f'{col}_med'] = data['site'].map(med)
        data[f'{col}_rel'] = (data[col] / data[f'{col}_med'].replace(0, np.nan)).fillna(0)
    mundlak_cols = [f'{c}_med' for c in NUTRIENT_COLS.values()] + [f'{c}_rel' for c in NUTRIENT_COLS.values()]
    data = data.reset_index(drop=True)
    audit['rows_used'] = int(len(data)); audit['sites_used'] = int(data['site'].nunique())
    audit['sites_used_with_14_obs'] = int((data.groupby('site').size() == 14).sum())
    feature_sets = {'original': feats, 'mundlak': feats + mundlak_cols}
    audit['features'] = feature_sets
    if verbose:
        log('data audit: ' + json.dumps({k: v for k, v in audit.items() if k not in ('features',)}, ensure_ascii=False))
    return data, feature_sets, audit


def make_group_folds(groups, n_splits=5, seed=42):
    """Assign whole groups (sites, blocks, provinces) to folds at random and return positional (train, test) index pairs."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    fold_of_group = np.empty(len(uniq), dtype=int)
    fold_of_group[perm] = np.arange(len(uniq)) % n_splits
    gmap = dict(zip(uniq, fold_of_group))
    fold_of_row = np.array([gmap[g] for g in groups])
    return [(np.where(fold_of_row != k)[0], np.where(fold_of_row == k)[0]) for k in range(n_splits)]


def make_random_folds(n, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    fold_of_row = np.empty(n, dtype=int)
    fold_of_row[perm] = np.arange(n) % n_splits
    return [(np.where(fold_of_row != k)[0], np.where(fold_of_row == k)[0]) for k in range(n_splits)]


def leave_one_group_out_folds(groups):
    groups = np.asarray(groups)
    return [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in pd.unique(groups)]


def spatial_block_id(lon, lat, block_deg=2.0):
    bx = np.floor(np.asarray(lon) / block_deg).astype(int)
    by = np.floor(np.asarray(lat) / block_deg).astype(int)
    return pd.factorize(pd.Series(bx).astype(str) + '_' + pd.Series(by).astype(str))[0]


def make_xgb(params, seed=SEED, use_gpu=USE_GPU):
    import xgboost as xgb
    kw = dict(params); kw.update(random_state=seed, verbosity=0, tree_method='hist')
    if use_gpu:
        kw['device'] = 'cuda'
    return xgb.XGBRegressor(**kw)


def make_stack(xgb_params, inner_folds, seed=SEED, use_gpu=USE_GPU, n_jobs=STACK_N_JOBS):
    """Stacking ensemble of XGBoost, LightGBM, random forest and linear regression with a linear meta-learner, fitted with the given inner folds."""
    import lightgbm as lgb
    from sklearn.ensemble import StackingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    ests = [('xgb', make_xgb(xgb_params, seed, use_gpu)),
            ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=seed, verbose=-1)),
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1)),
            ('lr', LinearRegression())]
    return StackingRegressor(estimators=ests, final_estimator=LinearRegression(), cv=list(inner_folds), n_jobs=n_jobs)


def tune_xgb(X, Y, groups, n_trials=50, seed=SEED, use_gpu=USE_GPU, n_splits=N_SPLITS):
    """Optuna search (TPE, site-grouped cross-validation) that selected XGB_PARAMS; not run by main()."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = make_group_folds(groups, n_splits, seed)

    def objective(trial):
        params = dict(n_estimators=trial.suggest_int('n_estimators', 100, 500),
                      max_depth=trial.suggest_int('max_depth', 4, 12),
                      learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                      subsample=trial.suggest_float('subsample', 0.6, 1.0),
                      colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0))
        mses = []
        for tr, te in folds:
            m = make_xgb(params, seed, use_gpu); m.fit(X.iloc[tr], Y.iloc[tr])
            mses.append(mean_squared_error(Y.iloc[te], m.predict(X.iloc[te])))
        return float(np.mean(mses))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def overall_metrics(y_log, p_log):
    y = np.expm1(y_log); p = np.expm1(p_log)
    return dict(n=int(len(y)), R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)), MBE=float(np.mean(p - y)),
                R2_log=r2_score(y_log, p_log), RMSE_log=float(np.sqrt(mean_squared_error(y_log, p_log))))


def within_between_metrics(df, site='site', y='y', p='yhat'):
    """Accuracy between and within sites (R2 of site means, R2 of treatment differences within sites, per-site ranking) on the original yield scale."""
    g = df.groupby(site)
    ybar = g[y].transform('mean'); pbar = g[p].transform('mean')
    yw = df[y] - ybar; pw = df[p] - pbar
    mu = df[y].mean()
    ss_tot = float(((df[y] - mu) ** 2).sum()); ss_between = float(((ybar - mu) ** 2).sum()); ss_within = float((yw ** 2).sum())
    out = dict(n_sites=int(g.ngroups), frac_between=ss_between / ss_tot, frac_within=ss_within / ss_tot,
               R2_total=r2_score(df[y], df[p]), R2_between=r2_score(g[y].mean(), g[p].mean()),
               R2_within=1 - float(((yw - pw) ** 2).sum()) / ss_within,
               R2_sitemean_only=1 - float(((df[y] - pbar) ** 2).sum()) / ss_tot,
               RMSE_within=float(np.sqrt(((yw - pw) ** 2).mean())), SD_within_obs=float(np.sqrt((yw ** 2).mean())),
               within_slope=float(np.polyfit(yw, pw, 1)[0]))

    def _per_site(d):
        if len(d) < 3 or d[y].nunique() < 2 or d[p].nunique() < 2:
            return pd.Series(dict(n=len(d), spearman=np.nan, top1_exact=np.nan, top1_in_top3=np.nan))
        rho = spearmanr(d[y], d[p]).correlation
        obs_rank = d[y].rank(ascending=False); ib = d[p].values.argmax()
        return pd.Series(dict(n=len(d), spearman=rho, top1_exact=float(obs_rank.iloc[ib] == 1),
                              top1_in_top3=float(obs_rank.iloc[ib] <= 3)))

    ps = g[[y, p]].apply(_per_site)
    full = ps[ps['n'] >= 14]
    out['spearman_median'] = float(ps['spearman'].median())
    out['spearman_IQR'] = f"{ps['spearman'].quantile(.25):.2f}–{ps['spearman'].quantile(.75):.2f}"
    out['top1_exact'] = float(ps['top1_exact'].mean()); out['top1_in_obs_top3'] = float(ps['top1_in_top3'].mean())
    out['top1_in_obs_top3_complete_sites'] = float(full['top1_in_top3'].mean()) if len(full) else np.nan
    return out, ps


def estimate_plot_noise(data, y='yield', seed=SEED):
    full = data[data['regular_3414']].groupby('site').filter(lambda s_: len(s_) == 14)
    ss, fitted = [], []
    for _, s_ in full.groupby('site'):
        n, p, k = s_['N'].values, s_['P2O5'].values, s_['K20'].values
        A = np.column_stack([np.ones(14), n, p, k, n ** 2, p ** 2, k ** 2, n * p, n * k, p * k])
        coef = np.linalg.lstsq(A, s_[y].values, rcond=None)[0]
        f = A @ coef; ss.append(((s_[y].values - f) ** 2).sum()); fitted.append(f)
    n_sites = len(ss)
    sigma2 = float(np.sum(ss) / (4 * n_sites)); sigma = float(np.sqrt(sigma2))
    rng = np.random.RandomState(seed); sp, t1, t3 = [], [], []
    for f in fitted:
        yo = f + rng.normal(0, sigma, len(f))
        sp.append(spearmanr(yo, f).correlation)
        r = pd.Series(yo).rank(ascending=False); ib = f.argmax(); t1.append(r.iloc[ib] == 1); t3.append(r.iloc[ib] <= 3)
    return dict(n_sites_used=n_sites, noise_sd=sigma, noise_cv=sigma / float(full[y].mean()),
                ceiling_spearman_median=float(np.median(sp)), ceiling_top1_exact=float(np.mean(t1)),
                ceiling_top1_in_top3=float(np.mean(t3)))


def noise_ceiling_table(data, noise, y='yield', assumed_cv=(0.05, 0.08, 0.10, 0.15)):
    g = data.groupby('site'); ybar = g[y].transform('mean'); n_i = g[y].transform('count')
    ss_within = float(((data[y] - ybar) ** 2).sum())
    rows = [dict(source='empirical (quadratic-surface residual, 4 df/site)', plot_CV=noise['noise_cv'],
                 noise_sd=noise['noise_sd'],
                 R2_within_ceiling=1 - float((noise['noise_sd'] ** 2 * (1 - 1 / n_i)).sum()) / ss_within,
                 ceiling_spearman_median=noise['ceiling_spearman_median'], ceiling_top1_exact=noise['ceiling_top1_exact'],
                 ceiling_top1_in_top3=noise['ceiling_top1_in_top3'])]
    for cv in assumed_cv:
        rows.append(dict(source='assumed', plot_CV=cv, noise_sd=np.nan,
                         R2_within_ceiling=1 - float((((cv * ybar) ** 2) * (1 - 1 / n_i)).sum()) / ss_within))
    return pd.DataFrame(rows)


def add_contrast_ceilings(contrasts, noise_sd):
    c = contrasts.copy()
    var_true = np.clip(c['obs_sd'] ** 2 - 2 * noise_sd ** 2, 0, None)
    c['true_sd_est'] = np.sqrt(var_true)
    c['r_ceiling'] = np.sqrt(var_true) / c['obs_sd']
    c['r_frac_of_ceiling'] = np.where(c['r_ceiling'] > 0.05, c['pearson_r'] / c['r_ceiling'], np.nan)
    return c


def extract_gradients(df, site='site', y='y', p='yhat'):
    d = df[df['regular_3414']]
    out = []
    for nut in ['N', 'P', 'K']:
        o1, o2 = [o for o in ['N', 'P', 'K'] if o != nut]
        sub = d[(d[f'{o1}_level'] == 2) & (d[f'{o2}_level'] == 2)]
        cnt = sub.groupby(site)[f'{nut}_level'].agg(['nunique', 'count'])
        sub = sub[sub[site].isin(cnt[(cnt['nunique'] == 4) & (cnt['count'] == 4)].index)]
        if len(sub) == 0:
            continue
        wy = sub.pivot(index=site, columns=f'{nut}_level', values=y); wy.columns = [f'y{c}' for c in wy.columns]
        wp = sub.pivot(index=site, columns=f'{nut}_level', values=p); wp.columns = [f'p{c}' for c in wp.columns]
        wr = sub.pivot(index=site, columns=f'{nut}_level', values=NUTRIENT_COLS[nut]); wr.columns = [f'rate{c}' for c in wr.columns]
        w = pd.concat([wy, wp, wr], axis=1); w['nutrient'] = nut
        out.append(w.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['site', 'nutrient'])


def gradient_metrics(grad):
    """Observed against predicted yield responses along the N, P2O5 and K2O gradients of the 3414 design."""
    crows, curows = [], []
    contrasts = {'d10': (1, 0), 'd21': (2, 1), 'd32': (3, 2), 'd20': (2, 0), 'd30': (3, 0)}
    for nut, w in grad.groupby('nutrient'):
        w = w.dropna(subset=[f'y{i}' for i in range(4)] + [f'p{i}' for i in range(4)])
        n = int(len(w))
        if n < 3:
            continue
        for name, (a, b) in contrasts.items():
            oy = w[f'y{a}'] - w[f'y{b}']; op = w[f'p{a}'] - w[f'p{b}']
            crows.append(dict(nutrient=nut, n_sites=n, contrast=name, obs_mean=float(oy.mean()), pred_mean=float(op.mean()),
                              obs_sd=float(oy.std()), pred_sd=float(op.std()), pearson_r=pearsonr(oy, op)[0],
                              slope_pred_on_obs=float(np.polyfit(oy, op, 1)[0]),
                              RMSE=float(np.sqrt(((oy - op) ** 2).mean())),
                              sign_agreement=float((np.sign(oy) == np.sign(op)).mean())))
        Y4 = w[[f'y{i}' for i in range(4)]].values; P4 = w[[f'p{i}' for i in range(4)]].values
        Yc = Y4 - Y4.mean(1, keepdims=True); Pc = P4 - P4.mean(1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            curve_r = (Yc * Pc).sum(1) / np.sqrt((Yc ** 2).sum(1) * (Pc ** 2).sum(1))
        am_y = Y4.argmax(1); am_p = P4.argmax(1)
        curows.append(dict(nutrient=nut, n_sites=n, curve_r_median=float(np.nanmedian(curve_r)),
                           curve_r_q25=float(np.nanquantile(curve_r, .25)), curve_r_q75=float(np.nanquantile(curve_r, .75)),
                           argmax_exact=float((am_y == am_p).mean()), argmax_within1=float((np.abs(am_y - am_p) <= 1).mean()),
                           obs_argmax_at_top=float((am_y == 3).mean()), pred_argmax_at_top=float((am_p == 3).mean())))
    return pd.DataFrame(crows), pd.DataFrame(curows)


def env_versions():
    out = dict(python=platform.python_version())
    for m in ['numpy', 'pandas', 'sklearn', 'xgboost', 'lightgbm', 'optuna', 'shap', 'scipy']:
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def analyse(df):
    """Observed yield and profit of the treatment that the model selects at each held-out site, compared with the medium rate."""
    df = df.copy()
    df['nfb_obs'] = df['y'] * PRICE_GRAIN - df['N'] * PRICE_N - df['P2O5'] * PRICE_P - df['K20'] * PRICE_K
    df['nfb_pred'] = df['yhat'] * PRICE_GRAIN - df['N'] * PRICE_N - df['P2O5'] * PRICE_P - df['K20'] * PRICE_K
    df['trt'] = df['N_level'].astype(str) + df['P_level'].astype(str) + df['K_level'].astype(str)
    full = df.groupby('site').filter(lambda s: len(s) == 14 and s['trt'].nunique() == 14)
    if len(full) == 0:
        return None, None
    full['yfrac'] = full['y'] / full.groupby('site')['y'].transform('max')
    grp = 'province' if 'province' in full else None
    reg_y = full.groupby([grp, 'trt'])['yfrac'].mean().groupby(grp).idxmax().apply(lambda t: t[1]) if grp else None
    reg_p = full.groupby([grp, 'trt'])['nfb_obs'].mean().groupby(grp).idxmax().apply(lambda t: t[1]) if grp else None

    piv = lambda col: full.pivot(index='site', columns='trt', values=col)
    Y, P, YH, PH, NPK = piv('y'), piv('nfb_obs'), piv('yhat'), piv('nfb_pred'), piv('N') + piv('P2O5') + piv('K20')
    trts = list(Y.columns); sites = Y.index
    prov = full.drop_duplicates('site').set_index('site').loc[sites, grp] if grp else None
    col = {t: i for i, t in enumerate(trts)}
    Yv, Pv, YHv, PHv, NPKv = Y.values, P.values, YH.values, PH.values, NPK.values
    med = np.full(len(sites), col['222'])
    idx = np.arange(len(sites))

    def summarise(name, pick, objective):
        O_ = Yv if objective == 'yield' else Pv
        v_ = O_[idx, pick][:, None]
        avg_rank = (O_ > v_).sum(1) + ((O_ == v_).sum(1) + 1) / 2
        return dict(rule=name, objective=objective, n_sites=len(sites), obs_top3=float((avg_rank <= 3).mean()),
                    frac_deviating_from_medium=float((pick != med).mean()),
                    yield_obs=float(Yv[idx, pick].mean()), yield_frac_of_site_max=float((Yv[idx, pick] / Yv.max(1)).mean()),
                    profit_obs=float(Pv[idx, pick].mean()), npk_total=float(NPKv[idx, pick].mean()),
                    d_yield_vs_medium=float((Yv[idx, pick] - Yv[idx, med]).mean()),
                    d_profit_vs_medium=float((Pv[idx, pick] - Pv[idx, med]).mean()),
                    better_than_medium_when_deviating=(
                        float((((Yv if objective == 'yield' else Pv)[idx, pick] - (Yv if objective == 'yield' else Pv)[idx, med]) > 0)[pick != med].mean())
                        if (pick != med).any() else np.nan))

    rows, thr_rows = [], []
    for objective, Ov, Mv, thresholds in [('yield', Yv, YHv, YIELD_THRESHOLDS), ('profit', Pv, PHv, PROFIT_THRESHOLDS)]:
        rows.append(summarise('model argmax', Mv.argmax(1), objective))
        rows.append(summarise('rule: medium N2P2K2', med, objective))
        if grp:
            rb = reg_y if objective == 'yield' else reg_p
            rows.append(summarise('rule: regional best treatment', np.array([col[rb[p]] for p in prov]), objective))
        rows.append(summarise('rule: high N (N3P2K2)', np.full(len(sites), col['322']), objective))
        rows.append(summarise('rule: zero fertilizer', np.full(len(sites), col['000']), objective))
        rows.append(summarise('oracle: observed best', Ov.argmax(1), objective))
        gain = Mv.max(1) - Mv[idx, med]
        for t in thresholds:
            pick = np.where(gain > t, Mv.argmax(1), med)
            r = summarise(f'model if predicted gain > {t}', pick, objective); r['threshold'] = t
            thr_rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(thr_rows)

def make_figures(oof_store, summary, out):
    fig_dir = f'{out}/figures'
    model = 'stack' if any(k[2] == 'stack' for k in oof_store) else 'xgb'
    for fset in sorted({k[1] for k in oof_store}):
        schemes = [s for s in ['random', 'site', 'location', 'spatial', 'region', 'province', 'year_holdout', 'year_loo']
                   if (s, fset, model) in oof_store]
        if not schemes:
            continue
        fig, axes = plt.subplots(1, len(schemes), figsize=(4.2 * len(schemes), 4.2), squeeze=False)
        for ax, s in zip(axes[0], schemes):
            df = oof_store[(s, fset, model)]
            ax.hexbin(df['y'], df['yhat'], gridsize=60, bins='log', cmap='viridis')
            lim = [0, max(df['y'].max(), df['yhat'].max())]; ax.plot(lim, lim, 'r--', lw=1)
            r = summary[(summary.scheme == s) & (summary.features == fset) & (summary.model == model)].iloc[0]
            ax.set_title(f'{s} ({fset})\nR²={r.R2:.3f}  RMSE={r.RMSE:.0f}  R²_within={r.R2_within:.2f}')
            ax.set_xlabel('observed yield (kg/ha)'); ax.set_ylabel('predicted yield (kg/ha)')
        plt.tight_layout(); plt.savefig(f'{fig_dir}/obs_vs_pred_{fset}.png', dpi=150); plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        for s in schemes:
            f = f'{out}/oof/per_site_{s}_{fset}_{model}.csv'
            if os.path.exists(f):
                ps = pd.read_csv(f); ax.hist(ps['spearman'].dropna(), bins=40, histtype='step', lw=1.5, density=True, label=s)
        ax.set_xlabel('per-site Spearman ρ over treatments (observed vs predicted)'); ax.set_ylabel('density'); ax.legend()
        plt.tight_layout(); plt.savefig(f'{fig_dir}/site_spearman_{fset}.png', dpi=150); plt.close()

        comp = [s for s in ['random', 'site', 'location'] if s in schemes]
        if comp:
            fig, axes = plt.subplots(len(comp), 3, figsize=(12, 4 * len(comp)), squeeze=False)
            for i, s in enumerate(comp):
                g = extract_gradients(oof_store[(s, fset, model)])
                for j, nut in enumerate(['N', 'P', 'K']):
                    w = g[g.nutrient == nut].dropna(subset=['y2', 'y0', 'p2', 'p0']) if len(g) else g
                    ax = axes[i, j]
                    if len(w) < 3: ax.axis('off'); continue
                    oy = w['y2'] - w['y0']; op = w['p2'] - w['p0']
                    ax.hexbin(oy, op, gridsize=50, bins='log', cmap='magma')
                    lim = [min(oy.min(), op.min()), max(oy.max(), op.max())]
                    ax.plot(lim, lim, 'c--', lw=1); ax.axhline(0, color='w', lw=.5); ax.axvline(0, color='w', lw=.5)
                    ax.set_title(f'{s}: {nut} response (medium − zero)\nr={pearsonr(oy, op)[0]:.2f} slope={np.polyfit(oy, op, 1)[0]:.2f} n={len(w)}')
                    ax.set_xlabel('observed Δyield'); ax.set_ylabel('predicted Δyield')
            plt.tight_layout(); plt.savefig(f'{fig_dir}/response_contrasts_{fset}.png', dpi=150); plt.close()

        key = ('site', fset, model) if ('site', fset, model) in oof_store else (schemes[0], fset, model)
        g = extract_gradients(oof_store[key])
        if len(g):
            sites = g[g.nutrient == 'N'].dropna()['site'].drop_duplicates()
            sites = sites.sample(min(6, len(sites)), random_state=1).values
            fig, axes = plt.subplots(len(sites), 3, figsize=(11, 2.6 * len(sites)), squeeze=False)
            for i, st in enumerate(sites):
                for j, nut in enumerate(['N', 'P', 'K']):
                    w = g[(g.nutrient == nut) & (g.site == st)]; ax = axes[i, j]
                    if len(w) == 0: ax.axis('off'); continue
                    w = w.iloc[0]; rates = [w[f'rate{k}'] for k in range(4)]
                    ax.plot(rates, [w[f'y{k}'] for k in range(4)], 'o-', label='observed')
                    ax.plot(rates, [w[f'p{k}'] for k in range(4)], 's--', label=f'predicted ({key[0]}, {fset})')
                    ax.set_xlabel(f'{NUTRIENT_COLS[nut]} rate (kg/ha)'); ax.set_ylabel('yield (kg/ha)')
                    if i == 0: ax.set_title(f'{nut} gradient (others at medium)')


def decision_value_figure(thr, out):
    sub = thr[thr.scheme == 'site'] if (thr.scheme == 'site').any() else thr
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, objective, ycol, unit in [(axes[0], 'yield', 'd_yield_vs_medium', 'kg/ha'), (axes[1], 'profit', 'd_profit_vs_medium', '$/ha')]:
        for (fset, model), d in sub[sub.objective == objective].groupby(['features', 'model']):
            ax.plot(d['threshold'], d[ycol], 'o-', label=f'{fset} / {model}')
        ax.axhline(0, color='k', lw=.8)
        ax.set_xlabel(f'predicted gain required to leave medium ({unit})'); ax.set_ylabel(f'observed gain vs medium rule ({unit})')
        ax.set_title(f'{objective}: model pick vs medium N2P2K2 (held-out sites)'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f'{out}/figures/decision_value_thresholds.png', dpi=150); plt.close()


def decision_tables(oof_store, out):
    all_rows, all_thr = [], []
    for (scheme, fset, mdl), df in oof_store.items():
        res, thr = analyse(df)
        if res is None: continue
        for t in (res, thr):
            t.insert(0, 'model', mdl); t.insert(0, 'features', fset); t.insert(0, 'scheme', scheme)
        all_rows.append(res); all_thr.append(thr)
    if not all_rows:
        return
    res = pd.concat(all_rows, ignore_index=True); thr = pd.concat(all_thr, ignore_index=True)
    res.to_csv(f'{out}/tables/decision_value_all.csv', index=False)
    thr.to_csv(f'{out}/tables/decision_value_thresholds.csv', index=False)
    s10 = res[(res.scheme == 'site')][['features', 'objective', 'rule', 'n_sites', 'frac_deviating_from_medium', 'yield_obs',
                                      'yield_frac_of_site_max', 'obs_top3', 'profit_obs', 'npk_total', 'd_yield_vs_medium', 'd_profit_vs_medium']]
    s10.to_csv(f'{out}/tables/table_S10_decision_value.csv', index=False)
    decision_value_figure(thr, out)
    log('Table S10 written:\n' + s10.round(3).to_string(index=False))


def write_fold_and_matrix(data, feature_sets, groups_site, fold_id_site):
    pd.DataFrame({'site': groups_site, 'row': np.arange(len(data)), 'fold_site': fold_id_site}).to_csv(f'{OUT_DIR}/oof/fold_assignment_site.csv', index=False)
    for fset in FEATURE_SETS:
        mm = data[feature_sets[fset]].copy()
        mm.insert(0, 'fold_site', fold_id_site); mm.insert(0, 'yield', data['yield'].values); mm.insert(0, 'site', groups_site)
        mm.to_csv(f'{OUT_DIR}/oof/model_matrix_{fset}.csv', index=False)
        log(f'written oof/model_matrix_{fset}.csv ({len(mm):,} plots, {len(feature_sets[fset])} covariates)')


SCHEME_LABEL = {'random': 'Random split of plots (reference)', 'site': 'Whole sites held out',
                'spatial': '2-degree spatial blocks held out', 'province': 'Provinces held out',
                'region': 'Regions held out', 'year_holdout': 'Years held out (train 2005-2010, test 2011-2014)',
                'year_loo': 'Leave-one-year-out', 'location': 'Coordinates held out'}


def main():
    """Validate the stacking ensemble under the six splits, write the tables and figures, and save the fold models and the final model fitted on all plots."""
    t0 = time.time()
    for d in ['audit', 'oof', 'tables', 'models', 'figures']:
        os.makedirs(f'{OUT_DIR}/{d}', exist_ok=True)

    data, feature_sets, audit = load_and_clean()
    Y = np.log1p(data['yield']); groups_site = data['site'].values
    save_json(audit, f'{OUT_DIR}/audit/data_audit.json')
    log(f'{len(data):,} rows, {data["site"].nunique():,} sites; feature sets: ' +
        '; '.join(f'{k} ({len(v)})' for k, v in feature_sets.items()))
    diag = []
    for f in feature_sets['mundlak']:
        nun = data.groupby('site')[f].nunique()
        diag.append(dict(feature=f, frac_sites_constant=float((nun == 1).mean()),
                         note='treatment variable' if f in NUTRIENT_COLS.values() else
                         ('site-level' if (nun == 1).mean() > 0.95 else 'VARIES WITHIN SITE - check')))
    pd.DataFrame(diag).to_csv(f'{OUT_DIR}/audit/feature_site_level_diagnostic.csv', index=False)
    noise = estimate_plot_noise(data)
    noise_ceiling_table(data, noise).to_csv(f'{OUT_DIR}/tables/noise_ceiling.csv', index=False)
    log(f'plot noise estimate: SD {noise["noise_sd"]:.0f} kg/ha (CV {noise["noise_cv"]:.3f}) from {noise["n_sites_used"]:,} complete sites')

    xgb_params = dict(XGB_PARAMS); log(f'XGB hyper-parameters (Table S7): {xgb_params}')
    pd.DataFrame([dict(parameter=k, search_space=XGB_SEARCH_SPACE.get(k, ''), selected=v) for k, v in xgb_params.items()] +
                 [dict(parameter='LightGBM', search_space='fixed', selected='n_estimators 200, max_depth 8, learning_rate 0.05'),
                  dict(parameter='Random forest', search_space='fixed', selected='n_estimators 200, max_depth 8'),
                  dict(parameter='Linear regression', search_space='none', selected='ordinary least squares'),
                  dict(parameter='Meta-learner', search_space='none', selected='linear regression on out-of-fold predictions'),
                  dict(parameter='Tuning', search_space='Optuna TPE, 50 trials', selected=f'site-grouped {N_SPLITS}-fold CV, MSE of log(1 + yield)'),
                  dict(parameter='Seeds', search_space='', selected='folds and learners 42; inner folds of outer fold k 142 + k')]
                 ).to_csv(f'{OUT_DIR}/tables/table_S7_hyperparameters.csv', index=False)

    scheme_folds = {}
    if 'random' in SCHEMES: scheme_folds['random'] = make_random_folds(len(data), N_SPLITS, SEED)
    if 'site' in SCHEMES: scheme_folds['site'] = make_group_folds(groups_site, N_SPLITS, SEED)
    if 'spatial' in SCHEMES:
        data['spatial_block'] = spatial_block_id(data['longitude'], data['latitude'], SPATIAL_BLOCK_DEG)
        log(f'spatial blocks of {SPATIAL_BLOCK_DEG} deg: {data["spatial_block"].nunique()}')
        scheme_folds['spatial'] = make_group_folds(data['spatial_block'].values, N_SPLITS, SEED)
    if 'province' in SCHEMES:
        reg = data['province'].copy(); cnt = data.groupby(reg)['site'].nunique()
        reg = reg.where(~reg.isin(cnt[cnt < REGION_MIN_SITES].index), 'Other'); data['province_grp'] = reg.values
        scheme_folds['province'] = leave_one_group_out_folds(data['province_grp'].values)
    if 'region' in SCHEMES:
        data['region'] = data['Region'].values if 'Region' in data else (data['province_grp'].values if 'province_grp' in data else data['province'].values)
        scheme_folds['region'] = leave_one_group_out_folds(data['region'].values)
    if 'year' in SCHEMES:
        yr = data['year'].values; ok = ~np.isnan(yr)
        tr = np.where(ok & (yr <= YEAR_SPLIT))[0]; te = np.where(ok & (yr > YEAR_SPLIT))[0]
        scheme_folds['year_holdout'] = [(tr, te)]
    fold_id_site = np.full(len(data), -1)
    if 'site' in scheme_folds:
        for k, (tr, te) in enumerate(scheme_folds['site']): fold_id_site[te] = k
    write_fold_and_matrix(data, feature_sets, groups_site, fold_id_site)

    summary, fold_rows, grad_rows, curve_rows, region_rows, year_rows = [], [], [], [], [], []
    oof_store = {}
    model = 'stack'
    for scheme, folds in scheme_folds.items():
        for fset in FEATURE_SETS:
            feats = feature_sets[fset]; X = data[feats]; tag = f'{scheme}_{fset}_{model}'
            log(f'=== scheme={scheme} features={fset} ({len(folds)} folds) ===')
            oof = np.full(len(data), np.nan); fold_id = np.full(len(data), -1)
            for k, (tr, te) in enumerate(folds):
                tk = time.time()
                inner = make_group_folds(groups_site[tr], N_SPLITS, SEED + 100 + k)
                m = make_stack(xgb_params, inner); m.fit(X.iloc[tr], Y.iloc[tr])
                if scheme == 'site':
                    joblib.dump(m, f'{OUT_DIR}/models/stack_site_{fset}_fold{k}.pkl')
                pred = m.predict(X.iloc[te]); oof[te] = pred; fold_id[te] = k
                fm = overall_metrics(Y.iloc[te].values, pred)
                fold_rows.append(dict(scheme=scheme, features=fset, model=model, fold=k, n_train=len(tr), **fm))
                log(f'  fold {k}: R2={fm["R2"]:.3f} RMSE={fm["RMSE"]:.0f} (n_test={len(te):,}, {time.time() - tk:.0f}s)')
            has = ~np.isnan(oof)
            keep = ['site', 'ID', 'longitude', 'latitude', 'year', 'province', 'N', 'P2O5', 'K20',
                    'N_level', 'P_level', 'K_level', 'regular_3414'] + \
                   [c for c in ['region', 'province_grp', 'Region', 'Ndep', 'Nmax', 'Pmax', 'Kmax', 'N_med', 'P2O5_med', 'K20_med'] if c in data]
            df = data.loc[has, keep].copy()
            df['fold'] = fold_id[has]; df['y_log'] = Y[has].values; df['yhat_log'] = oof[has]
            df['y'] = np.expm1(df['y_log']); df['yhat'] = np.expm1(df['yhat_log'])
            df.to_csv(f'{OUT_DIR}/oof/oof_{tag}.csv', index=False)
            oof_store[(scheme, fset, model)] = df
            om = overall_metrics(df['y_log'].values, df['yhat_log'].values)
            wb, per_site = within_between_metrics(df)
            per_site.to_csv(f'{OUT_DIR}/oof/per_site_{tag}.csv')
            fr = pd.DataFrame([r for r in fold_rows if r['scheme'] == scheme and r['features'] == fset])
            summary.append(dict(scheme=scheme, features=fset, model=model, n_folds=len(folds),
                                R2_fold_mean=fr['R2'].mean(), R2_fold_sd=fr['R2'].std(), **om, **wb))
            gc, gu = gradient_metrics(extract_gradients(df))
            gc = add_contrast_ceilings(gc, noise['noise_sd']) if len(gc) else gc
            for t in (gc, gu):
                t.insert(0, 'model', model); t.insert(0, 'features', fset); t.insert(0, 'scheme', scheme)
            grad_rows.append(gc); curve_rows.append(gu)
            log(f'  ALL: R2={om["R2"]:.3f} RMSE={om["RMSE"]:.0f} | R2_between={wb["R2_between"]:.3f} R2_within={wb["R2_within"]:.3f} '
                f'| spearman_med={wb["spearman_median"]:.2f} top1_in_top3={wb["top1_in_obs_top3"]:.2f}')
            pd.DataFrame(summary).to_csv(f'{OUT_DIR}/tables/summary_all.csv', index=False)
            if scheme in ('region', 'province'):
                for r_, d_ in df.groupby('region' if scheme == 'region' else 'province_grp'):
                    region_rows.append(dict(scheme=scheme, features=fset, model=model, region=r_, n_sites=d_['site'].nunique(),
                                            **overall_metrics(d_['y_log'].values, d_['yhat_log'].values),
                                            R2_within=within_between_metrics(d_)[0]['R2_within']))
            if scheme.startswith('year'):
                for y_, d_ in df.groupby('year'):
                    year_rows.append(dict(scheme=scheme, features=fset, model=model, year=int(y_), n_sites=d_['site'].nunique(),
                                          **overall_metrics(d_['y_log'].values, d_['yhat_log'].values),
                                          R2_within=within_between_metrics(d_)[0]['R2_within']))

    summary = pd.DataFrame(summary); summary.to_csv(f'{OUT_DIR}/tables/summary_all.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(f'{OUT_DIR}/tables/fold_metrics.csv', index=False)
    gc_all = pd.concat(grad_rows, ignore_index=True); gu_all = pd.concat(curve_rows, ignore_index=True)
    gc_all.to_csv(f'{OUT_DIR}/tables/gradient_contrasts_all.csv', index=False)
    gu_all.to_csv(f'{OUT_DIR}/tables/gradient_curves_all.csv', index=False)
    if region_rows: pd.DataFrame(region_rows).to_csv(f'{OUT_DIR}/tables/region_metrics.csv', index=False)
    if year_rows: pd.DataFrame(year_rows).to_csv(f'{OUT_DIR}/tables/year_metrics.csv', index=False)
    order = ['random', 'site', 'spatial', 'province', 'region', 'year_holdout']
    s8 = summary[summary.scheme.isin(order)].copy()
    s8['Split'] = s8['scheme'].map(SCHEME_LABEL)
    s8 = s8.set_index('scheme').loc[[o for o in order if o in set(s8.scheme)]].reset_index()
    s8 = s8[['Split', 'features', 'n', 'R2', 'RMSE', 'R2_between', 'R2_within', 'spearman_median', 'top1_in_obs_top3', 'top1_in_obs_top3_complete_sites']]
    s8.columns = ['Split', 'Features', 'Plots', 'R2', 'RMSE (kg/ha)', 'R2 between', 'R2 within', 'Median Spearman rho', 'Top-3 (sites >= 3 plots)', 'Top-3 (complete sites)']
    s8.to_csv(f'{OUT_DIR}/tables/table_S8_validation.csv', index=False)
    site_c = gc_all[(gc_all.scheme == 'site')]; site_u = gu_all[(gu_all.scheme == 'site')]
    rows = []
    for fset in FEATURE_SETS:
        for nut, name in [('N', 'N'), ('P', 'P2O5'), ('K', 'K2O')]:
            c = site_c[(site_c.features == fset) & (site_c.nutrient == nut)].set_index('contrast')
            u = site_u[(site_u.features == fset) & (site_u.nutrient == nut)].iloc[0]
            rows.append(dict(features=fset, nutrient=name, sites=int(u['n_sites']),
                             zero_to_medium_obs=c.loc['d20', 'obs_mean'], zero_to_medium_pred=c.loc['d20', 'pred_mean'],
                             zero_to_medium_r=c.loc['d20', 'pearson_r'], zero_to_medium_sign_agreement=c.loc['d20', 'sign_agreement'],
                             medium_to_high_obs=c.loc['d32', 'obs_mean'], medium_to_high_pred=c.loc['d32', 'pred_mean'],
                             curve_r_median=u['curve_r_median'], obs_argmax_at_top=u['obs_argmax_at_top'], pred_argmax_at_top=u['pred_argmax_at_top']))
    pd.DataFrame(rows).to_csv(f'{OUT_DIR}/tables/table_S9_response_heldout.csv', index=False)

    decision_tables(oof_store, OUT_DIR)

    make_figures(oof_store, summary, OUT_DIR)

    for fset in FEATURE_SETS:
        feats = feature_sets[fset]
        inner = make_group_folds(groups_site, N_SPLITS, SEED + 100)
        final = make_stack(xgb_params, inner); final.fit(data[feats], Y)
        joblib.dump(final, f'{OUT_DIR}/models/SE_YP_model.pkl')
        save_json(feats, f'{OUT_DIR}/models/feature_list_{fset}.json')
        log(f'final stacking model ({fset}, {len(feats)} features) fitted on all {len(data):,} plots and saved')

    save_json(dict(data=DATA_PATH, site_file=SITE_FILE, out_dir=OUT_DIR, seed=SEED, site_key=SITE_KEY,
                   year_split=YEAR_SPLIT, spatial_block_deg=SPATIAL_BLOCK_DEG, region_min_sites=REGION_MIN_SITES,
                   schemes=list(scheme_folds), feature_sets={k: feature_sets[k] for k in FEATURE_SETS}, n_splits=N_SPLITS,
                   xgb_params=xgb_params, use_gpu=USE_GPU, plot_noise=noise,
                   stack=dict(base=['xgb(tuned)', 'lgb(n_estimators=200, lr=0.05, max_depth=8)', 'rf(n_estimators=200, max_depth=8)', 'lr'],
                              meta='LinearRegression', oof='site-grouped 5-fold inside each outer training fold'),
                   target='log1p(yield); metrics on expm1 scale unless *_log', prices=dict(grain=PRICE_GRAIN, N=PRICE_N, P2O5=PRICE_P, K2O=PRICE_K),
                   versions=env_versions(), runtime_min=(time.time() - t0) / 60, audit=audit),
              f'{OUT_DIR}/run_config.json')
    log('\n' + s8.round(3).to_string(index=False))
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {OUT_DIR}')


if __name__ == '__main__':
    main()
